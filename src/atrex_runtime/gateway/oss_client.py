"""Upload native Eval source archives and oversized Dev files through Agate OSS."""

from __future__ import annotations

import gzip
import hashlib
import io
import logging
import shlex
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Protocol, cast

from . import oss_remote
from .eval_sources import EVAL_ARCHIVES_KEY

AGATE_MAX_INLINE_DEV_BYTES = 4 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


class _UploadClient(Protocol):
    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]: ...

    def prepare_uploads(
        self, gpu: str, files: list[dict[str, object]], *, kind: str
    ) -> dict[str, object]: ...

    def upload_file(self, url: str, path: str) -> None: ...


class OssAgateClient:
    """Stage sealed Eval archives or large Dev files without changing logical identities.

    The wrapped client retries each prepare, PUT and submission independently.
    Submission retries reuse the already uploaded reference rather than uploading
    again. Terminal-job replacement receives a new reservation, as in Agate CLI.
    """

    def __init__(self, client: object, *, max_inline_bytes: int = AGATE_MAX_INLINE_DEV_BYTES):
        if max_inline_bytes <= 0:
            raise ValueError("Agate inline byte limit must be positive")
        self._client = cast(_UploadClient, client)
        self._max_inline_bytes = max_inline_bytes

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        if EVAL_ARCHIVES_KEY in request:
            if kind != "eval":
                raise ValueError("native source archives can only be submitted through Eval")
            return self._submit_eval(request)
        files = request.get("files")
        if kind != "dev" or not isinstance(files, dict) or not files:
            return self._client.submit_job(kind, request)
        # Leave malformed wire payloads to the Gateway's schema validation.
        if not all(isinstance(path, str) and isinstance(text, str) for path, text in files.items()):
            return self._client.submit_job(kind, request)
        texts = cast(dict[str, str], files)
        decoded_bytes = sum(len(text.encode("utf-8")) for text in texts.values())
        if decoded_bytes <= self._max_inline_bytes:
            return self._client.submit_job(kind, request)
        command = request.get("command")
        spec = request.get("spec")
        targets = spec.get("target_hardware") if isinstance(spec, dict) else None
        if not isinstance(command, str) or not command.strip():
            raise ValueError("OSS Dev transport requires a nonempty command")
        if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], str):
            raise ValueError("OSS Dev transport requires exactly one target hardware")
        existing = request.get("oss_files", [])
        if not isinstance(existing, list):
            raise ValueError("OSS Dev transport requires oss_files to be a list")
        for path in texts:
            oss_remote.validate_path(path)
        with tempfile.TemporaryDirectory(prefix="atrex-agate-oss-") as directory:
            archive = Path(directory) / "files.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                for path, text in sorted(texts.items()):
                    info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
                    info.create_system = 3
                    info.external_attr = 0o100644 << 16
                    bundle.writestr(info, text.encode("utf-8"), zipfile.ZIP_DEFLATED, 9)
            with archive.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            archive_path = f"__atrex_oss_payload_{digest}.zip"
            driver_path = f"__atrex_oss_unpack_{digest}.py"
            occupied = set(texts) | {
                entry.get("path")
                for entry in existing
                if isinstance(entry, dict) and isinstance(entry.get("path"), str)
            }
            if archive_path in occupied or driver_path in occupied:
                raise ValueError("OSS transport staging paths conflict with request files")
            reservation = self._client.prepare_uploads(
                targets[0],
                [{"path": archive_path, "bytes": archive.stat().st_size, "sha256": digest}],
                kind=kind,
            )
            uploads = reservation.get("uploads")
            if not reservation.get("job_id") or not isinstance(uploads, list) or len(uploads) != 1:
                raise ValueError("Agate OSS reservation must contain a job_id and one upload")
            upload = uploads[0]
            if (
                not isinstance(upload, dict)
                or upload.get("path") != archive_path
                or not isinstance(upload.get("put_url"), str)
                or not upload.get("put_url")
                or not upload.get("upload_ref")
            ):
                raise ValueError(
                    "Agate OSS reservation has an invalid path, URL or upload reference"
                )
            self._client.upload_file(upload["put_url"], str(archive))
            wire = {
                **request,
                "files": {driver_path: Path(oss_remote.__file__).read_text(encoding="utf-8")},
                "oss_files": [
                    *existing,
                    {"path": archive_path, "upload_ref": upload["upload_ref"]},
                ],
                "command": (
                    f"python3 {shlex.quote(driver_path)} {shlex.quote(archive_path)} {digest} "
                    f"&& (\n{command}\n)"
                ),
            }
            _LOGGER.info(
                "Agate Dev OSS transport: %d files, %d decoded bytes, %d packed bytes",
                len(texts),
                decoded_bytes,
                archive.stat().st_size,
            )
            return self._client.submit_job(kind, wire)

    def _submit_eval(self, request: dict[str, object]) -> dict[str, object]:
        archives = request[EVAL_ARCHIVES_KEY]
        if not isinstance(archives, dict) or not archives:
            raise ValueError("native Eval requires nonempty source archives")
        spec = request.get("spec")
        targets = spec.get("target_hardware") if isinstance(spec, dict) else None
        if not isinstance(targets, list) or len(targets) != 1 or not isinstance(targets[0], str):
            raise ValueError("OSS Eval transport requires exactly one target hardware")
        existing = request.get("oss_files", [])
        if not isinstance(existing, list):
            raise ValueError("OSS Eval transport requires oss_files to be a list")
        if existing:
            raise ValueError(
                "native Eval cannot mix existing OSS references with a new upload reservation"
            )
        for archive_path, files in archives.items():
            oss_remote.validate_path(archive_path)
            if not isinstance(files, dict) or not files:
                raise ValueError("native Eval archive must contain source files")
            for path, content in files.items():
                oss_remote.validate_path(path)
                if not isinstance(content, str):
                    raise ValueError("native Eval source files must be UTF-8 text")
        abba = request.get("abba")
        sources = [request.get("candidate")]
        if isinstance(abba, dict):
            sources.append(abba.get("baseline"))
        referenced = set()
        for source in sources:
            if isinstance(source, dict):
                archive_path, entry = source.get("archive"), source.get("entry_point")
                if not isinstance(archive_path, str) or archive_path not in archives:
                    raise ValueError("native Eval source must reference its sealed archive")
                if not isinstance(entry, str) or entry not in archives[archive_path]:
                    raise ValueError("native Eval entry_point is missing from source archive")
                referenced.add(archive_path)
        if referenced != set(archives):
            raise ValueError("native Eval contains unreferenced source archives")

        with tempfile.TemporaryDirectory(prefix="atrex-agate-eval-") as directory:
            staged = []
            descriptors: list[dict[str, object]] = []
            for index, (archive_path, files) in enumerate(sorted(archives.items())):
                archive = Path(directory) / f"{index}.tar.gz"
                # Stable bytes across retries, independent of local paths, clocks and UID.
                with (
                    archive.open("wb") as output,
                    gzip.GzipFile(filename="", mode="wb", fileobj=output, mtime=0) as gz,
                    tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar,
                ):
                    for path, content in sorted(files.items()):
                        data = content.encode("utf-8")
                        info = tarfile.TarInfo(path)
                        info.size, info.mode = len(data), 0o644
                        tar.addfile(info, io.BytesIO(data))
                with archive.open("rb") as stream:
                    digest = hashlib.file_digest(stream, "sha256").hexdigest()
                descriptors.append(
                    {"path": archive_path, "bytes": archive.stat().st_size, "sha256": digest}
                )
                staged.append(archive)
            reservation = self._client.prepare_uploads(targets[0], descriptors, kind="eval")
            uploads = reservation.get("uploads")
            if (
                not reservation.get("job_id")
                or not isinstance(uploads, list)
                or len(uploads) != len(staged)
            ):
                raise ValueError("Agate Eval reservation must contain a job_id and every upload")
            # Validate the complete response before uploading either side of an ABBA pair.
            for descriptor, upload in zip(descriptors, uploads, strict=True):
                if (
                    not isinstance(upload, dict)
                    or upload.get("path") != descriptor["path"]
                    or not isinstance(upload.get("put_url"), str)
                    or not upload.get("put_url")
                    or not upload.get("upload_ref")
                ):
                    raise ValueError("Agate Eval reservation has an invalid path, URL or reference")
            attachments = list(existing)
            for archive, upload in zip(staged, uploads, strict=True):
                self._client.upload_file(upload["put_url"], str(archive))
                attachments.append({"path": upload["path"], "upload_ref": upload["upload_ref"]})
            wire = {key: value for key, value in request.items() if key != EVAL_ARCHIVES_KEY}
            wire["oss_files"] = attachments
            return self._client.submit_job("eval", wire)
