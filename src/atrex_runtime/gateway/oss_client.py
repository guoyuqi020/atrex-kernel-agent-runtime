"""Move oversized Dev file maps to Agate's public OSS upload transport."""

from __future__ import annotations

import hashlib
import logging
import shlex
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Protocol, cast

from . import oss_remote

AGATE_MAX_INLINE_DEV_BYTES = 4 * 1024 * 1024
_LOGGER = logging.getLogger(__name__)


class _UploadClient(Protocol):
    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]: ...

    def prepare_uploads(
        self, gpu: str, files: list[dict[str, object]], *, kind: str
    ) -> dict[str, object]: ...

    def upload_file(self, url: str, path: str) -> None: ...


class OssAgateClient:
    """Adapt only large Dev submissions; identities remain based on logical files.

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
