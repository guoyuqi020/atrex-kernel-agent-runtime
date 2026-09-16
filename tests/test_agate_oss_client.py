"""Agate OSS transport preserves exact file maps without oversized inline requests."""

from __future__ import annotations

import hashlib
import io
import json
import stat
import subprocess
import zipfile
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Thread
from typing import cast

import pytest

from atrex_runtime.gateway import oss_remote
from atrex_runtime.gateway.agate import AgateConnectionConfig, load_agate_sdk
from atrex_runtime.gateway.oss_client import OssAgateClient
from atrex_runtime.gateway.retrying_client import RetryingAgateClient


class StatusError(Exception):
    def __init__(self, status: int):
        self.status = status
        super().__init__(f"HTTP {status}")


class UploadClient:
    def __init__(self) -> None:
        self.prepared: list[tuple[str, list[dict[str, object]], str]] = []
        self.uploads: list[bytes] = []
        self.upload_paths: list[Path] = []
        self.submitted: list[tuple[str, dict[str, object]]] = []
        self.failures: dict[str, list[int]] = {}

    def _fail(self, method: str) -> None:
        statuses = self.failures.get(method, [])
        if statuses:
            raise StatusError(statuses.pop(0))

    def prepare_uploads(
        self, gpu: str, files: list[dict[str, object]], *, kind: str
    ) -> dict[str, object]:
        self.prepared.append((gpu, deepcopy(files), kind))
        self._fail("prepare")
        return {
            "job_id": f"dv_reserved_{len(self.prepared)}",
            "uploads": [
                {
                    "path": files[0]["path"],
                    "put_url": "https://oss.example.test/presigned?secret=not-for-agent",
                    "upload_ref": {"opaque": f"upload-{len(self.prepared)}"},
                }
            ],
        }

    def upload_file(self, url: str, path: str) -> None:
        assert url.startswith("https://oss.example.test/")
        self.upload_paths.append(Path(path))
        self._fail("upload")
        self.uploads.append(Path(path).read_bytes())

    def submit_job(self, kind: str, request: dict[str, object]) -> dict[str, object]:
        self.submitted.append((kind, request))
        self._fail("submit")
        return {"job_id": "dv_accepted", "status": "queued"}

    def health(self) -> bool:
        return True


def _request() -> dict[str, object]:
    return {
        "spec": {"target_hardware": ["L20D"]},
        "files": {
            "snapshots/incumbent/kernel.py": "print('A')\n",
            "snapshots/candidate/kernel.py": "print('B')\n# 中文\n",
            "reference/input.py": "def _make_inputs(): return []\n",
            "request.json": '{"lock_clocks":true}',
        },
        "command": "python3 snapshots/candidate/kernel.py",
        "timeout_s": 600,
        "env_vars": {"EXAMPLE": "unchanged"},
        "idempotency_key": "runtime-abba:logical-files-digest",
        "dev_intent": "custom_harness",
        "dev_note": "trusted same-allocation ABBA performance gate",
        "recycle": True,
    }


def _materialize(raw: UploadClient, root: Path) -> dict[str, object]:
    wire = raw.submitted[-1][1]
    for path, text in cast(dict[str, str], wire["files"]).items():
        root.joinpath(path).write_text(text, encoding="utf-8")
    attachments = cast(list[dict[str, str]], wire["oss_files"])
    root.joinpath(attachments[-1]["path"]).write_bytes(raw.uploads[-1])
    return wire


@pytest.mark.parametrize("kind", ["eval", "profile", "compile", "disassemble", "dev"])
def test_small_requests_remain_unchanged_and_other_methods_are_delegated(kind: str) -> None:
    raw = UploadClient()
    client = OssAgateClient(raw, max_inline_bytes=10_000)
    request = _request()
    assert client.submit_job(kind, request)["job_id"] == "dv_accepted"
    assert raw.submitted == [(kind, request)]
    assert raw.submitted[0][1] is request
    assert not raw.prepared and not raw.uploads
    assert client.health()


def test_inline_limit_counts_utf8_decoded_bytes_and_includes_the_boundary() -> None:
    raw = UploadClient()
    client = OssAgateClient(raw, max_inline_bytes=6)
    request = {**_request(), "files": {"file.txt": "中文"}}
    client.submit_job("dev", request)
    assert not raw.prepared
    client.submit_job("dev", {**request, "files": {"file.txt": "中文x"}})
    assert len(raw.prepared) == 1


def test_oversized_request_uses_one_archive_and_restores_exact_files(tmp_path: Path) -> None:
    raw = UploadClient()
    request = _request()
    before = deepcopy(request)
    client = OssAgateClient(raw, max_inline_bytes=1)
    assert client.submit_job("dev", request)["job_id"] == "dv_accepted"
    assert request == before
    assert len(raw.prepared) == len(raw.uploads) == 1
    gpu, descriptors, kind = raw.prepared[0]
    assert (gpu, kind) == ("L20D", "dev")
    assert descriptors[0]["bytes"] == len(raw.uploads[0])
    assert descriptors[0]["sha256"] == hashlib.sha256(raw.uploads[0]).hexdigest()
    assert all(not path.exists() for path in raw.upload_paths)
    wire = _materialize(raw, tmp_path)
    for key, value in before.items():
        if key not in {"files", "command"}:
            assert wire[key] == value
    completed = subprocess.run(
        ["/bin/sh", "-c", wire["command"]], cwd=tmp_path, capture_output=True, text=True, check=True
    )
    assert completed.stdout == "B\n"
    for path, text in cast(dict[str, str], before["files"]).items():
        assert tmp_path.joinpath(path).read_bytes() == text.encode("utf-8")
    assert not list(tmp_path.glob("__atrex_oss_payload_*.zip"))
    assert "presigned" not in json.dumps(wire)


def test_prepare_put_and_submit_retry_independently_without_reuploading() -> None:
    raw = UploadClient()
    raw.failures = {"prepare": [503], "upload": [503], "submit": [503, 503]}
    delays: list[float] = []
    client = OssAgateClient(RetryingAgateClient(raw, sleeper=delays.append), max_inline_bytes=1)
    assert client.submit_job("dev", _request())["job_id"] == "dv_accepted"
    assert delays == [5, 5, 5, 10]
    assert len(raw.prepared) == 2
    assert len(raw.upload_paths) == 2 and len(raw.uploads) == 1
    assert len(raw.submitted) == 3
    assert all(request is raw.submitted[0][1] for _, request in raw.submitted)
    assert all(not path.exists() for path in raw.upload_paths)


@pytest.mark.parametrize("stage", ["prepare", "upload", "submit"])
def test_permanent_upload_errors_propagate_and_local_staging_is_cleaned(stage: str) -> None:
    raw = UploadClient()
    raw.failures = {stage: [400]}
    delays: list[float] = []
    client = OssAgateClient(RetryingAgateClient(raw, sleeper=delays.append), max_inline_bytes=1)
    with pytest.raises(StatusError, match="HTTP 400"):
        client.submit_job("dev", _request())
    assert delays == []
    assert all(not path.exists() for path in raw.upload_paths)


def test_replacement_reserves_new_reference_but_preserves_deterministic_source_archive() -> None:
    raw = UploadClient()
    client = OssAgateClient(raw, max_inline_bytes=1)
    request = _request()
    request["oss_files"] = [{"path": "existing.bin", "upload_ref": "existing-reference"}]
    client.submit_job("dev", request)
    client.submit_job("dev", {**request, "idempotency_key": "infra-retry:new-job"})
    assert len(raw.prepared) == 2 and raw.uploads[0] == raw.uploads[1]
    first, second = [wire for _, wire in raw.submitted]
    assert first["idempotency_key"] == request["idempotency_key"]
    assert second["idempotency_key"] == "infra-retry:new-job"
    assert first["oss_files"][0] == second["oss_files"][0] == request["oss_files"][0]
    assert first["oss_files"][1] != second["oss_files"][1]


@pytest.mark.parametrize("path", ["../escape", "/absolute", "a/../escape", "a\\b", "a//b"])
def test_unsafe_paths_are_rejected_before_reserving_or_uploading(path: str) -> None:
    raw = UploadClient()
    with pytest.raises(ValueError, match="normalized and relative"):
        OssAgateClient(raw, max_inline_bytes=1).submit_job(
            "dev", {**_request(), "files": {path: "unsafe"}}
        )
    assert raw.prepared == raw.uploads == raw.submitted == []


def test_invalid_reservation_is_rejected_without_logging_credentials(monkeypatch) -> None:
    raw = UploadClient()
    monkeypatch.setattr(
        raw,
        "prepare_uploads",
        lambda *args, **kwargs: {
            "job_id": "dv_reserved",
            "uploads": [{"path": "wrong-path", "put_url": "secret", "upload_ref": "secret"}],
        },
    )
    with pytest.raises(ValueError, match="invalid path") as failure:
        OssAgateClient(raw, max_inline_bytes=1).submit_job("dev", _request())
    assert "secret" not in str(failure.value)
    assert not raw.uploads and not raw.submitted


def test_unpack_failure_does_not_execute_original_command_even_with_shell_fallback(
    tmp_path,
) -> None:
    raw = UploadClient()
    OssAgateClient(raw, max_inline_bytes=1).submit_job(
        "dev", {**_request(), "command": "false || touch should-not-execute"}
    )
    wire = _materialize(raw, tmp_path)
    archive = next(tmp_path.glob("__atrex_oss_payload_*.zip"))
    archive.write_bytes(b"corrupt uploaded bytes")
    completed = subprocess.run(
        ["/bin/sh", "-c", wire["command"]], cwd=tmp_path, capture_output=True, text=True
    )
    assert completed.returncode != 0
    assert "SHA-256" in completed.stderr
    assert not tmp_path.joinpath("should-not-execute").exists()


@pytest.mark.parametrize("case", ["traversal", "symlink", "duplicate", "parent-conflict"])
def test_unpack_validates_all_archive_entries_before_writing(tmp_path, monkeypatch, case) -> None:
    packed = io.BytesIO()
    with zipfile.ZipFile(packed, "w") as bundle:
        bundle.writestr("safe.txt", "must not be written")
        if case == "traversal":
            bundle.writestr("../escape.txt", "unsafe")
        elif case == "symlink":
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            bundle.writestr(info, "../escape")
        elif case == "duplicate":
            with pytest.warns(UserWarning, match="Duplicate name"):
                bundle.writestr("safe.txt", "duplicate")
        else:
            bundle.writestr("parent", "file")
            bundle.writestr("parent/child", "conflicting child")
    archive = tmp_path / "archive.zip"
    archive.write_bytes(packed.getvalue())
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        oss_remote.unpack(archive, hashlib.sha256(packed.getvalue()).hexdigest())
    assert not tmp_path.joinpath("safe.txt").exists()


def test_published_sdk_uses_upload_reservation_put_and_oss_reference_on_the_wire() -> None:
    posts: list[tuple[str, dict]] = []
    uploads: list[bytes] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args) -> None:
            pass

        def do_POST(self) -> None:
            value = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posts.append((self.path, value))
            if self.path == "/v1/uploads":
                descriptor = value["files"][0]
                reply = {
                    "job_id": "dv_reserved",
                    "uploads": [
                        {
                            "path": descriptor["path"],
                            "put_url": f"http://127.0.0.1:{self.server.server_port}/put",
                            "upload_ref": "opaque-reserved-reference",
                        }
                    ],
                }
            else:
                assert self.path == "/v1/jobs/dev"
                reply = {"job_id": "dv_reserved", "status": "queued"}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_PUT(self) -> None:
            assert self.path == "/put"
            uploads.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client, _ = load_agate_sdk(
            AgateConnectionConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                auth_mode="none",
                http_timeout_s=5,
                wait_timeout_s=5,
            )
        )
        request = {**_request(), "files": {"kernel.py": "#" * (4 * 1024 * 1024 + 1)}}
        assert client.submit_job("dev", request)["job_id"] == "dv_reserved"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert [path for path, _ in posts] == ["/v1/uploads", "/v1/jobs/dev"]
    assert len(uploads) == 1
    descriptor = posts[0][1]["files"][0]
    wire = posts[1][1]
    assert wire["oss_files"] == [
        {
            "path": descriptor["path"],
            "upload_ref": "opaque-reserved-reference",
        }
    ]
    assert descriptor["sha256"] == hashlib.sha256(uploads[0]).hexdigest()
    assert descriptor["bytes"] == len(uploads[0])
    assert sum(len(text.encode()) for text in wire["files"].values()) < 4 * 1024 * 1024
