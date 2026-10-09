"""Native Eval preserves sealed trees through the published SDK's OSS protocol."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread

import pytest
from test_agate_oss_client import UploadClient

from atrex_runtime.gateway.agate import AgateConnectionConfig, load_agate_sdk
from atrex_runtime.gateway.eval_sources import EVAL_ARCHIVES_KEY, prepare_eval_sources
from atrex_runtime.gateway.oss_client import OssAgateClient
from atrex_runtime.gateway.retrying_client import RetryingAgateClient
from atrex_runtime.kernel_sources import KernelSourceBundle, KernelSourceContract


def _source(value=1, **contract_updates):
    return KernelSourceBundle(
        {"kernel.py": "from impl import VALUE\n", "impl.py": f"VALUE = {value}\n# 中文\n"},
        KernelSourceContract(
            source_revision="a" * 40,
            seed_digest="sha256:" + "b" * 64,
            package_root=".",
            editable_roots=("impl.py",),
            immutable_files={},
        ).model_copy(update=contract_updates),
        "kernel.py",
    )


def _request(*, abba=False):
    candidate, baseline, attachments = prepare_eval_sources(_source(2), _source() if abba else None)
    request = {
        "candidate": candidate,
        "spec": {"target_hardware": ["L20D"]},
        "idempotency_key": "exact-logical-request",
        **attachments,
    }
    if abba:
        request["abba"] = {"baseline": baseline, "repeats": 2}
    return request


class NativeUploadClient(UploadClient):
    def prepare_uploads(self, gpu, files, *, kind):
        result = super().prepare_uploads(gpu, files, kind=kind)
        result["uploads"] = [
            {**result["uploads"][0], "path": f["path"], "upload_ref": f"ref-{i}"}
            for i, f in enumerate(files)
        ]
        return result


@pytest.mark.parametrize("abba", [False, True])
def test_archives_preserve_exact_files_and_one_eval_reservation(abba):
    raw = NativeUploadClient()
    request = _request(abba=abba)
    before = deepcopy(request)
    client = OssAgateClient(raw)
    client.submit_job("eval", request)
    assert request == before
    assert len(raw.prepared) == 1
    gpu, descriptors, kind = raw.prepared[0]
    assert (gpu, kind) == ("L20D", "eval")
    assert len(descriptors) == len(raw.uploads) == (2 if abba else 1)
    for descriptor, data in zip(descriptors, raw.uploads, strict=True):
        assert descriptor["bytes"] == len(data)
        assert descriptor["sha256"] == hashlib.sha256(data).hexdigest()
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            files = {member.name: archive.extractfile(member).read().decode() for member in archive}
        assert files == request[EVAL_ARCHIVES_KEY][descriptor["path"]]
    kind, wire = raw.submitted[0]
    assert kind == "eval" and EVAL_ARCHIVES_KEY not in wire
    assert not {"files", "command", "dev_intent"}.intersection(wire)
    assert wire["candidate"] == request["candidate"]
    assert wire["idempotency_key"] == request["idempotency_key"]
    assert all(not path.exists() for path in raw.upload_paths)
    client.submit_job("eval", request)
    count = len(descriptors)
    assert raw.uploads[:count] == raw.uploads[count:]  # deterministic bytes


def test_upload_and_submission_retries_preserve_both_sides_and_idempotency():
    raw = NativeUploadClient()
    raw.failures = {"prepare": [503], "upload": [502], "submit": [503, 503]}
    delays = []
    client = OssAgateClient(RetryingAgateClient(raw, sleeper=delays.append))
    client.submit_job("eval", _request(abba=True))
    assert delays == [5, 5, 5, 10]
    assert len(raw.prepared) == 2 and len(raw.uploads) == 2
    assert len(raw.submitted) == 3
    assert all(wire is raw.submitted[0][1] for _, wire in raw.submitted)
    assert all(not path.exists() for path in raw.upload_paths)


@pytest.mark.parametrize("fault", ["path", "entry", "reservation"])
def test_invalid_archive_or_reservation_never_submits(fault, monkeypatch):
    raw = NativeUploadClient()
    request = _request(abba=True)
    if fault == "path":
        request[EVAL_ARCHIVES_KEY]["archives/candidate.tar.gz"]["../escape"] = "unsafe"
    elif fault == "entry":
        request["abba"]["baseline"]["entry_point"] = "missing.py"
    else:
        monkeypatch.setattr(
            raw, "prepare_uploads", lambda *a, **kw: {"job_id": "id", "uploads": []}
        )
    with pytest.raises(ValueError):
        OssAgateClient(raw).submit_job("eval", request)
    assert not raw.uploads and not raw.submitted


def test_nonroot_package_is_rejected_and_dependencies_are_preserved():
    with pytest.raises(ValueError, match="package_root"):
        prepare_eval_sources(_source(package_root="src"))
    source = _source(runtime_requirements=({"distribution": "example", "version": "==2"},))
    _, _, fields = prepare_eval_sources(source, requirements=("torch>=2.9",))
    assert fields["requirements"] == ["torch>=2.9", "example==2"]


def test_existing_oss_references_cannot_be_mixed_with_new_reservation():
    raw = NativeUploadClient()
    request = {
        **_request(), "oss_files": [{"path": "old.tar.gz", "upload_ref": "another-job"}]
    }
    with pytest.raises(ValueError, match="cannot mix existing OSS references"):
        OssAgateClient(raw).submit_job("eval", request)
    assert not raw.prepared and not raw.uploads and not raw.submitted


@pytest.mark.parametrize("abba", [False, True])
def test_published_sdk_posts_multi_file_requests_only_to_eval(abba):
    posts, uploads = [], []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            posts.append((self.path, body))
            if self.path == "/v1/uploads":
                reply = {
                    "job_id": "ev_reserved",
                    "uploads": [
                        {
                            "path": descriptor["path"],
                            "put_url": f"http://127.0.0.1:{self.server.server_port}/put/{i}",
                            "upload_ref": f"opaque-{i}",
                        }
                        for i, descriptor in enumerate(body["files"])
                    ],
                }
            else:
                reply = {"job_id": "ev_reserved", "status": "queued"}
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_PUT(self):
            uploads.append(self.rfile.read(int(self.headers["Content-Length"])))
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client, builder = load_agate_sdk(
            AgateConnectionConfig(
                base_url=f"http://127.0.0.1:{server.server_port}",
                auth_mode="none", http_timeout_s=5, wait_timeout_s=5,
            )
        )
        candidate, baseline, attachments = prepare_eval_sources(
            _source(2), _source() if abba else None
        )
        payload = builder(
            candidate,
            {"operator": "example", "reference_py": "ref", "input_py": "input", "shapes": {}},
            "L20D",
            name="source-eval",
            abba={"baseline": baseline, "repeats": 1} if abba else None,
            idempotency_key="native-test",
        )
        client.submit_job("eval", {**payload, **attachments})
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert [path for path, _ in posts] == ["/v1/uploads", "/v1/jobs/eval"]
    assert posts[0][1]["kind"] == "eval"
    wire = posts[1][1]
    assert len(wire["oss_files"]) == len(uploads) == (2 if abba else 1)
    assert wire["idempotency_key"] == "native-test"
    assert EVAL_ARCHIVES_KEY not in wire and "files" not in wire
