"""Exercise both bundled CLIs against the real authenticated report HTTP handler."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import anyio
import pytest
from httpx import ASGITransport, AsyncClient
from test_attempt_report import _value as report_value
from test_dev_only_ablation import enable_dev_only
from test_gateway_proxy import _service

from atrex_runtime.gateway import GatewayProxyAsgiApp, GatewayProxyLimits
from atrex_runtime.gateway.proxy import GatewayAdapterResult
from atrex_runtime.workers.session_contract import materialize_session_contract

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("bundle", ["atrex-kernel-agent-core", "kernel-design-agents"])
def test_cli_report_has_no_hidden_query_and_gets_registered_receipt(tmp_path, bundle):
    registry, control, attempt, capability, service, adapter = _service(tmp_path)
    enable_dev_only(registry, attempt)
    adapter.result = GatewayAdapterResult("completed", {"stdout": "probe passed", "exit_code": 0})
    contract = materialize_session_contract(
        tmp_path,
        phase="optimization_attempt",
        dsl="cuda",
        hardware_target="L20D",
        agent_backend="claude",
        model=None,
        session_timeout_seconds=3600,
        usage_unit="provider_tokens",
        usage_budget=1000,
        max_attempt_report_bytes=100_000,
        wiki_available=False,
        tool_modules=(),
        result_artifact_read_enabled=False,
        dev_only=True,
    )
    (tmp_path / "work/kernel").mkdir(parents=True)
    (tmp_path / "work/kernel/kernel.py").write_text("def kernel(): pass\n")
    (tmp_path / "scratch").mkdir()
    report = report_value(attempt.id)
    report = {
        key: report.get(key, [])
        for key in ("status", "final_candidate", "contributing_result_artifact_digests", "blocker")
    }
    report["contributing_result_artifact_digests"] = []
    (tmp_path / "scratch/input-report.json").write_text(json.dumps(report))
    app = GatewayProxyAsgiApp(service, GatewayProxyLimits(65536, 8, 16384))
    requests = []

    async def dispatch(path, payload, authorization):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://runtime") as http:
            return await http.post(path, content=payload, headers={"Authorization": authorization})

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            payload = self.rfile.read(int(self.headers["Content-Length"]))
            requests.append(json.loads(payload)["operation"])
            response = anyio.run(dispatch, self.path, payload, self.headers["Authorization"])
            self.send_response(response.status_code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response.content)))
            self.end_headers()
            self.wfile.write(response.content)

    script = r"""
import contextlib, io, json, sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, sys.argv[1])
import runtime_tools as rt
root=Path(sys.argv[2])
request=json.loads((root/'scratch/input-report.json').read_text())
request={key:request[key] for key in rt._REPORT_FIELDS}
(root/'scratch/request.json').write_text(json.dumps(request))
ctx=SimpleNamespace(workspace=root, attempt_id=sys.argv[3],
    report_path=root/'scratch/attempt-report.json', working_kernel=root/'work/kernel',
    gateway_url=sys.argv[4], gateway_capability=sys.argv[5], manifest={})
rt._context=lambda command:ctx
denied=root/'scratch/disabled.json'
denied.write_text(json.dumps({'operation':'evaluate','latency_prediction':'improved'}))
output=io.StringIO()
with contextlib.redirect_stdout(output):
    code=rt.main(['gateway-execute','--request',str(denied)])
assert code != 0
error=json.loads(output.getvalue())
assert set(error['request_schema']['operations']) == {'dev'}
assert 'only Dev' in error['recovery'][0]['instruction']
probe=root/'scratch/probe.json'
probe.write_text(json.dumps({'operation':'dev','command':'python3 kernel.py'}))
output=io.StringIO()
with contextlib.redirect_stdout(output):
    assert rt.main(['gateway-execute','--request',str(probe)]) == 0
observed=json.loads(output.getvalue())
source=root/'scratch/source.json'
source.write_text(json.dumps({'kernel_artifact_digest':observed['kernel_artifact_digest'],
    'artifact_file':'kernel.py','file':'scratch/historical-kernel.py'}))
output=io.StringIO()
with contextlib.redirect_stdout(output):
    assert rt.main(['kernel-artifact-read','--request',str(source)]) == 0
assert json.loads(output.getvalue())['status'] == 'completed'
assert (root/'scratch/historical-kernel.py').read_text() == 'def kernel(): pass\n'
disabled_read=root/'scratch/result.json'
disabled_read.write_text(json.dumps({'result_artifact_digest':observed['result_artifact_digest']}))
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    try:
        rt.main(['result-artifact-read','--request',str(disabled_read)])
    except SystemExit as error:
        assert error.code == 2
    else:
        raise AssertionError('Result reads must not be an available CLI command')
raise SystemExit(rt.main(['attempt-report','--request',str(root/'scratch/request.json')]))
"""
    try:
        with HTTPServer(("127.0.0.1", 0), Handler) as server:
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                result = subprocess.run(
                    [
                        sys.executable,
                        "-c",
                        script,
                        str(ROOT / "src" / bundle / "src"),
                        str(tmp_path),
                        attempt.id,
                        f"http://127.0.0.1:{server.server_port}",
                        capability.token,
                    ],
                    env={**os.environ, "ATREX_RUNTIME_CONTRACT_PATH": str(contract)},
                    text=True,
                    capture_output=True,
                    timeout=20,
                )
                assert result.returncode == 0, result.stdout + result.stderr
            finally:
                server.shutdown()
                thread.join(timeout=5)
        assert requests == ["dev", "kernel_artifact_read", "attempt_report"]
        assert json.loads(result.stdout)["status"] == "published"
        published = json.loads((tmp_path / "scratch/attempt-report.json").read_text())
        assert published["tool_modules"] == published["experiments"] == []
        assert published["direction_events"] == []
        assert not control.list_evaluations(attempt.id)
        assert len(adapter.requests) == 1
    finally:
        control.close()
        registry.close()
