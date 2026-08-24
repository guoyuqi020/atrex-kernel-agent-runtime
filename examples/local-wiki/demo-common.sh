#!/usr/bin/env bash

atrex_demo_example_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
atrex_demo_runtime_root="$(cd -- "${atrex_demo_example_dir}/../.." && pwd)"
atrex_demo_state_dir="${ATREX_DEMO_STATE_DIR:-${atrex_demo_runtime_root}/local-wiki/state}"
atrex_demo_env_file="${ATREX_DEMO_ENV_FILE:-${atrex_demo_state_dir}/demo.env}"
atrex_demo_bootstrap_result_file="${ATREX_DEMO_BOOTSTRAP_RESULT_FILE:-${atrex_demo_state_dir}/last-bootstrap.json}"
atrex_demo_python="${ATREX_PYTHON:-python3}"
atrex_demo_runtime_cli="${ATREX_RUNTIME_CLI:-atrex-kernel-agent-runtime}"
# shellcheck source=../shared/local-secrets.sh
source "${atrex_demo_runtime_root}/examples/shared/local-secrets.sh"

atrex_demo_ensure_env() {
  atrex_shared_ensure_local_secrets "${atrex_demo_env_file}" "demo"
}

atrex_demo_load_env() {
  atrex_shared_load_local_secrets "${atrex_demo_env_file}" "demo"
}

atrex_demo_require_optimizer_backend() {
  local python_executable="${atrex_demo_python}"
  local backend
  backend="$("${python_executable}" -c '
import json
import sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(value["campaign"]["optimizer"]["agent_backend"])
' "$(atrex_demo_config_path)")"
  if ! command -v "${backend}" >/dev/null 2>&1; then
    echo "Optimizer Backend '${backend}' is not reachable through PATH" >&2
    return 69
  fi
  if [[ "${backend}" == "codex" ]]; then
    if [[ -z "${CODEX_HOME:-}" ]]; then
      export CODEX_HOME="${HOME}/.codex"
    fi
    if [[ ! -f "${CODEX_HOME}/auth.json" ]]; then
      echo "Optimizer selects Codex, but ${CODEX_HOME}/auth.json is unavailable." >&2
      return 78
    fi
  elif [[ "${backend}" == "claude" ]] \
    && [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" && -z "${ANTHROPIC_API_KEY:-}" ]] \
    && [[ ! -d "${CLAUDE_CONFIG_DIR:-${HOME}/.claude}" && ! -f "${HOME}/.claude.json" ]]; then
    echo "Optimizer selects Claude, but no token or host login state is available." >&2
    return 78
  elif [[ "${backend}" == "qodercli" ]] \
    && [[ -z "${QODER_PERSONAL_ACCESS_TOKEN:-}" ]] \
    && [[ ! -d "${HOME}/.qoder" && ! -d "${HOME}/.qodersec" ]]; then
    echo "Optimizer selects QoderCLI, but no PAT or host login state is available." >&2
    return 78
  fi
}

atrex_demo_config_path() {
  printf '%s\n' "${atrex_demo_example_dir}/runtime.json"
}

atrex_demo_last_lineage_id() {
  local python_executable="${atrex_demo_python}"
  if [[ ! -f "${atrex_demo_bootstrap_result_file}" ]]; then
    echo "Bootstrap result not found: ${atrex_demo_bootstrap_result_file}" >&2
    echo "Run examples/local-wiki/bootstrap-campaign.sh first." >&2
    return 66
  fi
  if ! command -v "${python_executable}" >/dev/null 2>&1; then
    echo "Runtime Python not found in PATH: ${python_executable}" >&2
    return 69
  fi
  "${python_executable}" -c '
import json
import re
import sys

value = json.loads(open(sys.argv[1], encoding="utf-8").read())
lineages = value.get("lineages")
lineage_id = (
    lineages[0].get("lineage_id")
    if isinstance(lineages, list) and lineages and isinstance(lineages[0], dict)
    else value.get("lineage_id")
)
if not isinstance(lineage_id, str) or re.fullmatch(r"lineage_[0-9a-f]{32}", lineage_id) is None:
    raise SystemExit("saved Bootstrap result has no valid lineage_id")
print(lineage_id)
' "${atrex_demo_bootstrap_result_file}"
}
