#!/usr/bin/env bash

atrex_prod_script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
atrex_prod_root="$(cd -- "${atrex_prod_script_dir}/../.." && pwd)"

if [[ -n "${ATREX_PYTHON:-}" ]]; then
  atrex_prod_python="${ATREX_PYTHON}"
elif [[ -x "${HOME}/.venvs/atrex-runtime/bin/python" ]]; then
  atrex_prod_python="${HOME}/.venvs/atrex-runtime/bin/python"
elif [[ -x "${atrex_prod_root}/.venv/bin/python" ]]; then
  atrex_prod_python="${atrex_prod_root}/.venv/bin/python"
else
  atrex_prod_python="python3"
fi

if [[ -n "${ATREX_RUNTIME_CLI:-}" ]]; then
  atrex_prod_cli="${ATREX_RUNTIME_CLI}"
elif [[ -x "$(dirname -- "${atrex_prod_python}")/atrex-kernel-agent-runtime" ]]; then
  atrex_prod_cli="$(dirname -- "${atrex_prod_python}")/atrex-kernel-agent-runtime"
else
  atrex_prod_cli="atrex-kernel-agent-runtime"
fi

atrex_prod_absolute_path() {
  "${atrex_prod_python}" -c 'from pathlib import Path; import sys; print(Path(sys.argv[1]).expanduser().resolve())' "$1"
}

atrex_prod_workspace_paths() {
  atrex_prod_workspace="$(atrex_prod_absolute_path "$1")" || return
  atrex_prod_config="${atrex_prod_workspace}/runtime.json"
  atrex_prod_manifest="${atrex_prod_workspace}/production-manifest.json"
  atrex_prod_secrets="${atrex_prod_workspace}/runtime.env"
  atrex_prod_services="${atrex_prod_workspace}/services"
  atrex_prod_dsls_root="${atrex_prod_workspace}/dsls"
  atrex_prod_bootstrap_summary="${atrex_prod_workspace}/bootstrap-results.json"
  atrex_prod_campaign_summary="${atrex_prod_workspace}/campaign-results.json"
  atrex_prod_ablation_plan="${atrex_prod_workspace}/ablation.json"
}

atrex_prod_arm_paths() {
  local dsl="$1"
  local label="$2"
  atrex_prod_dsl_paths "${dsl}"
  atrex_prod_arm_workspace="${atrex_prod_dsl_workspace}/${label}"
  atrex_prod_arm_spec="${atrex_prod_arm_workspace}/arm.json"
  atrex_prod_arm_seed_result="${atrex_prod_arm_workspace}/seed-result.json"
  atrex_prod_arm_campaign_result="${atrex_prod_arm_workspace}/campaign-result.json"
  atrex_prod_arm_log="${atrex_prod_arm_workspace}/campaign.log"
}

atrex_prod_dsl_paths() {
  local dsl="$1"
  case "${dsl}" in cuda|triton|cutedsl) ;; *) return 64 ;; esac
  atrex_prod_dsl="${dsl}"
  atrex_prod_dsl_workspace="${atrex_prod_dsls_root}/${dsl}"
  atrex_prod_campaign="${atrex_prod_dsl_workspace}/campaign.json"
  atrex_prod_bootstrap_result="${atrex_prod_dsl_workspace}/bootstrap-result.json"
  atrex_prod_campaign_result="${atrex_prod_dsl_workspace}/campaign-result.json"
  atrex_prod_bootstrap_log="${atrex_prod_dsl_workspace}/bootstrap.log"
  atrex_prod_campaign_log="${atrex_prod_dsl_workspace}/campaign.log"
}

atrex_prod_require_workspace() {
  local path
  for path in "${atrex_prod_config}" "${atrex_prod_manifest}" "${atrex_prod_secrets}"; do
    if [[ ! -f "${path}" ]]; then
      echo "Production workspace is not prepared; missing: ${path}" >&2
      return 66
    fi
  done
  local dsl
  for dsl in cuda triton cutedsl; do
    if [[ ! -f "${atrex_prod_dsls_root}/${dsl}/campaign.json" ]]; then
      echo "Production workspace is incomplete; missing DSL Campaign: ${dsl}" >&2
      return 66
    fi
  done
}

atrex_prod_require_service_workspace() {
  local path
  for path in \
    "${atrex_prod_config}" \
    "${atrex_prod_manifest}" \
    "${atrex_prod_secrets}" \
    "${atrex_prod_workspace}/local-wiki.json"; do
    if [[ ! -f "${path}" ]]; then
      echo "Production service workspace is not initialized; missing: ${path}" >&2
      return 66
    fi
  done
}

atrex_prod_require_policy_gate() {
  "${atrex_prod_python}" -c '
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
gate=value.get("gate_policy")
if not isinstance(gate, dict):
    campaign=value.get("campaign")
    gate=campaign.get("gate_policy") if isinstance(campaign, dict) else None
if not isinstance(gate, dict) or gate.get("production_gate") is not True:
    raise SystemExit(
        "Production workspace must enable gate_policy.production_gate"
    )
' "${atrex_prod_config}"
}

atrex_prod_load_environment() {
  local extra_env_file="${1:-}"
  # shellcheck disable=SC1090
  source "${atrex_prod_secrets}"
  if [[ -n "${extra_env_file}" ]]; then
    if [[ ! -f "${extra_env_file}" ]]; then
      echo "Environment file not found: ${extra_env_file}" >&2
      return 66
    fi
    # shellcheck disable=SC1090
    source "${extra_env_file}"
  fi
}

atrex_prod_require_agate() {
  local missing=()
  local name
  for name in AGATE_URL AGATE_AK AGATE_SK AGATE_GPU; do
    if [[ -z "${!name:-}" ]]; then
      missing+=("${name}")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "Missing required Agate environment: ${missing[*]}" >&2
    return 64
  fi
}

atrex_prod_backend() {
  "${atrex_prod_python}" -c '
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
print(value["backend"])
' "${atrex_prod_manifest}"
}

atrex_prod_require_backend() {
  local backend
  backend="$(atrex_prod_backend)" || return
  if ! command -v "${backend}" >/dev/null 2>&1; then
    echo "Configured Agent backend is not in PATH: ${backend}" >&2
    return 69
  fi
  case "${backend}" in
    codex)
      export CODEX_HOME="${CODEX_HOME:-${HOME}/.codex}"
      if [[ ! -f "${CODEX_HOME}/auth.json" ]]; then
        echo "Codex login state not found: ${CODEX_HOME}/auth.json" >&2
        return 78
      fi
      ;;
    claude)
      if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" && -z "${ANTHROPIC_API_KEY:-}" \
        && ! -d "${CLAUDE_CONFIG_DIR:-${HOME}/.claude}" \
        && ! -f "${HOME}/.claude.json" ]]; then
        echo "Claude credentials or host login state are unavailable." >&2
        return 78
      fi
      ;;
    qodercli)
      if [[ -z "${QODER_PERSONAL_ACCESS_TOKEN:-}" \
        && ! -d "${HOME}/.qoder" && ! -d "${HOME}/.qodersec" ]]; then
        echo "QoderCLI credentials or host login state are unavailable." >&2
        return 78
      fi
      ;;
    pi)
      ;;
    *)
      echo "Unsupported configured backend: ${backend}" >&2
      return 64
      ;;
  esac
}

atrex_prod_require_commands() {
  local command
  for command in "${atrex_prod_python}" "${atrex_prod_cli}" curl; do
    if ! command -v "${command}" >/dev/null 2>&1; then
      echo "Required command is not in PATH: ${command}" >&2
      return 69
    fi
  done
}

atrex_prod_json_value() {
  local file="$1"
  local expression="$2"
  "${atrex_prod_python}" -c '
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
for part in sys.argv[2].split("."):
    value=value[part]
print(value)
' "${file}" "${expression}"
}

atrex_prod_pid_alive() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid expected_start actual_start
  read -r pid expected_start <"${pid_file}" || return 1
  [[ "${pid}" =~ ^[1-9][0-9]*$ ]] || return 1
  [[ "${expected_start}" =~ ^[0-9]+$ ]] || return 1
  [[ -r "/proc/${pid}/stat" ]] || return 1
  actual_start="$(atrex_prod_process_start "${pid}")" || return 1
  [[ "${actual_start}" == "${expected_start}" ]]
}

atrex_prod_process_start() {
  local pid="$1"
  "${atrex_prod_python}" -c '
from pathlib import Path
import sys
value=(Path("/proc")/sys.argv[1]/"stat").read_text(encoding="utf-8")
closing=value.rfind(")")
fields=value[closing+1:].split()
if closing < 0 or len(fields) <= 19: raise SystemExit(1)
print(fields[19])
' "${pid}"
}

atrex_prod_write_pid() {
  local pid_file="$1"
  local pid="$2"
  local start
  start="$(atrex_prod_process_start "${pid}")" || {
    echo "Cannot record stable process identity for pid ${pid}." >&2
    return 1
  }
  printf '%s %s\n' "${pid}" "${start}" >"${pid_file}"
  chmod 0600 "${pid_file}"
}

atrex_prod_read_pid() {
  local pid_file="$1"
  local pid _start
  read -r pid _start <"${pid_file}" || return 1
  printf '%s\n' "${pid}"
}

atrex_prod_stop_pid() {
  local label="$1"
  local pid_file="$2"
  if ! atrex_prod_pid_alive "${pid_file}"; then
    rm -f -- "${pid_file}"
    echo "${label}: stopped"
    return 0
  fi
  local pid
  pid="$(atrex_prod_read_pid "${pid_file}")"
  kill "${pid}" >/dev/null 2>&1 || true
  local _
  for _ in {1..100}; do
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      break
    fi
    sleep 0.2
  done
  if kill -0 "${pid}" >/dev/null 2>&1; then
    echo "${label} did not stop gracefully; sending SIGKILL to owned pid ${pid}." >&2
    kill -KILL "${pid}" >/dev/null 2>&1 || true
  fi
  rm -f -- "${pid_file}"
  echo "${label}: stopped (pid ${pid})"
}

atrex_prod_wait_url() {
  local label="$1"
  local url="$2"
  local pid_file="$3"
  local timeout="${4:-90}"
  local deadline=$((SECONDS + timeout))
  while ! curl --fail --silent --show-error --max-time 3 "${url}" >/dev/null 2>&1; do
    if [[ -n "${pid_file}" ]] && ! atrex_prod_pid_alive "${pid_file}"; then
      echo "${label} exited before becoming ready: ${url}" >&2
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "${label} did not become ready within ${timeout}s: ${url}" >&2
      return 1
    fi
    sleep 1
  done
}

atrex_prod_require_control_plane_ready() {
  local config="$1"
  local runtime_host runtime_port runtime_url wiki_url
  runtime_host="$(atrex_prod_json_value "${config}" server.host)" || return
  runtime_port="$(atrex_prod_json_value "${config}" server.port)" || return
  runtime_url="http://${runtime_host}:${runtime_port}"
  wiki_url="$(atrex_prod_json_value "${config}" gpu_wiki.base_url)" || return
  if ! curl --fail --silent --show-error --max-time 3 \
    "${runtime_url}/healthz" >/dev/null; then
    echo "Runtime is not healthy: ${runtime_url}" >&2
    echo "Start the shared services before running a Campaign task." >&2
    return 69
  fi
  if ! curl --fail --silent --show-error --max-time 3 \
    "${wiki_url}/readyz" >/dev/null; then
    echo "GPU Wiki is not ready: ${wiki_url}" >&2
    echo "Start the shared services before running a Campaign task." >&2
    return 69
  fi
  echo "Shared Runtime: healthy at ${runtime_url}"
  echo "Shared GPU Wiki: ready at ${wiki_url}"
}

atrex_prod_escalate() {
  local script="$1"
  shift
  local launcher_mode="${ATREX_LAUNCHER_MODE:-}"
  if [[ -z "${launcher_mode}" && -f "${atrex_prod_manifest:-}" ]]; then
    launcher_mode="$("${atrex_prod_python}" -c '
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
print(value.get("launcher_mode", "sandbox"))
' "${atrex_prod_manifest}")" || return
  fi
  if [[ "${launcher_mode}" == "container" ]]; then
    return 0
  fi
  if [[ "$(uname -s)" != "Linux" ]]; then
    echo "Production sandbox execution requires Linux." >&2
    return 69
  fi
  if (( EUID == 0 )); then
    return 0
  fi
  if ! command -v sudo >/dev/null 2>&1; then
    echo "Production sandbox execution requires root or sudo." >&2
    return 77
  fi
  export ATREX_PRODUCTION_ROOT=1
  export ATREX_SANDBOX_HOST_HOME="${ATREX_SANDBOX_HOST_HOME:-${HOME}}"
  export ATREX_SANDBOX_HOST_PATH="${ATREX_SANDBOX_HOST_PATH:-${PATH}}"
  export ATREX_PYTHON="$(command -v "${atrex_prod_python}")"
  export ATREX_RUNTIME_CLI="$(command -v "${atrex_prod_cli}")"
  exec sudo --preserve-env=AGATE_URL,AGATE_AK,AGATE_SK,AGATE_GPU,AGATE_HTTP_TIMEOUT,AGATE_WAIT_TIMEOUT,AGATE_HEALTH_CHECK_INTERVAL,ATREX_LAUNCHER_MODE,ATREX_PRODUCTION_ROOT,ATREX_SANDBOX_HOST_HOME,ATREX_SANDBOX_HOST_PATH,ATREX_PYTHON,ATREX_RUNTIME_CLI,ATREX_WIKI_URL,ATREX_WIKI_TOKEN_ENV,ANTHROPIC_AUTH_TOKEN,ANTHROPIC_API_KEY,ANTHROPIC_BASE_URL,ANTHROPIC_MODEL,ANTHROPIC_DEFAULT_HAIKU_MODEL,ANTHROPIC_DEFAULT_OPUS_MODEL,ANTHROPIC_DEFAULT_SONNET_MODEL,CODEX_HOME,QODER_PERSONAL_ACCESS_TOKEN \
    -- bash "${script}" "$@"
}

atrex_prod_restore_host_environment() {
  if [[ -n "${ATREX_SANDBOX_HOST_HOME:-}" ]]; then
    export HOME="${ATREX_SANDBOX_HOST_HOME}"
  fi
  if [[ -n "${ATREX_SANDBOX_HOST_PATH:-}" ]]; then
    export PATH="${ATREX_SANDBOX_HOST_PATH}"
  fi
}
