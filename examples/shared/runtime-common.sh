#!/usr/bin/env bash

for required_name in \
  atrex_example_dir atrex_runtime_root atrex_state_dir atrex_env_file \
  atrex_config_file atrex_runtime_template atrex_campaign_file \
  atrex_campaign_template atrex_bootstrap_result_file; do
  if [[ -z "${!required_name:-}" ]]; then
    echo "shared Runtime helper requires ${required_name}" >&2
    return 64 2>/dev/null || exit 64
  fi
done

atrex_runtime_cli="${ATREX_RUNTIME_CLI:-atrex-kernel-agent-runtime}"
atrex_python="${ATREX_PYTHON:-python3}"
atrex_managed_local_wiki_pid=""
atrex_managed_local_wiki_log="${atrex_state_dir}/local-wiki.log"
# shellcheck source=local-secrets.sh
source "${atrex_runtime_root}/examples/shared/local-secrets.sh"

atrex_example_cleanup_managed_local_wiki() {
  local exit_status=$?
  trap - EXIT
  if [[ -n "${atrex_managed_local_wiki_pid}" ]] \
    && kill -0 "${atrex_managed_local_wiki_pid}" >/dev/null 2>&1; then
    kill "${atrex_managed_local_wiki_pid}" >/dev/null 2>&1 || true
    wait "${atrex_managed_local_wiki_pid}" >/dev/null 2>&1 || true
    echo "Local Wiki stopped (pid ${atrex_managed_local_wiki_pid})."
  fi
  exit "${exit_status}"
}

atrex_example_prepare_gpu_wiki() {
  if [[ -n "${ATREX_WIKI_URL:-}" ]]; then
    echo "Using configured GPU Wiki: ${ATREX_WIKI_URL}"
    return 0
  fi

  export ATREX_WIKI_URL="http://127.0.0.1:8091"
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to start the managed Local Wiki" >&2
    return 69
  fi
  if curl --fail --silent --max-time 2 "${ATREX_WIKI_URL}/readyz" >/dev/null 2>&1; then
    echo "Using the existing Local Wiki at ${ATREX_WIKI_URL}."
    return 0
  fi

  local start_timeout="${ATREX_LOCAL_WIKI_START_TIMEOUT:-60}"
  if [[ ! "${start_timeout}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ATREX_LOCAL_WIKI_START_TIMEOUT must be a positive integer" >&2
    return 64
  fi
  mkdir -p "${atrex_state_dir}"
  umask 077
  bash "${atrex_runtime_root}/examples/shared/start-local-wiki.sh" \
    >"${atrex_managed_local_wiki_log}" 2>&1 &
  atrex_managed_local_wiki_pid=$!
  trap atrex_example_cleanup_managed_local_wiki EXIT
  echo "Local Wiki starting at ${ATREX_WIKI_URL} (pid ${atrex_managed_local_wiki_pid})"
  echo "Local Wiki log: ${atrex_managed_local_wiki_log}"

  local deadline=$((SECONDS + start_timeout))
  while ! curl --fail --silent --max-time 2 \
    "${ATREX_WIKI_URL}/readyz" >/dev/null 2>&1; do
    if ! kill -0 "${atrex_managed_local_wiki_pid}" >/dev/null 2>&1; then
      echo "Local Wiki exited before becoming ready:" >&2
      tail -n 80 "${atrex_managed_local_wiki_log}" >&2 || true
      return 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Local Wiki did not become ready within ${start_timeout} seconds:" >&2
      tail -n 80 "${atrex_managed_local_wiki_log}" >&2 || true
      return 1
    fi
    sleep 1
  done
  echo "Local Wiki is ready."
  echo
}

atrex_example_ensure_local_secrets() {
  atrex_shared_ensure_local_secrets "${atrex_env_file}" "Runtime"
}

atrex_example_load_local_secrets() {
  atrex_shared_load_local_secrets "${atrex_env_file}" "Runtime"
}

atrex_example_require_remote_agate() {
  local missing=()
  for key in AGATE_URL AGATE_AK AGATE_SK AGATE_GPU; do
    if [[ -z "${!key:-}" ]]; then
      missing+=("${key}")
    fi
  done
  if (( ${#missing[@]} > 0 )); then
    echo "missing required remote Agate environment: ${missing[*]}" >&2
    return 64
  fi
}

atrex_example_agent_backend() {
  local role="${1:-}"
  if [[ "${role}" != "optimizer" && "${role}" != "evolver" ]]; then
    echo "Agent role must be optimizer or evolver" >&2
    return 64
  fi
  local source="${atrex_runtime_template}"
  if [[ -f "${atrex_config_file}" ]]; then
    source="${atrex_config_file}"
  fi
  "${atrex_python}" -c '
import json
import sys
value = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(value["campaign"][sys.argv[2]]["agent_backend"])
' "${source}" "${role}"
}

atrex_example_require_agent_backend() {
  local role="${1:-}"
  local backend
  backend="$(atrex_example_agent_backend "${role}")" || return
  if ! command -v "${backend}" >/dev/null 2>&1; then
    echo "${role} Backend '${backend}' is not reachable through PATH" >&2
    return 69
  fi
  case "${backend}" in
    codex)
      if [[ -z "${CODEX_HOME:-}" ]]; then
        export CODEX_HOME="${HOME}/.codex"
      fi
      if [[ ! -f "${CODEX_HOME}/auth.json" ]]; then
        echo "${role} selects Codex, but ${CODEX_HOME}/auth.json is unavailable." >&2
        echo "Run 'codex login' before starting the example." >&2
        return 78
      fi
      ;;
    claude)
      if [[ -z "${ANTHROPIC_AUTH_TOKEN:-}" && -z "${ANTHROPIC_API_KEY:-}" \
        && ! -d "${CLAUDE_CONFIG_DIR:-${HOME}/.claude}" \
        && ! -f "${HOME}/.claude.json" ]]; then
        echo "${role} selects Claude, but no token or host login state is available." >&2
        return 78
      fi
      ;;
    qodercli)
      if [[ -z "${QODER_PERSONAL_ACCESS_TOKEN:-}" \
        && ! -d "${HOME}/.qoder" \
        && ! -d "${HOME}/.qodersec" ]]; then
        echo "${role} selects QoderCLI, but no PAT or host login state is available." >&2
        return 78
      fi
      ;;
    pi)
      ;;
    *)
      echo "unsupported ${role} Backend in Runtime config: ${backend}" >&2
      return 64
      ;;
  esac
}

atrex_example_prepare_inputs() {
  if ! command -v "${atrex_python}" >/dev/null 2>&1; then
    echo "Runtime Python not found in PATH: ${atrex_python}" >&2
    return 69
  fi
  "${atrex_python}" "${atrex_runtime_root}/examples/shared/prepare_campaign.py" \
    --state-dir "${atrex_state_dir}" \
    --runtime-template "${atrex_runtime_template}" \
    --campaign-template "${atrex_campaign_template}" \
    --config "${atrex_config_file}" \
    --campaign "${atrex_campaign_file}"
}

atrex_example_campaign_id() {
  "${atrex_python}" -c '
import json
import re
import sys

value = json.loads(open(sys.argv[1], encoding="utf-8").read())
campaign_id = value.get("campaign_id")
if not isinstance(campaign_id, str) or re.fullmatch(r"campaign_[0-9a-f]{32}", campaign_id) is None:
    raise SystemExit("saved Bootstrap result has no valid campaign_id")
print(campaign_id)
' "${atrex_bootstrap_result_file}"
}

atrex_example_lineage_id() {
  "${atrex_python}" -c '
import json
import re
import sys

value = json.loads(open(sys.argv[1], encoding="utf-8").read())
lineages = value.get("lineages")
if not isinstance(lineages, list) or len(lineages) != 1:
    raise SystemExit("saved Bootstrap result must contain exactly one lineage")
lineage_id = lineages[0].get("lineage_id")
if not isinstance(lineage_id, str) or re.fullmatch(r"lineage_[0-9a-f]{32}", lineage_id) is None:
    raise SystemExit("saved Bootstrap result has no valid lineage_id")
print(lineage_id)
' "${atrex_bootstrap_result_file}"
}

atrex_example_require_runtime_health() {
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to verify Runtime health" >&2
    return 69
  fi
  local runtime_host="${ATREX_RUNTIME_HOST:-127.0.0.1}"
  local runtime_port="${ATREX_RUNTIME_PORT:-8765}"
  if ! curl --fail --silent --show-error --max-time 10 \
    "http://${runtime_host}:${runtime_port}/healthz" >/dev/null; then
    echo "Runtime is not reachable at http://${runtime_host}:${runtime_port}." >&2
    return 69
  fi
}

atrex_example_bootstrap_campaign() {
  if [[ ! -f "${atrex_config_file}" || ! -f "${atrex_campaign_file}" ]]; then
    echo "prepared Runtime config or Campaign definition is missing" >&2
    return 66
  fi
  if ! command -v "${atrex_runtime_cli}" >/dev/null 2>&1; then
    echo "Runtime CLI not found in PATH: ${atrex_runtime_cli}" >&2
    return 69
  fi
  atrex_example_require_runtime_health || return

  mkdir -p "$(dirname -- "${atrex_bootstrap_result_file}")" || return
  local temporary="${atrex_bootstrap_result_file}.tmp.$$"
  trap 'rm -f "${temporary}"' RETURN
  umask 077
  "${atrex_runtime_cli}" bootstrap \
    --config "${atrex_config_file}" \
    --campaign "${atrex_campaign_file}" | tee "${temporary}"
  local bootstrap_status="${PIPESTATUS[0]}"
  if (( bootstrap_status != 0 )); then
    rm -f "${temporary}"
    trap - RETURN
    return "${bootstrap_status}"
  fi
  mv "${temporary}" "${atrex_bootstrap_result_file}"
  trap - RETURN

  echo
  echo "Bootstrap result saved to: ${atrex_bootstrap_result_file}"
  echo "Campaign: $(atrex_example_campaign_id)"
}

atrex_example_ensure_bootstrapped_campaign() {
  if [[ -f "${atrex_bootstrap_result_file}" ]]; then
    local saved_lineage_id=""
    if saved_lineage_id="$(atrex_example_lineage_id 2>/dev/null)" \
      && "${atrex_runtime_cli}" list-epochs \
        --config "${atrex_config_file}" \
        --lineage "${saved_lineage_id}" \
        --format json >/dev/null 2>&1; then
      echo "Reusing registered Campaign: $(atrex_example_campaign_id)"
      echo "Reusing registered Lineage: ${saved_lineage_id}"
      return 0
    fi
    echo "Saved Bootstrap result is stale; bootstrapping the prepared Campaign again."
  fi
  atrex_example_bootstrap_campaign
}

atrex_example_with_runtime() (
  set -euo pipefail
  if ! command -v "${atrex_runtime_cli}" >/dev/null 2>&1; then
    echo "Runtime CLI not found in PATH: ${atrex_runtime_cli}" >&2
    exit 69
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to wait for Runtime readiness" >&2
    exit 69
  fi

  local runtime_host="${ATREX_RUNTIME_HOST:-127.0.0.1}"
  local runtime_port="${ATREX_RUNTIME_PORT:-8765}"
  local runtime_url="http://${runtime_host}:${runtime_port}"
  local runtime_log="${atrex_state_dir}/runtime.log"
  local start_timeout="${ATREX_RUNTIME_START_TIMEOUT:-60}"
  if [[ ! "${start_timeout}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ATREX_RUNTIME_START_TIMEOUT must be a positive integer" >&2
    exit 64
  fi
  if curl --fail --silent --max-time 2 "${runtime_url}/healthz" >/dev/null 2>&1; then
    echo "A Runtime is already reachable at ${runtime_url}." >&2
    echo "This wrapper only manages a Runtime process that it starts itself." >&2
    exit 69
  fi

  mkdir -p "${atrex_state_dir}"
  umask 077
  "${atrex_runtime_cli}" serve --config "${atrex_config_file}" >"${runtime_log}" 2>&1 &
  local runtime_pid=$!

  cleanup_runtime() {
    local status=$?
    trap - EXIT
    if kill -0 "${runtime_pid}" >/dev/null 2>&1; then
      kill "${runtime_pid}" >/dev/null 2>&1 || true
      for _ in {1..50}; do
        if ! kill -0 "${runtime_pid}" >/dev/null 2>&1; then
          break
        fi
        sleep 0.2
      done
      if kill -0 "${runtime_pid}" >/dev/null 2>&1; then
        echo "Runtime did not stop gracefully; terminating owned pid ${runtime_pid}." >&2
        kill -KILL "${runtime_pid}" >/dev/null 2>&1 || true
      fi
      wait "${runtime_pid}" >/dev/null 2>&1 || true
    fi
    echo "Runtime stopped (pid ${runtime_pid})."
    exit "${status}"
  }
  trap cleanup_runtime EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  echo "Runtime starting at ${runtime_url} (pid ${runtime_pid})"
  echo "Runtime log: ${runtime_log}"
  local deadline=$((SECONDS + start_timeout))
  while ! curl --fail --silent --max-time 2 "${runtime_url}/healthz" >/dev/null 2>&1; do
    if ! kill -0 "${runtime_pid}" >/dev/null 2>&1; then
      echo "Runtime exited before becoming healthy:" >&2
      tail -n 80 "${runtime_log}" >&2 || true
      exit 1
    fi
    if (( SECONDS >= deadline )); then
      echo "Runtime did not become healthy within ${start_timeout} seconds:" >&2
      tail -n 80 "${runtime_log}" >&2 || true
      exit 1
    fi
    sleep 1
  done
  echo "Runtime is healthy."
  echo

  set +e
  (set -e; "$@")
  local status=$?
  set -e
  if (( status != 0 )); then
    echo "Example execution failed; recent Runtime log follows:" >&2
    tail -n 80 "${runtime_log}" >&2 || true
    exit "${status}"
  fi
)
