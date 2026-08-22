#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
  echo "usage: $0 start|stop|restart|status --workspace DIR [--hardware-target GPU] [--env-file FILE]" >&2
}

if (( $# < 3 )); then
  usage
  exit 64
fi
action="$1"
shift
workspace=""
env_file=""
hardware_target="${AGATE_GPU:-}"
while (( $# > 0 )); do
  case "$1" in
    --workspace)
      (( $# >= 2 )) || { usage; exit 64; }
      workspace="$2"
      shift 2
      ;;
    --env-file)
      (( $# >= 2 )) || { usage; exit 64; }
      env_file="$2"
      shift 2
      ;;
    --hardware-target)
      (( $# >= 2 )) || { usage; exit 64; }
      hardware_target="$2"
      shift 2
      ;;
    *)
      usage
      exit 64
      ;;
  esac
done
case "${action}" in
  start|stop|restart|status) ;;
  *) usage; exit 64 ;;
esac
if [[ -z "${workspace}" ]]; then
  usage
  exit 64
fi

atrex_prod_workspace_paths "${workspace}"
if [[ "${action}" == "start" || "${action}" == "restart" ]]; then
  if [[ ! -f "${atrex_prod_config}" && ! -f "${atrex_prod_manifest}" ]]; then
    if [[ -n "${env_file}" ]]; then
      if [[ ! -f "${env_file}" ]]; then
        echo "Environment file not found: ${env_file}" >&2
        exit 66
      fi
      # shellcheck disable=SC1090
      source "${env_file}"
      hardware_target="${hardware_target:-${AGATE_GPU:-}}"
    fi
    if [[ -z "${hardware_target}" ]]; then
      echo "--hardware-target or AGATE_GPU is required to initialize services" >&2
      exit 64
    fi
    "${atrex_prod_python}" "${script_dir}/prepare.py" \
      --services-only \
      --workspace "${atrex_prod_workspace}" \
      --hardware-target "${hardware_target}"
  fi
fi
atrex_prod_require_service_workspace
atrex_prod_load_environment "${env_file}"
atrex_prod_require_commands

if [[ "${action}" != "status" ]]; then
  root_args=("${action}" --workspace "${atrex_prod_workspace}")
  [[ -n "${hardware_target}" ]] && root_args+=(--hardware-target "${hardware_target}")
  [[ -n "${env_file}" ]] && root_args+=(--env-file "${env_file}")
  atrex_prod_escalate "$0" "${root_args[@]}"
fi
atrex_prod_restore_host_environment

mkdir -p "${atrex_prod_services}"
chmod 0700 "${atrex_prod_services}"
wiki_pid_file="${atrex_prod_services}/wiki.pid"
runtime_pid_file="${atrex_prod_services}/runtime.pid"
wiki_log="${atrex_prod_services}/wiki.log"
runtime_log="${atrex_prod_services}/runtime.log"
runtime_host="$(atrex_prod_json_value "${atrex_prod_config}" server.host)"
runtime_port="$(atrex_prod_json_value "${atrex_prod_config}" server.port)"
runtime_url="http://${runtime_host}:${runtime_port}"
wiki_url="$(atrex_prod_json_value "${atrex_prod_config}" gpu_wiki.base_url)"

start_local_wiki() {
  local managed=false
  case "${wiki_url}" in
    http://127.0.0.1:8091|http://localhost:8091) managed=true ;;
  esac
  if curl --fail --silent --max-time 3 "${wiki_url}/readyz" >/dev/null 2>&1; then
    if [[ "${managed}" == true ]] && ! atrex_prod_pid_alive "${wiki_pid_file}"; then
      echo "Local GPU Wiki port is served by an unowned process: ${wiki_url}" >&2
      return 69
    fi
    echo "GPU Wiki: ready at ${wiki_url}"
    return 0
  fi
  if [[ "${managed}" != true ]]; then
    echo "External GPU Wiki is not ready: ${wiki_url}" >&2
    return 69
  fi
  if atrex_prod_pid_alive "${wiki_pid_file}"; then
    echo "Owned Local GPU Wiki process is alive but not ready: ${wiki_url}" >&2
    return 69
  fi
  local wiki_source="${atrex_prod_root}/workspaces/local-wiki/src"
  local wiki_config="${atrex_prod_workspace}/local-wiki.json"
  if [[ ! -d "${wiki_source}" || ! -f "${wiki_config}" ]]; then
    echo "Local GPU Wiki implementation/config is unavailable." >&2
    return 66
  fi
  if ! "${atrex_prod_python}" -c 'import anyio, pydantic, uvicorn' >/dev/null 2>&1; then
    echo "Local GPU Wiki dependencies are missing from ${atrex_prod_python}." >&2
    return 69
  fi
  local wiki_user wiki_uid wiki_gid
  if ! wiki_user="$(atrex_prod_json_value "${atrex_prod_config}" \
    campaign.launcher.sandbox.worker_user 2>/dev/null)"; then
    wiki_user="$(atrex_prod_json_value "${atrex_prod_manifest}" worker_user)"
  fi
  local wiki_command=(
    env
    "HOME=${HOME}"
    "PYTHONPATH=${wiki_source}"
    "${atrex_prod_python}" -m atrex_local_wiki serve --config "${wiki_config}"
  )
  if (( EUID == 0 )); then
    if ! command -v setpriv >/dev/null 2>&1; then
      echo "Local GPU Wiki privilege drop requires setpriv." >&2
      return 69
    fi
    wiki_uid="$(id -u "${wiki_user}")" || return
    wiki_gid="$(id -g "${wiki_user}")" || return
    wiki_command=(
      setpriv --reuid="${wiki_uid}" --regid="${wiki_gid}" --init-groups
      "${wiki_command[@]}"
    )
  fi
  nohup "${wiki_command[@]}" >"${wiki_log}" 2>&1 &
  atrex_prod_write_pid "${wiki_pid_file}" "$!"
  if ! atrex_prod_wait_url "GPU Wiki" "${wiki_url}/readyz" "${wiki_pid_file}" 90; then
    tail -n 100 "${wiki_log}" >&2 || true
    return 1
  fi
  echo "GPU Wiki: started at ${wiki_url} (pid $(atrex_prod_read_pid "${wiki_pid_file}"))"
  echo "GPU Wiki log: ${wiki_log}"
}

start_runtime() {
  if curl --fail --silent --max-time 3 "${runtime_url}/healthz" >/dev/null 2>&1; then
    if ! atrex_prod_pid_alive "${runtime_pid_file}"; then
      echo "Runtime port is served by an unowned process: ${runtime_url}" >&2
      return 69
    fi
    echo "Runtime: healthy at ${runtime_url}"
    return 0
  fi
  if atrex_prod_pid_alive "${runtime_pid_file}"; then
    echo "Owned Runtime process is alive but not healthy: ${runtime_url}" >&2
    return 69
  fi
  nohup "${atrex_prod_cli}" serve --config "${atrex_prod_config}" \
    >"${runtime_log}" 2>&1 &
  atrex_prod_write_pid "${runtime_pid_file}" "$!"
  if ! atrex_prod_wait_url "Runtime" "${runtime_url}/healthz" "${runtime_pid_file}" 90; then
    tail -n 100 "${runtime_log}" >&2 || true
    return 1
  fi
  echo "Runtime: started at ${runtime_url} (pid $(atrex_prod_read_pid "${runtime_pid_file}"))"
  echo "Runtime log: ${runtime_log}"
}

stop_services() {
  atrex_prod_stop_pid "Runtime" "${runtime_pid_file}"
  atrex_prod_stop_pid "GPU Wiki" "${wiki_pid_file}"
}

start_services() {
  if ! start_local_wiki; then
    stop_services
    return 1
  fi
  if ! start_runtime; then
    stop_services
    return 1
  fi
}

show_status() {
  if curl --fail --silent --max-time 3 "${wiki_url}/readyz" >/dev/null 2>&1; then
    echo "GPU Wiki: ready (${wiki_url})"
  else
    echo "GPU Wiki: unavailable (${wiki_url})"
  fi
  if curl --fail --silent --max-time 3 "${runtime_url}/healthz" >/dev/null 2>&1; then
    echo "Runtime: healthy (${runtime_url})"
  else
    echo "Runtime: unavailable (${runtime_url})"
  fi
}

case "${action}" in
  start)
    atrex_prod_require_policy_gate
    atrex_prod_require_agate
    start_services
    ;;
  stop)
    stop_services
    ;;
  restart)
    stop_services
    atrex_prod_require_policy_gate
    atrex_prod_require_agate
    start_services
    ;;
  status)
    show_status
    ;;
esac
