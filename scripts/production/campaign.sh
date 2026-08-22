#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
  cat >&2 <<EOF
usage:
  $0 start --service-workspace DIR --kernel KERNEL --backend BACKEND [options]
  $0 stop|status|restart --workspace DIR

start options:
  --target-epoch N
  --workspace DIR
  --hardware-target GPU
  --seed-source FILE
  --optimizer-model MODEL
  --evolver-model MODEL
  --launcher-mode sandbox|container
  --env-file FILE

The shared Runtime and Wiki must already be running. A managed Campaign task
runs in the background and stores its PID, log, launch arguments, and terminal
state under WORKSPACE/campaign-run/.
EOF
}

if (( $# == 0 )); then
  usage
  exit 64
fi

action="$1"
shift
case "${action}" in
  start|stop|status|restart|__run) ;;
  *) usage; exit 64 ;;
esac
if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Managed production Campaign execution requires Linux." >&2
  exit 69
fi

kernel=""
backend=""
target_epoch="10"
workspace=""
service_workspace=""
hardware_target="${AGATE_GPU:-}"
seed_source=""
optimizer_model=""
evolver_model=""
launcher_mode="${ATREX_LAUNCHER_MODE:-}"
env_file=""
while (( $# > 0 )); do
  case "$1" in
    --kernel) kernel="${2:-}"; shift 2 ;;
    --backend) backend="${2:-}"; shift 2 ;;
    --target-epoch) target_epoch="${2:-}"; shift 2 ;;
    --workspace) workspace="${2:-}"; shift 2 ;;
    --service-workspace) service_workspace="${2:-}"; shift 2 ;;
    --hardware-target) hardware_target="${2:-}"; shift 2 ;;
    --seed-source) seed_source="${2:-}"; shift 2 ;;
    --optimizer-model) optimizer_model="${2:-}"; shift 2 ;;
    --evolver-model) evolver_model="${2:-}"; shift 2 ;;
    --launcher-mode) launcher_mode="${2:-}"; shift 2 ;;
    --env-file) env_file="${2:-}"; shift 2 ;;
    *) usage; exit 64 ;;
  esac
done

campaign_control_paths() {
  campaign_control="${atrex_prod_workspace}/campaign-run"
  campaign_pid_file="${campaign_control}/runner.pid"
  campaign_log="${campaign_control}/runner.log"
  campaign_args_file="${campaign_control}/launch-args.json"
  campaign_state_file="${campaign_control}/state.json"
}

write_state() {
  local state="$1"
  local exit_code="${2:-}"
  local temporary="${campaign_state_file}.tmp.$$"
  "${atrex_prod_python}" -c '
import datetime, json, sys
value = {
    "schema_version": 1,
    "state": sys.argv[2],
    "updated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
}
if sys.argv[3]:
    value["exit_code"] = int(sys.argv[3])
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(value, stream, indent=2, sort_keys=True)
    stream.write("\n")
' "${temporary}" "${state}" "${exit_code}"
  chmod 0600 "${temporary}"
  mv "${temporary}" "${campaign_state_file}"
}

write_launch_args() {
  local destination="$1"
  shift
  "${atrex_prod_python}" -c '
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(sys.argv[2:], stream, indent=2)
    stream.write("\n")
' "${destination}" "$@"
  chmod 0600 "${destination}"
}

load_launch_args() {
  launch_args=()
  while IFS= read -r -d '' value; do
    launch_args+=("${value}")
  done < <("${atrex_prod_python}" -c '
import json, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
    raise SystemExit("invalid managed Campaign launch arguments")
for item in value:
    sys.stdout.buffer.write(item.encode() + b"\0")
' "${campaign_args_file}")
}

workspace_process_pids() {
  "${atrex_prod_python}" -c '
import os, pathlib, sys
target = str(pathlib.Path(sys.argv[1]).resolve())
excluded = {os.getpid()}
ancestor = os.getppid()
while ancestor > 1 and ancestor not in excluded:
    excluded.add(ancestor)
    try:
        stat = (pathlib.Path("/proc") / str(ancestor) / "stat").read_text()
        closing = stat.rfind(")")
        fields = stat[closing + 1:].split()
        ancestor = int(fields[1])
    except (FileNotFoundError, PermissionError, ProcessLookupError, ValueError, IndexError):
        break
for entry in pathlib.Path("/proc").iterdir():
    if not entry.name.isdigit() or int(entry.name) in excluded:
        continue
    try:
        arguments = (entry / "cmdline").read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        continue
    decoded = [item.decode(errors="surrogateescape") for item in arguments if item]
    if any(item == target or item.startswith(target + os.sep) for item in decoded):
        print(entry.name)
' "${atrex_prod_workspace}"
}

campaign_running() {
  if atrex_prod_pid_alive "${campaign_pid_file}"; then
    return 0
  fi
  local pids
  pids="$(workspace_process_pids)"
  [[ -n "${pids}" ]]
}

stop_campaign() {
  local pids=()
  while IFS= read -r pid; do
    [[ -n "${pid}" ]] && pids+=("${pid}")
  done < <(workspace_process_pids)
  if atrex_prod_pid_alive "${campaign_pid_file}"; then
    local runner_pid
    runner_pid="$(atrex_prod_read_pid "${campaign_pid_file}")"
    if [[ ! " ${pids[*]:-} " =~ [[:space:]]${runner_pid}[[:space:]] ]]; then
      pids+=("${runner_pid}")
    fi
  fi
  if (( ${#pids[@]} == 0 )); then
    rm -f -- "${campaign_pid_file}"
    write_state "stopped"
    echo "Campaign task: stopped"
    return 0
  fi
  echo "Campaign task: sending SIGTERM to ${#pids[@]} process(es)"
  kill -TERM "${pids[@]}" >/dev/null 2>&1 || \
    sudo kill -TERM "${pids[@]}" >/dev/null 2>&1 || true
  local _
  for _ in {1..100}; do
    local remaining=()
    local pid
    for pid in "${pids[@]}"; do
      [[ -d "/proc/${pid}" ]] && remaining+=("${pid}")
    done
    pids=("${remaining[@]}")
    (( ${#pids[@]} == 0 )) && break
    sleep 0.2
  done
  if (( ${#pids[@]} > 0 )); then
    echo "Campaign task: sending SIGKILL to ${#pids[@]} remaining process(es)" >&2
    kill -KILL "${pids[@]}" >/dev/null 2>&1 || \
      sudo kill -KILL "${pids[@]}" >/dev/null 2>&1 || true
    sleep 0.2
  fi
  local survivors
  survivors="$(workspace_process_pids)"
  if [[ -n "${survivors}" ]]; then
    echo "Campaign task did not stop; remaining pids: ${survivors//$'\n'/,}" >&2
    return 1
  fi
  rm -f -- "${campaign_pid_file}"
  write_state "stopped"
  echo "Campaign task: stopped"
}

start_prepared() {
  if campaign_running; then
    echo "Campaign task is already running: ${atrex_prod_workspace}" >&2
    return 69
  fi
  rm -f -- "${campaign_pid_file}"
  write_state "starting"
  nohup bash "${script_dir}/campaign.sh" __run --workspace "${atrex_prod_workspace}" \
    >"${campaign_log}" 2>&1 &
  local pid="$!"
  if ! atrex_prod_write_pid "${campaign_pid_file}" "${pid}"; then
    kill "${pid}" >/dev/null 2>&1 || true
    return 1
  fi
  sleep 1
  if ! campaign_running; then
    rm -f -- "${campaign_pid_file}"
    echo "Campaign task exited during startup; recent log follows:" >&2
    tail -n 100 "${campaign_log}" >&2 || true
    return 1
  fi
  echo "Campaign task: started (pid ${pid})"
  echo "Campaign workspace: ${atrex_prod_workspace}"
  echo "Campaign log: ${campaign_log}"
}

prepare_start() {
  if [[ -z "${kernel}" || -z "${backend}" || -z "${service_workspace}" ]]; then
    usage
    return 64
  fi
  if [[ ! "${target_epoch}" =~ ^[1-9][0-9]*$ ]]; then
    echo "--target-epoch must be a positive integer" >&2
    return 64
  fi
  case "${backend}" in claude|codex|qodercli|pi) ;; *) usage; return 64 ;; esac
  if [[ -n "${env_file}" ]]; then
    if [[ ! -f "${env_file}" ]]; then
      echo "Environment file not found: ${env_file}" >&2
      return 66
    fi
    # shellcheck disable=SC1090
    source "${env_file}"
    hardware_target="${hardware_target:-${AGATE_GPU:-}}"
    env_file="$(atrex_prod_absolute_path "${env_file}")"
  fi
  service_workspace="$(atrex_prod_absolute_path "${service_workspace}")"
  local workspace_output
  workspace_output="$(mktemp "${TMPDIR:-/tmp}/atrex-production-workspace.XXXXXX")"
  local prepare_args=(
    --kernel "${kernel}"
    --backend "${backend}"
    --service-workspace "${service_workspace}"
    --workspace-output "${workspace_output}"
  )
  [[ -n "${seed_source}" ]] && prepare_args+=(--seed-source "${seed_source}")
  [[ -n "${workspace}" ]] && prepare_args+=(--workspace "${workspace}")
  [[ -n "${hardware_target}" ]] && prepare_args+=(--hardware-target "${hardware_target}")
  [[ -n "${optimizer_model}" ]] && prepare_args+=(--optimizer-model "${optimizer_model}")
  [[ -n "${evolver_model}" ]] && prepare_args+=(--evolver-model "${evolver_model}")
  [[ -n "${launcher_mode}" ]] && prepare_args+=(--launcher-mode "${launcher_mode}")
  if ! "${atrex_prod_python}" "${script_dir}/prepare.py" "${prepare_args[@]}"; then
    rm -f -- "${workspace_output}"
    return 1
  fi
  workspace="$(<"${workspace_output}")"
  rm -f -- "${workspace_output}"
  atrex_prod_workspace_paths "${workspace}"
  atrex_prod_require_workspace
  campaign_control_paths
  mkdir -p "${campaign_control}"
  chmod 0700 "${campaign_control}"
  if campaign_running; then
    echo "Campaign task is already running: ${atrex_prod_workspace}" >&2
    return 69
  fi
  local run_args=(
    --prepared
    --kernel "${kernel}"
    --backend "${backend}"
    --target-epoch "${target_epoch}"
    --workspace "${atrex_prod_workspace}"
    --hardware-target "${hardware_target}"
    --service-workspace "${service_workspace}"
    --external-services
  )
  [[ -n "${seed_source}" ]] && run_args+=(--seed-source "${seed_source}")
  [[ -n "${optimizer_model}" ]] && run_args+=(--optimizer-model "${optimizer_model}")
  [[ -n "${evolver_model}" ]] && run_args+=(--evolver-model "${evolver_model}")
  [[ -n "${launcher_mode}" ]] && run_args+=(--launcher-mode "${launcher_mode}")
  [[ -n "${env_file}" ]] && run_args+=(--env-file "${env_file}")
  write_launch_args "${campaign_args_file}" "${run_args[@]}"
  start_prepared
}

require_managed_workspace() {
  if [[ -z "${workspace}" ]]; then
    usage
    return 64
  fi
  atrex_prod_workspace_paths "${workspace}"
  atrex_prod_require_workspace
  campaign_control_paths
  if [[ ! -d "${campaign_control}" ]]; then
    echo "Campaign task has not been started under management: ${atrex_prod_workspace}" >&2
    return 66
  fi
}

case "${action}" in
  start)
    prepare_start
    ;;
  stop)
    require_managed_workspace
    stop_campaign
    ;;
  status)
    require_managed_workspace
    if campaign_running; then
      pids="$(workspace_process_pids | paste -sd, -)"
      echo "Campaign task: running${pids:+ (pids ${pids})}"
    elif [[ -f "${campaign_state_file}" ]]; then
      "${atrex_prod_python}" -c '
import json, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
text=f"Campaign task: {value.get('"'"'state'"'"', '"'"'unknown'"'"')}"
if "exit_code" in value:
    text += f" (exit_code {value['"'"'exit_code'"'"']})"
print(text)
print(f"Updated: {value.get('"'"'updated_at'"'"', '"'"'-'"'"')}")
' "${campaign_state_file}"
    else
      echo "Campaign task: stopped"
    fi
    echo "Campaign workspace: ${atrex_prod_workspace}"
    echo "Campaign log: ${campaign_log}"
    ;;
  restart)
    require_managed_workspace
    if [[ ! -f "${campaign_args_file}" ]]; then
      echo "Managed Campaign launch arguments are missing: ${campaign_args_file}" >&2
      exit 66
    fi
    stop_campaign
    start_prepared
    ;;
  __run)
    require_managed_workspace
    if [[ ! -f "${campaign_args_file}" ]]; then
      echo "Managed Campaign launch arguments are missing: ${campaign_args_file}" >&2
      exit 66
    fi
    write_state "running"
    load_launch_args
    set +e
    bash "${script_dir}/run.sh" "${launch_args[@]}"
    result="$?"
    set -e
    if (( result == 0 )); then
      write_state "succeeded" "${result}"
    else
      write_state "failed" "${result}"
    fi
    if atrex_prod_pid_alive "${campaign_pid_file}" \
      && [[ "$(atrex_prod_read_pid "${campaign_pid_file}")" == "$$" ]]; then
      rm -f -- "${campaign_pid_file}"
    fi
    exit "${result}"
    ;;
esac
