#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${script_dir}/common.sh"

usage() {
  cat >&2 <<EOF
usage: $0 --kernel KERNEL --backend BACKEND [options]

Required:
  --kernel PATH|SUITE/OPERATOR  Atrex-Bench operator directory
  --backend NAME               claude, codex, qodercli, or pi
Options:
  --target-epoch N             absolute Epoch target to complete; defaults to 10
  --workspace DIR
  --service-workspace DIR      reuse its running Runtime/Wiki without managing services
  --hardware-target GPU        defaults to AGATE_GPU
  --seed-source FILE           defaults to reference.py
  --optimizer-model MODEL
  --evolver-model MODEL
  --launcher-mode sandbox|container
  --env-file FILE              optional trusted shell environment file
EOF
}

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
prepared=false
external_services=false
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
    --prepared) prepared=true; shift ;;
    --external-services) external_services=true; shift ;;
    *) usage; exit 64 ;;
  esac
done
if [[ -z "${kernel}" || -z "${backend}" ]]; then
  usage
  exit 64
fi
if [[ ! "${target_epoch}" =~ ^[1-9][0-9]*$ ]]; then
  echo "--target-epoch must be a positive integer" >&2
  exit 64
fi
case "${backend}" in claude|codex|qodercli|pi) ;; *) usage; exit 64 ;; esac
if [[ "${external_services}" == true && -z "${service_workspace}" ]]; then
  echo "--service-workspace is required when services are managed externally" >&2
  exit 64
fi

if [[ -n "${env_file}" ]]; then
  if [[ ! -f "${env_file}" ]]; then
    echo "Environment file not found: ${env_file}" >&2
    exit 66
  fi
  # shellcheck disable=SC1090
  source "${env_file}"
  hardware_target="${hardware_target:-${AGATE_GPU:-}}"
fi

if [[ "${prepared}" == false ]]; then
  workspace_output="$(mktemp "${TMPDIR:-/tmp}/atrex-production-workspace.XXXXXX")"
  trap 'rm -f -- "${workspace_output}"' EXIT
  prepare_args=(
    --kernel "${kernel}"
    --backend "${backend}"
    --workspace-output "${workspace_output}"
  )
  [[ -n "${seed_source}" ]] && prepare_args+=(--seed-source "${seed_source}")
  [[ -n "${workspace}" ]] && prepare_args+=(--workspace "${workspace}")
  [[ -n "${service_workspace}" ]] && \
    prepare_args+=(--service-workspace "${service_workspace}")
  [[ -n "${hardware_target}" ]] && prepare_args+=(--hardware-target "${hardware_target}")
  [[ -n "${optimizer_model}" ]] && prepare_args+=(--optimizer-model "${optimizer_model}")
  [[ -n "${evolver_model}" ]] && prepare_args+=(--evolver-model "${evolver_model}")
  [[ -n "${launcher_mode}" ]] && prepare_args+=(--launcher-mode "${launcher_mode}")
  "${atrex_prod_python}" "${script_dir}/prepare.py" "${prepare_args[@]}"
  workspace="$(<"${workspace_output}")"
  rm -f -- "${workspace_output}"
  trap - EXIT
  resume_args=(
    --prepared
    --kernel "${kernel}"
    --backend "${backend}"
    --target-epoch "${target_epoch}"
    --workspace "${workspace}"
    --hardware-target "${hardware_target}"
  )
  [[ -n "${seed_source}" ]] && resume_args+=(--seed-source "${seed_source}")
  [[ -n "${optimizer_model}" ]] && resume_args+=(--optimizer-model "${optimizer_model}")
  [[ -n "${evolver_model}" ]] && resume_args+=(--evolver-model "${evolver_model}")
  [[ -n "${launcher_mode}" ]] && resume_args+=(--launcher-mode "${launcher_mode}")
  [[ -n "${env_file}" ]] && resume_args+=(--env-file "${env_file}")
  [[ -n "${service_workspace}" ]] && \
    resume_args+=(--service-workspace "${service_workspace}")
  [[ "${external_services}" == true ]] && resume_args+=(--external-services)
  exec "$0" "${resume_args[@]}"
fi

if [[ -z "${workspace}" ]]; then
  echo "internal error: prepared execution requires --workspace" >&2
  exit 70
fi
atrex_prod_workspace_paths "${workspace}"
atrex_prod_require_workspace
atrex_prod_require_policy_gate
atrex_prod_load_environment "${env_file}"
# A sudo-resumed production invocation inherits the trusted caller HOME/PATH in
# dedicated variables because sudo may replace PATH with secure_path. Restore
# them before resolving the configured provider CLI.
atrex_prod_restore_host_environment
atrex_prod_require_agate
atrex_prod_require_commands
atrex_prod_require_backend
root_args=(
  --prepared
  --kernel "${kernel}"
  --backend "${backend}"
  --target-epoch "${target_epoch}"
  --workspace "${atrex_prod_workspace}"
  --hardware-target "${hardware_target}"
)
[[ -n "${seed_source}" ]] && root_args+=(--seed-source "${seed_source}")
[[ -n "${optimizer_model}" ]] && root_args+=(--optimizer-model "${optimizer_model}")
[[ -n "${evolver_model}" ]] && root_args+=(--evolver-model "${evolver_model}")
[[ -n "${launcher_mode}" ]] && root_args+=(--launcher-mode "${launcher_mode}")
[[ -n "${env_file}" ]] && root_args+=(--env-file "${env_file}")
[[ -n "${service_workspace}" ]] && \
  root_args+=(--service-workspace "${service_workspace}")
[[ "${external_services}" == true ]] && root_args+=(--external-services)
atrex_prod_escalate "$0" "${root_args[@]}"

if [[ "${external_services}" == true ]]; then
  atrex_prod_require_control_plane_ready "${atrex_prod_config}"
else
  service_args=(start --workspace "${atrex_prod_workspace}")
  [[ -n "${env_file}" ]] && service_args+=(--env-file "${env_file}")
  bash "${script_dir}/services.sh" "${service_args[@]}"
fi

echo
echo "Production content policy gate: enabled"
echo "Running independent CUDA, Triton, and CuteDSL pipelines in parallel."
echo "Each DSL enters its Campaign as soon as its own Bootstrap succeeds."
echo "A failure in one DSL does not stop or delay the other DSL pipelines."

dsls=(cuda triton cutedsl)
job_pids=()
job_dsls=()

stop_jobs() {
  local stopped_pids=("${job_pids[@]:-}")
  job_pids=()
  job_dsls=()
  local pid
  for pid in "${stopped_pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
  wait >/dev/null 2>&1 || true
}

stop_jobs_for_signal() {
  trap - INT TERM
  stop_jobs
  exit 143
}
trap stop_jobs_for_signal INT TERM
trap stop_jobs EXIT

bootstrap_one() (
  set -o pipefail
  local dsl="$1"
  atrex_prod_dsl_paths "${dsl}"
  local temporary="${atrex_prod_bootstrap_result}.tmp.${BASHPID}"
  rm -f -- "${temporary}"
  : >"${atrex_prod_bootstrap_log}"
  echo "[${dsl}] Bootstrap started: ${atrex_prod_dsl_workspace}"
  set +e
  "${atrex_prod_cli}" bootstrap --config "${atrex_prod_config}" \
    --campaign "${atrex_prod_campaign}" \
    2> >(tee "${atrex_prod_bootstrap_log}" >&2) | tee "${temporary}"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  if (( statuses[0] != 0 || statuses[1] != 0 )); then
    rm -f -- "${temporary}"
    echo "[${dsl}] Bootstrap failed; log: ${atrex_prod_bootstrap_log}" >&2
    return 1
  fi
  mv "${temporary}" "${atrex_prod_bootstrap_result}"
  echo "[${dsl}] Bootstrap completed: ${atrex_prod_bootstrap_result}"
)

campaign_id_for() {
  local dsl="$1"
  atrex_prod_dsl_paths "${dsl}"
  "${atrex_prod_python}" -c '
import json, re, sys
value=json.load(open(sys.argv[1], encoding="utf-8"))
campaign=value.get("campaign_id", "")
if re.fullmatch(r"campaign_[0-9a-f]{32}", campaign) is None:
    raise SystemExit(f"Bootstrap result has no valid campaign_id: {sys.argv[1]}")
lineages=value.get("lineages", [])
if len(lineages) != 1:
    raise SystemExit(f"Bootstrap result does not own exactly one {sys.argv[2]} Lineage")
print(campaign)
' "${atrex_prod_bootstrap_result}" "${dsl}"
}

ablation_arm_labels() {
  if [[ ! -f "${atrex_prod_ablation_plan}" ]]; then
    return 0
  fi
  "${atrex_prod_python}" -c '
import json, re, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
if value.get("schema_version") != 2:
    raise SystemExit(f"unsupported ablation plan schema: {sys.argv[1]}")
if not value.get("enabled"):
    raise SystemExit(0)
for arm in value.get("arms", []):
    label = arm["label"]
    if re.fullmatch(r"ablation-[a-z0-9-]+", label) is None:
        raise SystemExit(f"invalid ablation arm label: {label}")
    print(label)
' "${atrex_prod_ablation_plan}"
}

seed_arm() (
  set -o pipefail
  local dsl="$1"
  local label="$2"
  atrex_prod_arm_paths "${dsl}" "${label}"
  mkdir -p -- "${atrex_prod_arm_workspace}"
  # The arm clones the evolution Lineage's frozen baseline, so its spec only needs that
  # Lineage plus its own Trajectory shape.
  if ! "${atrex_prod_python}" -c '
import json, sys
bootstrap = json.load(open(sys.argv[1], encoding="utf-8"))
plan = json.load(open(sys.argv[2], encoding="utf-8"))
label = sys.argv[4]
lineages = bootstrap.get("lineages", [])
if len(lineages) != 1:
    raise SystemExit("Bootstrap result does not own exactly one Lineage")
arm = next(item for item in plan["arms"] if item["label"] == label)
json.dump(
    {
        "schema_version": 1,
        "creation_key": f"{label}-{sys.argv[5]}",
        "source_lineage_id": lineages[0]["lineage_id"],
        "attempts_per_trajectory": int(plan["attempts_per_trajectory"]),
        "trajectories_per_branch": int(arm["trajectories_per_branch"]),
        "ephemeral_agent_state": bool(arm["ephemeral_agent_state"]),
    },
    open(sys.argv[3], "w", encoding="utf-8"),
    indent=2,
    sort_keys=True,
)
' "${atrex_prod_bootstrap_result}" "${atrex_prod_ablation_plan}" "${atrex_prod_arm_spec}" \
    "${label}" "${dsl}"; then
    echo "[${dsl}/${label}] Could not write the arm spec." >&2
    return 1
  fi
  local temporary="${atrex_prod_arm_seed_result}.tmp.${BASHPID}"
  rm -f -- "${temporary}"
  set +e
  "${atrex_prod_cli}" seed-ablation-arm --config "${atrex_prod_config}" \
    --spec "${atrex_prod_arm_spec}" | tee "${temporary}"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  if (( statuses[0] != 0 || statuses[1] != 0 )); then
    rm -f -- "${temporary}"
    echo "[${dsl}/${label}] Arm seeding failed." >&2
    return 1
  fi
  mv "${temporary}" "${atrex_prod_arm_seed_result}"
  echo "[${dsl}/${label}] Arm seeded: ${atrex_prod_arm_seed_result}"
)

run_arm() (
  set -o pipefail
  local dsl="$1"
  local label="$2"
  atrex_prod_arm_paths "${dsl}" "${label}"
  local campaign_id
  campaign_id="$("${atrex_prod_python}" -c '
import json, re, sys
value = json.load(open(sys.argv[1], encoding="utf-8"))
campaign = value.get("campaign_id", "")
if re.fullmatch(r"campaign_[0-9a-f]{32}", campaign) is None:
    raise SystemExit(f"Arm seed result has no valid campaign_id: {sys.argv[1]}")
print(campaign)
' "${atrex_prod_arm_seed_result}")" || return 1
  local temporary="${atrex_prod_arm_campaign_result}.tmp.${BASHPID}"
  rm -f -- "${temporary}"
  : >"${atrex_prod_arm_log}"
  echo "[${dsl}/${label}] Campaign ${campaign_id} started through Epoch ${target_epoch}."
  set +e
  "${atrex_prod_cli}" run-campaign --config "${atrex_prod_config}" \
    --campaign "${campaign_id}" --target-epoch "${target_epoch}" \
    2> >(tee "${atrex_prod_arm_log}" >&2) | tee "${temporary}"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  if (( statuses[0] != 0 || statuses[1] != 0 )); then
    rm -f -- "${temporary}"
    echo "[${dsl}/${label}] Campaign failed; log: ${atrex_prod_arm_log}" >&2
    return 1
  fi
  mv "${temporary}" "${atrex_prod_arm_campaign_result}"
  echo "[${dsl}/${label}] Campaign completed: ${atrex_prod_arm_campaign_result}"
)

run_one() (
  set -o pipefail
  local dsl="$1"
  local campaign_id="$2"
  atrex_prod_dsl_paths "${dsl}"
  local temporary="${atrex_prod_campaign_result}.tmp.${BASHPID}"
  rm -f -- "${temporary}"
  : >"${atrex_prod_campaign_log}"
  echo "[${dsl}] Campaign ${campaign_id} started through Epoch ${target_epoch}."
  set +e
  "${atrex_prod_cli}" run-campaign --config "${atrex_prod_config}" \
    --campaign "${campaign_id}" --target-epoch "${target_epoch}" \
    2> >(tee "${atrex_prod_campaign_log}" >&2) | tee "${temporary}"
  local statuses=("${PIPESTATUS[@]}")
  set -e
  if (( statuses[0] != 0 || statuses[1] != 0 )); then
    rm -f -- "${temporary}"
    echo "[${dsl}] Campaign failed; log: ${atrex_prod_campaign_log}" >&2
    return 1
  fi
  mv "${temporary}" "${atrex_prod_campaign_result}"
  echo "[${dsl}] Campaign completed: ${atrex_prod_campaign_result}"
)

run_dsl_pipeline() (
  local dsl="$1"
  if ! bootstrap_one "${dsl}"; then
    echo "[${dsl}] Pipeline stopped after Bootstrap failure; other DSLs continue." >&2
    return 1
  fi
  local campaign_id
  if ! campaign_id="$(campaign_id_for "${dsl}")"; then
    echo "[${dsl}] Pipeline could not resolve its bootstrapped Campaign ID." >&2
    return 1
  fi
  local arms=()
  local label
  if ! mapfile -t arms < <(ablation_arm_labels); then
    echo "[${dsl}] Pipeline could not read the ablation plan." >&2
    return 1
  fi
  # Seeding reuses the Bootstrap baseline's measurement, so it costs no GPU time and is
  # cheap to do serially before the Campaigns fan out.
  for label in "${arms[@]}"; do
    if ! seed_arm "${dsl}" "${label}"; then
      echo "[${dsl}] Pipeline stopped after ablation arm seeding failed." >&2
      return 1
    fi
  done
  echo "[${dsl}] Bootstrap succeeded; entering Epoch execution immediately."
  local pids=()
  local labels=()
  run_one "${dsl}" "${campaign_id}" &
  pids+=("$!")
  labels+=("evolution")
  for label in "${arms[@]}"; do
    run_arm "${dsl}" "${label}" &
    pids+=("$!")
    labels+=("${label}")
  done
  local failed=0
  local index
  for index in "${!pids[@]}"; do
    if ! wait "${pids[index]}"; then
      echo "[${dsl}/${labels[index]}] Campaign failed; other Campaigns continue." >&2
      failed=1
    fi
  done
  if (( failed != 0 )); then
    return 1
  fi
)

echo
echo "Each Campaign has 2 serial Attempts per Branch and targets Epoch ${target_epoch}."
echo "Epoch 1 is Active-only. From Epoch 2, Active and one Challenger run concurrently."

for dsl in "${dsls[@]}"; do
  run_dsl_pipeline "${dsl}" &
  job_pids+=("$!")
  job_dsls+=("${dsl}")
done
pipeline_failed=0
for index in "${!job_pids[@]}"; do
  if ! wait "${job_pids[index]}"; then
    echo "[${job_dsls[index]}] DSL pipeline failed." >&2
    pipeline_failed=1
  fi
done
job_pids=()
job_dsls=()
trap - INT TERM EXIT

summary_partial=()
if (( pipeline_failed != 0 )); then
  summary_partial+=(--allow-partial)
fi
"${atrex_prod_python}" "${script_dir}/summarize.py" \
  --workspace "${atrex_prod_workspace}" --phase bootstrap "${summary_partial[@]}"
"${atrex_prod_python}" "${script_dir}/summarize.py" \
  --workspace "${atrex_prod_workspace}" --phase campaign --target-epoch "${target_epoch}" \
  "${summary_partial[@]}"

echo
echo "Campaign summary: ${atrex_prod_campaign_summary}"
if [[ "${external_services}" == true ]]; then
  echo "Shared services were left untouched."
else
  echo "Services remain running; stop them with:"
  echo "  bash scripts/production/services.sh stop --workspace ${atrex_prod_workspace}"
fi
if (( pipeline_failed != 0 )); then
  echo "At least one DSL pipeline failed; every successful DSL was allowed to finish." >&2
  exit 1
fi
