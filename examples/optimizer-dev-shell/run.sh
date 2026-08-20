#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "The Optimizer sandbox example requires Linux." >&2
  exit 69
fi
if (( EUID != 0 )); then
  if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true >/dev/null 2>&1; then
    echo "The sandbox example requires passwordless sudo for its trusted launcher." >&2
    exit 77
  fi
  export ATREX_SANDBOX_WORKER_USER="${ATREX_SANDBOX_WORKER_USER:-$(id -un)}"
  export ATREX_SANDBOX_HOST_HOME="${ATREX_SANDBOX_HOST_HOME:-${HOME}}"
  export ATREX_SANDBOX_HOST_PATH="${ATREX_SANDBOX_HOST_PATH:-${PATH}}"
  export ATREX_PYTHON="$(command -v "${ATREX_PYTHON:-python3}")"
  export ATREX_RUNTIME_CLI="$(command -v "${ATREX_RUNTIME_CLI:-atrex-kernel-agent-runtime}")"
  exec sudo --preserve-env=AGATE_URL,AGATE_AK,AGATE_SK,AGATE_GPU,AGATE_HTTP_TIMEOUT,AGATE_WAIT_TIMEOUT,AGATE_HEALTH_CHECK_INTERVAL,ATREX_PYTHON,ATREX_RUNTIME_CLI,ATREX_OPTIMIZER_AGENT_BACKEND,ATREX_SANDBOX_WORKER_USER,ATREX_SANDBOX_HOST_HOME,ATREX_SANDBOX_HOST_PATH,ATREX_RUNTIME_HOST,ATREX_RUNTIME_PORT,ATREX_RUNTIME_START_TIMEOUT,ATREX_WIKI_URL,ATREX_WIKI_TOKEN_ENV,ATREX_CAPABILITY_SIGNING_KEY,ATREX_ADMIN_BEARER_TOKEN,ANTHROPIC_AUTH_TOKEN,ANTHROPIC_API_KEY,ANTHROPIC_BASE_URL,ANTHROPIC_MODEL,CODEX_HOME,QODER_PERSONAL_ACCESS_TOKEN \
    -- bash "$0" "$@"
fi
if [[ -z "${ATREX_SANDBOX_WORKER_USER:-${SUDO_USER:-}}" ]]; then
  echo "Set ATREX_SANDBOX_WORKER_USER when invoking the sandbox example directly as root." >&2
  exit 64
fi
if [[ -z "${ATREX_SANDBOX_HOST_HOME:-}" || -z "${ATREX_SANDBOX_HOST_PATH:-}" ]]; then
  echo "Sandbox Backend Home/PATH were not preserved across the trusted sudo boundary." >&2
  exit 64
fi
export HOME="${ATREX_SANDBOX_HOST_HOME}"
export PATH="${ATREX_SANDBOX_HOST_PATH}"

temporary_state="$(mktemp -d "${TMPDIR:-/tmp}/atrex-optimizer-dev-shell.XXXXXX")"
chmod 0711 "${temporary_state}"
export ATREX_OPTIMIZER_DEV_SHELL_STATE_DIR="${temporary_state}"

cleanup_workspace() {
  local status=$?
  trap - EXIT
  if [[ -d "${temporary_state}" ]]; then
    chmod -R u+w -- "${temporary_state}" 2>/dev/null || true
    if ! rm -rf -- "${temporary_state}"; then
      echo "Failed to destroy temporary Optimizer workspace: ${temporary_state}" >&2
      status=1
    fi
  fi
  if [[ ! -e "${temporary_state}" ]]; then
    echo "Temporary Optimizer workspace destroyed: ${temporary_state}"
  fi
  exit "${status}"
}
trap cleanup_workspace EXIT

# shellcheck source=common.sh
source "${script_dir}/common.sh"

shell_name="zsh"
backend="${ATREX_OPTIMIZER_AGENT_BACKEND:-qodercli}"
shell_seen=false
backend_seen=false
for argument in "$@"; do
  case "${argument}" in
    zsh|bash)
      if [[ "${shell_seen}" == true ]]; then
        echo "dev-shell family was specified more than once" >&2
        exit 64
      fi
      shell_name="${argument}"
      shell_seen=true
      ;;
    claude|codex|qodercli|pi)
      if [[ "${backend_seen}" == true ]]; then
        echo "Optimizer Backend was specified more than once" >&2
        exit 64
      fi
      backend="${argument}"
      backend_seen=true
      ;;
    *)
      echo "usage: $0 [zsh|bash] [claude|codex|qodercli|pi]" >&2
      exit 64
      ;;
  esac
done
export ATREX_OPTIMIZER_AGENT_BACKEND="${backend}"

atrex_example_require_remote_agate
atrex_example_load_local_secrets
# Secret creation intentionally restores 0700; the non-root Sandbox Worker
# needs traverse-only access to reach its own nested workspace.
chmod 0711 "${temporary_state}"
atrex_example_prepare_inputs
atrex_example_with_runtime bash "${script_dir}/open-shell.sh" \
  --inputs-already-prepared "${shell_name}"
