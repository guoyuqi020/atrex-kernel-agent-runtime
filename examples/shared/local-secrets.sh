#!/usr/bin/env bash

atrex_shared_ensure_local_secrets() {
  local env_file="$1"
  local label="$2"
  if [[ -L "${env_file}" ]]; then
    echo "refusing symlinked ${label} environment: ${env_file}" >&2
    return 65
  fi
  if [[ -f "${env_file}" ]]; then
    chmod 600 "${env_file}"
    return
  fi
  if ! command -v openssl >/dev/null 2>&1; then
    echo "openssl is required to generate local Runtime secrets" >&2
    return 69
  fi

  local env_dir capability_key admin_token temporary
  env_dir="$(dirname -- "${env_file}")"
  mkdir -p "${env_dir}"
  chmod 700 "${env_dir}"
  capability_key="$(openssl rand -base64 32 | tr -d '\n')"
  admin_token="$(openssl rand -hex 32)"
  temporary="${env_file}.tmp.$$"
  umask 077
  {
    printf "export ATREX_CAPABILITY_SIGNING_KEY='%s'\n" "${capability_key}"
    printf "export ATREX_ADMIN_BEARER_TOKEN='%s'\n" "${admin_token}"
  } >"${temporary}"
  chmod 600 "${temporary}"
  if [[ -e "${env_file}" ]]; then
    rm -f "${temporary}"
  else
    mv "${temporary}" "${env_file}"
  fi
}

atrex_shared_load_local_secrets() {
  local env_file="$1"
  local label="$2"
  atrex_shared_ensure_local_secrets "${env_file}" "${label}"

  # A persisted workspace must always reload the same control-plane identity.
  # Letting inherited shell values override this file makes a restarted
  # Campaign unable to reproduce the capabilities stored in gateway.sqlite.
  # shellcheck disable=SC1090
  source "${env_file}"
}
