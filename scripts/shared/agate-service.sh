#!/usr/bin/env bash

atrex_default_agate_environment() {
  export AGATE_URL="${AGATE_URL:-https://atrex-gateway.alibaba-inc.com}"
  export AGATE_GPU="${AGATE_GPU:-L20N}"
}

atrex_require_agate_environment() {
  atrex_default_agate_environment
  case "${AGATE_URL}" in
    http://127.0.0.1|https://127.0.0.1|http://localhost|https://localhost|http://\[::1\]|https://\[::1\]|\
    http://127.0.0.1/*|https://127.0.0.1/*|http://localhost/*|https://localhost/*|http://\[::1\]/*|https://\[::1\]/*|\
    http://127.0.0.1:*|https://127.0.0.1:*|http://localhost:*|https://localhost:*|http://\[::1\]:*|https://\[::1\]:*)
      if [[ -z "${AGATE_AK:-}" && -z "${AGATE_SK:-}" ]]; then
        return 0
      fi
      ;;
  esac
  if [[ -z "${AGATE_AK:-}" || -z "${AGATE_SK:-}" ]]; then
    echo "Remote Agate requires both AGATE_AK and AGATE_SK." >&2
    return 64
  fi
}
