#!/usr/bin/env bash

atrex_default_agate_environment() {
  export AGATE_URL="${AGATE_URL:-http://127.0.0.1:8000}"
  export AGATE_GPU="${AGATE_GPU:-local}"
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
    echo "Agate requires both AGATE_AK and AGATE_SK; unauthenticated localhost may omit both." >&2
    return 64
  fi
}
