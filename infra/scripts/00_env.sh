#!/usr/bin/env bash
set -euo pipefail
# Load .env (not committed)
if [ -f "${BASH_SOURCE%/*}/../../.env" ]; then
  set -a; source "${BASH_SOURCE%/*}/../../.env"; set +a
else
  echo "Missing .env (copy .env.example)"; exit 1
fi
: "${AWS_PROFILE:?}"; : "${AWS_REGION:?}"; : "${DOMAIN:?}"; : "${SUBDOMAIN:?}"
export REGION="$AWS_REGION"
