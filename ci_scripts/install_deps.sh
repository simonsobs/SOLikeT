#!/usr/bin/env bash
set -e

# Parse args (simple)
EXTRAS=()

# Default to the pinned lockfile. The weekly Compatibility run passes --latest
# to re-resolve to the newest versions our pyproject constraints allow, which is
# the only way it can actually detect upstream breakage.
SYNC_MODE="--locked"

while [[ $# -gt 0 ]]; do
  case $1 in
    --extras)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        EXTRAS+=("$1")
        shift
      done
      ;;
    --latest)
      SYNC_MODE="--upgrade"
      shift
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

# Compose extras string for uv
EXTRAS_STR=""
for e in "${EXTRAS[@]}"; do
  EXTRAS_STR+="--extra $e "
done

echo "Installing with extras: $EXTRAS_STR (uv sync $SYNC_MODE)"

uv sync $SYNC_MODE $EXTRAS_STR

# Call common installer of likelihoods
bash "$(dirname "$0")/install_likelihoods.sh"
