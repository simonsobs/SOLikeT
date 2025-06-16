#!/usr/bin/env bash
set -e

# Parse args (simple)
EXTRAS=()
PYTHON_VERSION=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --extras)
      shift
      while [[ $# -gt 0 && $1 != --* ]]; do
        EXTRAS+=("$1")
        shift
      done
      ;;
    --python)
      PYTHON_VERSION="$2"
      shift 2
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

echo "Installing with extras: $EXTRAS_STR and python $PYTHON_VERSION"

# Install SOLikeT dependencies with uv
if [ -n "$PYTHON_VERSION" ]; then
  uv python install "$PYTHON_VERSION"
  uv python pin "$PYTHON_VERSION"
fi

uv sync --locked $EXTRAS_STR

# Call common installer of likelihoods
bash "$(dirname "$0")/install_likelihoods.sh"
