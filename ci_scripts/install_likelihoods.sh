#!/usr/bin/env bash
set -e

# Per-attempt timeout (seconds). Bounds a portal.nersc.gov outage, which
# otherwise hangs each attempt ~18 min (cobaya's downloader has no connect
# timeout) and previously ballooned a CI job to >2 hours.
INSTALL_TIMEOUT=${INSTALL_TIMEOUT:-300}

# Prefer a GNU timeout (coreutils on Linux, gtimeout on macOS); the --version
# probe rejects the unrelated Windows timeout.exe. Falls back to no timeout.
TIMEOUT_CMD=""
for cmd in timeout gtimeout; do
  if command -v "$cmd" >/dev/null 2>&1 && "$cmd" --version >/dev/null 2>&1; then
    TIMEOUT_CMD="$cmd"
    break
  fi
done

install_with_retry() {
  local name="$1"
  echo "Installing ${name} likelihood..."
  for i in 1 2 3; do
    if ${TIMEOUT_CMD:+$TIMEOUT_CMD $INSTALL_TIMEOUT} uv run cobaya-install "$name" --no-set-global; then
      return 0
    fi
    echo "Attempt ${i}/3 for ${name} failed (timeout or error); retrying in 10s..."
    sleep 10
  done
  echo "ERROR: Failed to install ${name} after 3 attempts." >&2
  return 1
}

# Detect Python major/minor
PY_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
echo "Detected Python version: ${PY_MAJOR}.${PY_MINOR}"

# Install MFLike only on Python < 3.13
if [[ $PY_MAJOR -eq 3 && $PY_MINOR -lt 13 ]]; then
  install_with_retry mflike.TTTEEE
else
  echo "Skipping MFLike on Python ${PY_MAJOR}.${PY_MINOR} (requires <3.13)"
fi

# Install Planck likelihood
install_with_retry planck_2018_highl_plik.TTTEEE_lite_native
