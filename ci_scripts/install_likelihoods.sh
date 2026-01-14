#!/usr/bin/env bash
set -e

# Detect Python major/minor
PY_MAJOR=$(python -c 'import sys; print(sys.version_info.major)')
PY_MINOR=$(python -c 'import sys; print(sys.version_info.minor)')
echo "Detected Python version: ${PY_MAJOR}.${PY_MINOR}"

# Install MFLike only on Python < 3.13
if [[ $PY_MAJOR -eq 3 && $PY_MINOR -lt 13 ]]; then
  echo "Installing MFLike likelihood..."
  for i in 1 2 3; do
    uv run cobaya-install mflike.TTTEEE --no-set-global && break || sleep 10
  done
else
  echo "Skipping MFLike on Python ${PY_MAJOR}.${PY_MINOR} (requires <3.13)"
fi

# Install Planck likelihood
echo "Installing Planck 2018 HighL Plik Lite Native likelihood..."
for i in 1 2 3; do
  uv run cobaya-install planck_2018_highl_plik.TTTEEE_lite_native --no-set-global && break || sleep 10
done
