#!/usr/bin/env bash
set -e

# Detect platform
OS="$(uname | tr '[:upper:]' '[:lower:]')"

echo "Running install_likelihoods.sh on $OS"

# Install MFLike
echo "Installing MFLike likelihood..."
uv run cobaya-install mflike.TTTEEE --no-set-global

# Install Planck likelihood only if not on Windows
if [[ "$OS" != "mingw"* && "$OS" != "cygwin"* && "$OS" != "msys"* ]]; then
  echo "Installing Planck 2018 HighL Plik Lite Native likelihood..."
  uv run cobaya-install planck_2018_highl_plik.TTTEEE_lite_native --no-set-global
else
  echo "Skipping Planck likelihood installation on Windows"
fi