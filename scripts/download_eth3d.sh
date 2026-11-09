#!/usr/bin/env bash
# Download the ETH3D Courtyard high-res multi-view training archive.
# The archive is ~478 MB and extracts to courtyard/images/dslr_images_undistorted/.
set -euo pipefail

DEST="${1:-data/raw}"
URL="${ETH3D_URL:-https://www.eth3d.net/data/courtyard_dslr_undistorted.7z}"
ARCHIVE="${DEST}/courtyard_dslr_undistorted.7z"

mkdir -p "${DEST}"
if ! command -v curl >/dev/null 2>&1; then
  echo "ERROR: curl is required. Install curl or download manually: ${URL}" >&2
  exit 2
fi
if [ -f "${ARCHIVE}" ]; then
  echo "Archive already exists: ${ARCHIVE}"
else
  echo "Downloading ${URL}"
  curl -L --fail --retry 3 -o "${ARCHIVE}" "${URL}"
fi

echo "Downloaded: ${ARCHIVE}"
echo
echo "Extract with one of:"
echo "  7z x \"${ARCHIVE}\" -o\"${DEST}/eth3d\""
echo "  python -c \"import py7zr; py7zr.SevenZipFile('${ARCHIVE}').extractall('${DEST}/eth3d')\""
echo
echo "Then prepare:"
echo "  python scripts/prepare_courtyard.py --extract"
echo
echo "ETH3D data usage is governed by the ETH3D terms: https://www.eth3d.net/"
