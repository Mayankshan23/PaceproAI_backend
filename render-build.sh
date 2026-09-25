#!/usr/bin/env bash
# PacePro AI - Render Deployment Build Script
set -e

echo "=== Installing Python dependencies ==="
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "=== Downloading and configuring OpenGL ES / MediaPipe shared libraries ==="
LIB_DIR="$HOME/.local/lib"
mkdir -p "$LIB_DIR"
mkdir -p /tmp/debs

cd /tmp/debs
# Download required shared libraries for MediaPipe / OpenCV on Debian/Ubuntu non-root environment
apt-get download libgles2 libgl1 libglvnd0 libglx0 libegl1 libglapi-mesa libxext6 libsm6 libxrender1 libgomp1 2>/dev/null || true

for deb in *.deb; do
  if [ -f "$deb" ]; then
    dpkg -x "$deb" /tmp/deb_extracted/ 2>/dev/null || true
  fi
done

if [ -d "/tmp/deb_extracted" ]; then
  find /tmp/deb_extracted -type f -name "*.so*" -exec cp -f {} "$LIB_DIR" \; 2>/dev/null || true
  rm -rf /tmp/deb_extracted /tmp/debs
fi

echo "=== Library setup complete in $LIB_DIR ==="
echo "=== PacePro Backend Build Completed Successfully ==="

