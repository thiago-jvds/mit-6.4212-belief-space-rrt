#!/usr/bin/env bash
set -e
apt-get update && apt-get install -y python3-venv python3-full >/dev/null || true

# Install X11 libraries for matplotlib display forwarding (macOS)
apt-get install -y \
    apt-utils \
    x11-apps \
    libx11-6 \
    libx11-dev \
    libxext6 \
    libxrender1 \
    libxtst6 \
    libxi6 \
    xauth \
    >/dev/null || true

python3 -m venv .rm-fp-venv --system-site-packages
.rm-fp-venv/bin/pip install -U pip wheel
if [ -f requirements.txt ]; then .rm-fp-venv/bin/pip install -U -r requirements.txt; fi
if [ -f requirements.user.txt ]; then .rm-fp-venv/bin/pip install -U -r requirements.user.txt; fi

# X11 setup diagnostics
echo ""
echo "=== X11 Display Configuration ==="
echo "DISPLAY: $DISPLAY"
if [ -n "$DISPLAY" ]; then
    echo "✓ DISPLAY is set"
else
    echo "✗ DISPLAY is not set"
fi
echo ""
echo "To enable X11 forwarding on macOS:"
echo "1. Make sure XQuartz is running"
echo "2. On your Mac (not in container), run: xhost +localhost"
echo "3. Restart the devcontainer"
echo "================================"