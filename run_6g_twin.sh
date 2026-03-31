#!/bin/bash
# =============================================================================
#  6G THz Digital Twin — Launch Script
#  Usage:  bash run_6g_twin.sh [--validate]
# =============================================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TWIN_FILE="$SCRIPT_DIR/6g_thz_digital_twin.py"

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║          6G THz DIGITAL TWIN — LAUNCH SCRIPT                ║"
echo "║  BBU → 300 GHz Radio Unit → THz Channel → UE                ║"
echo "║  High-Speed UDP | Urban Smart-City Scenario                  ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.8+"
    exit 1
fi

# Check dependencies
echo "[*] Checking Python dependencies..."
python3 -c "import matplotlib, numpy" 2>/dev/null || {
    echo "[*] Installing required packages..."
    pip3 install matplotlib numpy
}

# Check for tkinter (needed for TkAgg backend)
python3 -c "import tkinter" 2>/dev/null || {
    echo "[WARN] tkinter not found. Trying to install..."
    sudo apt-get install -y python3-tk 2>/dev/null || {
        echo "[WARN] Could not install tkinter. Switching to Agg backend..."
        sed -i 's/matplotlib.use("TkAgg")/matplotlib.use("Agg")/' "$TWIN_FILE"
    }
}

echo ""
echo "  Controls during simulation:"
echo "    [SPACE]  Pause / Resume"
echo "    [R]      Toggle Rain Attenuation"
echo "    [Q]      Quit"
echo ""
echo "[*] Starting 6G Digital Twin..."
echo ""

python3 "$TWIN_FILE" "$@"
