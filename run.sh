#!/usr/bin/env bash
# =============================================================================
# run.sh — Build and run HFAQE main.cpp on Linux
# Usage:  chmod +x run.sh && ./run.sh
# =============================================================================
set -euo pipefail

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN="$DIR/hfaqe_main"

echo "=============================================="
echo "  HFAQE — Build & Run"
echo "  Source dir: $DIR"
echo "=============================================="

# ── 1. Detect compiler ────────────────────────────────────────────────────
if command -v g++ &>/dev/null; then
    CXX=g++
elif command -v clang++ &>/dev/null; then
    CXX=clang++
else
    echo "[ERROR] No C++ compiler found. Install with:"
    echo "  sudo apt-get install -y g++"
    exit 1
fi
echo "[build] Compiler : $CXX ($($CXX --version | head -1))"

# ── 2. Detect AVX-512 support ─────────────────────────────────────────────
SIMD_FLAGS=""
if grep -q avx512f /proc/cpuinfo 2>/dev/null; then
    SIMD_FLAGS="-mavx512f -mavx512bw"
    echo "[build] SIMD     : AVX-512 detected — enabling"
else
    echo "[build] SIMD     : AVX-512 not available — scalar fallback"
fi

# ── 3. Compile ────────────────────────────────────────────────────────────
echo "[build] Compiling main.cpp ..."
$CXX \
    -std=c++17 \
    -O3 -march=native -funroll-loops \
    $SIMD_FLAGS \
    -Wall -Wextra \
    -Wno-unused-variable -Wno-unused-parameter \
    -I"$DIR" \
    "$DIR/main.cpp" \
    -o "$BIN" \
    -lm

echo "[build] Binary   : $BIN"
echo "[build] Done."
echo ""

# ── 4. Run ────────────────────────────────────────────────────────────────
echo "=============================================="
echo "  Running hfaqe_main"
echo "=============================================="
"$BIN"
EXIT_CODE=$?

echo ""
echo "=============================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "  RESULT: ALL STEPS PASSED"
else
    echo "  RESULT: $EXIT_CODE STEP(S) FAILED"
fi
echo "=============================================="
exit $EXIT_CODE
