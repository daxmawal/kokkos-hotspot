#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "error: $*" >&2
  exit 1
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ANALYZE_SCRIPT="${KOKKOS_TIMING_ANALYZE_SCRIPT:-"$ROOT_DIR/scripts/analyze_timings.py"}"
DEFAULT_CSV="$ROOT_DIR/build/kokkos_kernel_times.csv"
CSV_PATH="${1:-${KOKKOS_TIMING_OUT:-$DEFAULT_CSV}}"
APP_BIN="${KOKKOS_APP:-"$ROOT_DIR/build/kokkos_app"}"
TIMING_TOOL_LIB="${KOKKOS_TIMING_TOOL_LIB:-"$ROOT_DIR/build/libkokkos_profiling_tool.so"}"
NO_RUN="${KOKKOS_HOTSPOTS_NO_RUN:-0}"
TOP_N="${KOKKOS_HOTSPOTS_TOP:-}"

[[ -n "$CSV_PATH" ]] || die "missing timing CSV path (arg or KOKKOS_TIMING_OUT)"
[[ -f "$ANALYZE_SCRIPT" ]] || die "analyze script not found: $ANALYZE_SCRIPT"
command -v python3 >/dev/null 2>&1 || die "python3 not found"

if [[ "$NO_RUN" == "0" ]]; then
  [[ -x "$APP_BIN" ]] || die "missing app binary: $APP_BIN"
  [[ -f "$TIMING_TOOL_LIB" ]] || die "missing profiling tool library: $TIMING_TOOL_LIB"
  mkdir -p "$(dirname "$CSV_PATH")"
  echo "Running timing capture: $APP_BIN"
  env KOKKOS_TOOLS_LIBS="$TIMING_TOOL_LIB" \
      KOKKOS_TIMING_OUT="$CSV_PATH" \
      "$APP_BIN"
fi

[[ -f "$CSV_PATH" ]] || die "timing CSV not found: $CSV_PATH"

ARGS=("$CSV_PATH")
if [[ -n "$TOP_N" ]]; then
  ARGS+=(--top "$TOP_N")
fi

python3 "$ANALYZE_SCRIPT" "${ARGS[@]}"
