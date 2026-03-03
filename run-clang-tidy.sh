#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: ./run-clang-tidy.sh [options] [file1.cpp file2.cpp ...] [-- <extra clang-tidy args>]

Run clang-tidy on this project using compile_commands.json.

Options:
  -b, --build-dir DIR      Build directory containing compile_commands.json (default: build-clang)
      --reconfigure        Force CMake reconfiguration before running clang-tidy
      --no-configure       Do not auto-configure if compile_commands.json is missing
      --configure-only     Only (re)configure CMake and exit
      --no-sanitize-cuda-flags
                            Disable automatic removal of CUDA-only flags from compile_commands.json
  -j, --jobs N             Number of parallel clang-tidy jobs (default: number of CPUs)
      --file-regex REGEX   Regex applied to paths from compile_commands (default: ^(src|examples)/)
      --header-filter R    clang-tidy header-filter regex (default: project include/src/examples)
      --checks LIST        Override clang-tidy checks (same format as clang-tidy -checks)
      --fix                Apply fixes when possible (clang-tidy -fix)
      --clang-tidy-bin BIN clang-tidy executable to use (default: clang-tidy)
      --cmake-arg ARG      Extra argument forwarded to CMake configure step (repeatable)
  -h, --help               Show this help

Environment:
  KOKKOS_CLANG_TIDY_BUILD_DIR can override the default build directory.

Examples:
  ./run-clang-tidy.sh
  ./run-clang-tidy.sh -b build -j 8
  ./run-clang-tidy.sh --reconfigure --cmake-arg -DCMAKE_BUILD_TYPE=Debug
  ./run-clang-tidy.sh src/tool_common.cpp -- -extra-arg=-Wno-unknown-warning-option
USAGE
}

die() {
  echo "error: $*" >&2
  exit 1
}

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "required command not found: $1"
}

cleanup_temp_dirs() {
  for dir in "${TEMP_DIRS[@]:-}"; do
    [[ -n "$dir" && -d "$dir" ]] && rm -rf "$dir"
  done
}

cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    getconf _NPROCESSORS_ONLN 2>/dev/null || true
    return
  fi
  echo 4
}

escape_regex() {
  # Escape chars with special meaning in extended regexes.
  sed -e 's/[.[\*^$()+?{}|]/\\&/g' <<<"$1"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$SCRIPT_DIR"
[[ -f "$ROOT_DIR/CMakeLists.txt" ]] || die "CMakeLists.txt not found next to script: $ROOT_DIR"

DEFAULT_BUILD_DIR="${KOKKOS_CLANG_TIDY_BUILD_DIR:-$ROOT_DIR/build-clang}"
BUILD_DIR="$DEFAULT_BUILD_DIR"
COMPILE_DB=""
JOBS="$(cpu_count)"
RECONFIGURE=0
AUTO_CONFIGURE=1
CONFIGURE_ONLY=0
SANITIZE_CUDA_FLAGS=1
FIX=0
CHECKS=""
CLANG_TIDY_BIN="clang-tidy"
FILE_REGEX='^(src|examples)/'
CMAKE_ARGS=()
EXTRA_CLANG_TIDY_ARGS=()
FILE_ARGS=()
TEMP_DIRS=()

trap cleanup_temp_dirs EXIT

ROOT_REGEX="$(escape_regex "$ROOT_DIR")"
HEADER_FILTER="^${ROOT_REGEX}/(include|src|examples)/"

while (($#)); do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    -b|--build-dir)
      [[ $# -ge 2 ]] || die "missing value for $1"
      BUILD_DIR="$2"
      shift 2
      ;;
    --reconfigure)
      RECONFIGURE=1
      shift
      ;;
    --no-configure)
      AUTO_CONFIGURE=0
      shift
      ;;
    --configure-only)
      CONFIGURE_ONLY=1
      shift
      ;;
    --no-sanitize-cuda-flags)
      SANITIZE_CUDA_FLAGS=0
      shift
      ;;
    -j|--jobs)
      [[ $# -ge 2 ]] || die "missing value for $1"
      JOBS="$2"
      shift 2
      ;;
    --file-regex)
      [[ $# -ge 2 ]] || die "missing value for $1"
      FILE_REGEX="$2"
      shift 2
      ;;
    --header-filter)
      [[ $# -ge 2 ]] || die "missing value for $1"
      HEADER_FILTER="$2"
      shift 2
      ;;
    --checks)
      [[ $# -ge 2 ]] || die "missing value for $1"
      CHECKS="$2"
      shift 2
      ;;
    --fix)
      FIX=1
      shift
      ;;
    --clang-tidy-bin)
      [[ $# -ge 2 ]] || die "missing value for $1"
      CLANG_TIDY_BIN="$2"
      shift 2
      ;;
    --cmake-arg)
      [[ $# -ge 2 ]] || die "missing value for $1"
      CMAKE_ARGS+=("$2")
      shift 2
      ;;
    --)
      shift
      EXTRA_CLANG_TIDY_ARGS+=("$@")
      break
      ;;
    -*)
      die "unknown option: $1 (use --help)"
      ;;
    *)
      FILE_ARGS+=("$1")
      shift
      ;;
  esac
done

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs expects a positive integer, got: $JOBS"

require_cmd cmake
require_cmd python3
require_cmd "$CLANG_TIDY_BIN"

if [[ "$BUILD_DIR" != /* ]]; then
  BUILD_DIR="$ROOT_DIR/$BUILD_DIR"
fi
COMPILE_DB="$BUILD_DIR/compile_commands.json"

need_configure=0
if [[ ! -f "$COMPILE_DB" ]]; then
  need_configure=1
fi
if [[ "$RECONFIGURE" -eq 1 ]]; then
  need_configure=1
fi

if [[ "$need_configure" -eq 1 ]]; then
  if [[ "$AUTO_CONFIGURE" -ne 1 ]]; then
    die "missing compile database at $COMPILE_DB (remove --no-configure or run CMake manually)"
  fi
  echo "Configuring CMake in $BUILD_DIR (export compile commands)"
  cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_EXPORT_COMPILE_COMMANDS=ON "${CMAKE_ARGS[@]}"
fi

[[ -f "$COMPILE_DB" ]] || die "compile database not found: $COMPILE_DB"

if [[ "$CONFIGURE_ONLY" -eq 1 ]]; then
  echo "CMake configuration ready: $COMPILE_DB"
  exit 0
fi

cd "$ROOT_DIR"

TIDY_BUILD_DIR="$BUILD_DIR"
if [[ "$SANITIZE_CUDA_FLAGS" -eq 1 ]]; then
  TIDY_BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/clang-tidy-db.XXXXXX")"
  TEMP_DIRS+=("$TIDY_BUILD_DIR")
  SANITIZED_COMPILE_DB="$TIDY_BUILD_DIR/compile_commands.json"
  python3 - "$COMPILE_DB" "$SANITIZED_COMPILE_DB" <<'PY'
import json
import shlex
import sys
from collections import Counter

input_db, output_db = sys.argv[1:3]

drop_exact = {
    "-extended-lambda",
    "--extended-lambda",
    "-expt-extended-lambda",
    "--expt-extended-lambda",
    "-Wext-lambda-captures-this",
}
drop_prefixes = (
    "-arch=sm_",
    "--cuda-gpu-arch=sm_",
    "-gencode=",
    "--generate-code=",
)
drop_pairs = {
    "-arch",
    "--cuda-gpu-arch",
    "-gencode",
    "--generate-code",
}

with open(input_db, "r", encoding="utf-8") as f:
    entries = json.load(f)

removed = Counter()

for entry in entries:
    args = entry.get("arguments")
    if args is None:
        command = entry.get("command")
        if command:
            args = shlex.split(command)
    if not args:
        continue

    kept = [args[0]]
    i = 1
    while i < len(args):
        arg = args[i]
        drop = arg in drop_exact or any(arg.startswith(prefix) for prefix in drop_prefixes)
        if drop:
            removed[arg] += 1
            i += 1
            continue
        if arg in drop_pairs:
            removed[arg] += 1
            i += 2
            continue
        kept.append(arg)
        i += 1

    entry["arguments"] = kept
    entry.pop("command", None)

with open(output_db, "w", encoding="utf-8") as f:
    json.dump(entries, f, indent=2)

removed_total = sum(removed.values())
if removed_total:
    top = ", ".join(f"{flag} x{count}" for flag, count in removed.most_common(4))
    print(f"Sanitized compile database: removed {removed_total} CUDA-only flag(s) ({top})")
else:
    print("Sanitized compile database: no CUDA-only flags removed")
PY
fi

declare -a files
if ((${#FILE_ARGS[@]})); then
  files=("${FILE_ARGS[@]}")
else
  mapfile -t files < <(
    python3 - "$COMPILE_DB" "$ROOT_DIR" "$FILE_REGEX" <<'PY'
import json
import os
import re
import sys

compile_db, root_dir, path_regex = sys.argv[1:4]
try:
    matcher = re.compile(path_regex)
except re.error as exc:
    print(f"invalid --file-regex: {exc}", file=sys.stderr)
    sys.exit(2)

allowed_ext = {".c", ".cc", ".cpp", ".cxx", ".c++"}

with open(compile_db, "r", encoding="utf-8") as f:
    entries = json.load(f)

seen = set()
selected = []

for entry in entries:
    file_path = entry.get("file")
    if not file_path:
        continue

    if not os.path.isabs(file_path):
        directory = entry.get("directory") or root_dir
        file_path = os.path.join(directory, file_path)

    file_path = os.path.realpath(file_path)
    if not file_path.startswith(root_dir + os.sep):
        continue

    rel = os.path.relpath(file_path, root_dir)

    if rel.startswith("external/"):
        continue

    if os.path.splitext(rel)[1].lower() not in allowed_ext:
        continue

    if not matcher.search(rel):
        continue

    if rel in seen:
        continue

    seen.add(rel)
    selected.append(rel)

for rel in selected:
    print(rel)
PY
  )
fi

if ((${#files[@]} == 0)); then
  die "no source files selected (check --file-regex or pass files explicitly)"
fi

echo "Running clang-tidy on ${#files[@]} file(s) with $JOBS job(s)"
echo "Build directory: $BUILD_DIR"
if [[ "$TIDY_BUILD_DIR" != "$BUILD_DIR" ]]; then
  echo "Build directory used by clang-tidy: $TIDY_BUILD_DIR"
fi

declare -a clang_tidy_args
clang_tidy_args=("-p" "$TIDY_BUILD_DIR" "-header-filter=$HEADER_FILTER")
if [[ -n "$CHECKS" ]]; then
  clang_tidy_args+=("-checks=$CHECKS")
fi
if [[ "$FIX" -eq 1 ]]; then
  clang_tidy_args+=("-fix")
fi
clang_tidy_args+=("${EXTRA_CLANG_TIDY_ARGS[@]}")

# xargs returns 123 when one or more clang-tidy invocations fail.
set +e
printf '%s\0' "${files[@]}" | xargs -0 -n 1 -P "$JOBS" -- "$CLANG_TIDY_BIN" "${clang_tidy_args[@]}"
status=$?
set -e

if [[ "$status" -eq 123 ]]; then
  die "clang-tidy reported errors"
fi
if [[ "$status" -ne 0 ]]; then
  exit "$status"
fi

echo "clang-tidy completed successfully"
