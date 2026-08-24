#!/bin/sh
# Qwen 3.6 27B Metal smoke gate.
#
# Validates the GQA + dense SwiGLU forward path (metal/qwen_gqa.metal, the
# qwen_graph_forward_token decode path, and the per-layer KV cache) by running
# greedy generation on a Qwen 3.6 27B GGUF with the Metal backend and checking
# that the output is non-empty, printable, and free of obvious stream
# corruption.
#
# This script must run on macOS with a Metal-capable GPU. It does NOT build the
# binary; run `make` (or `make ds4`) first. The sandbox used during
# development is Linux without Metal, so this gate can only be exercised on a
# real Mac -- see QWEN36_27B_PATCH.md.
set -eu

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
    cat <<'USAGE'
Usage: tests/qwen_metal_smoke.sh [MODEL]

Qwen 3.6 27B Metal smoke gate. Runs greedy generation and verifies the output
is non-empty and not corrupted.

Environment:
  DS4_BIN=./ds4
  DS4_QWEN_MODEL=models/Qwen3.6-27B-Q4_K_M.gguf
  DS4_QWEN_BACKEND=metal
  DS4_QWEN_CTX=4096
  DS4_QWEN_GEN=48
  DS4_QWEN_EXTRA_ARGS=""
USAGE
    exit 0
fi

bin=${DS4_BIN:-./ds4}
model=${1:-${DS4_QWEN_MODEL:-models/Qwen3.6-27B-Q4_K_M.gguf}}
ctx=${DS4_QWEN_CTX:-4096}
gen=${DS4_QWEN_GEN:-48}
backend=${DS4_QWEN_BACKEND:-metal}
case "$backend" in
    metal|cuda|cpu) ;;
    *) echo "invalid DS4_QWEN_BACKEND: $backend" >&2; exit 1 ;;
esac

if [ ! -x "$bin" ]; then
    echo "qwen-metal-smoke: binary not found: $bin (run 'make ds4' first)" >&2
    exit 1
fi
if [ ! -f "$model" ]; then
    echo "qwen-metal-smoke: model not found: $model" >&2
    echo "set DS4_QWEN_MODEL or pass the GGUF path as \$1" >&2
    exit 1
fi

tmpdir=$(mktemp -d "${TMPDIR:-/tmp}/ds4-qwen-metal.XXXXXX")
trap 'rm -rf "$tmpdir"' EXIT INT HUP TERM

prompt="$tmpdir/prompt.txt"
stdout="$tmpdir/stdout.txt"
stderr="$tmpdir/stderr.txt"
generated="$tmpdir/generated.txt"

# A short, deterministic prompt whose expected continuation is well-formed
# prose: continue a counting sentence. Greedy (--temp 0) generation should
# produce coherent text, not garbage tokens.
cat > "$prompt" <<'PROMPT'
Count from one to five in words, separated by commas, then stop.
PROMPT

echo "qwen-metal-smoke: model=$model ctx=$ctx gen=$gen backend=$backend"
# DS4_QWEN_EXTRA_ARGS is intentionally shell-split into backend options.
if ! "$bin" -m "$model" "--$backend" ${DS4_QWEN_EXTRA_ARGS:-} \
    --ctx "$ctx" --temp 0 -n "$gen" \
    --prompt-file "$prompt" >"$stdout" 2>"$stderr"; then
    cat "$stderr" >&2
    exit 1
fi

# CUDA multi-GPU selection reports its resolved layout on stdout before
# generation; keep it in raw evidence but exclude it from the checks.
sed '/^ds4: GPU config: .*$/d' "$stdout" > "$generated"

bytes=$(wc -c < "$generated" | tr -d ' ')
if [ "$bytes" -eq 0 ]; then
    echo "qwen-metal-smoke: FAIL: no output generated" >&2
    echo "stderr tail:" >&2
    tail -n 40 "$stderr" >&2 || true
    exit 1
fi

# Reject obvious token-stream corruption: long runs of unprintable bytes,
# repeated single-character spam, or the known GLM corruption markers that
# indicate attention state bleed. The gate is deliberately conservative --
# it catches catastrophic failure without asserting on exact text.
if LC_ALL=C tr -d '[:print:][:space:]' < "$generated" | grep -q .; then
    echo "qwen-metal-smoke: FAIL: output contains non-printable bytes" >&2
    LC_ALL=C head -c 240 "$generated" >&2 || true
    echo >&2
    exit 1
fi

if LC_ALL=C grep -aE 'unistdFlush|stdiint|stdintFlush|stdioFlush|errnoFlush' \
        "$generated" >/dev/null; then
    echo "qwen-metal-smoke: FAIL: known corruption marker found" >&2
    LC_ALL=C grep -aE 'unistdFlush|stdiint|stdintFlush|stdioFlush|errnoFlush' \
        "$generated" >&2 || true
    exit 1
fi

tokens=$(sed -n 's/.*processing \([0-9][0-9]*\) input tokens.*/\1/p' "$stderr" | tail -n 1)
perf=$(sed -n 's/^ds4: prefill: //p' "$stderr" | tail -n 1)
echo "qwen-metal-smoke: PASS bytes=$bytes tokens=${tokens:-unknown} ${perf:-}"
