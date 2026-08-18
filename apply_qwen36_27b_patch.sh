#!/bin/bash
#
# Apply the Qwen 3.6 27B support patch to ds4.
#
# This applies qwen36_27b.patch, which is a standard `git diff` produced
# against a clean checkout of the target branch. It adds:
#   - ds4.h:      the ds4_model_config registry (enum, struct, file magic)
#   - ds4.c:      the qwen36_27b_config definition + ds4_get_model_config()
#   - ds4_metal.m: struct ds4_metal_config + configure_metal_for_model()
#
# Usage:  ./apply_qwen36_27b_patch.sh [--check]
#   --check   only verify the patch applies cleanly, do not modify files.
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH_FILE="$SCRIPT_DIR/qwen36_27b.patch"
CHECK_ONLY=0

if [[ "${1:-}" == "--check" ]]; then
    CHECK_ONLY=1
fi

echo "=== Qwen 3.6 27B patch ==="
echo "patch file : $PATCH_FILE"
echo ""

# Sanity: we must be inside the ds4 repo (ds4.h present).
if [[ ! -f "$SCRIPT_DIR/ds4.h" ]]; then
    echo "ERROR: ds4.h not found in $SCRIPT_DIR" >&2
    exit 1
fi
if [[ ! -f "$PATCH_FILE" ]]; then
    echo "ERROR: patch file not found: $PATCH_FILE" >&2
    exit 1
fi

echo "1. Validating patch..."
if git -C "$SCRIPT_DIR" apply --check "$PATCH_FILE"; then
    echo "   patch is valid and applies cleanly."
else
    echo "ERROR: patch does not apply cleanly. Run 'git -C \"$SCRIPT_DIR\" status' and resolve conflicts." >&2
    exit 1
fi
echo ""

if [[ "$CHECK_ONLY" -eq 1 ]]; then
    echo "--check given; no files modified."
    exit 0
fi

echo "2. Applying patch..."
git -C "$SCRIPT_DIR" apply "$PATCH_FILE"
echo "   done."
echo ""

echo "=== Patch applied. ==="
echo "Review : git -C \"$SCRIPT_DIR\" diff --stat"
echo "Commit : git -C \"$SCRIPT_DIR\" commit -am 'Add Qwen 3.6 27B Metal support'"
echo "Push   : git -C \"$SCRIPT_DIR\" push origin feat/qwen36-27b-metal"
