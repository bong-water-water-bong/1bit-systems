#!/usr/bin/env bash
# sherry-bench.sh — tok/s probe for Sherry 1.25-bit vs TQ1 baseline.
#
# Runs the gen-1 reference harness (`bitnet_decode`) against a Sherry-packed
# halo-1bit-2b model at 64-token and 1024-token contexts. Emits JSON to
# `/home/bcloud/claude output/sherry-bench.json` (the blessed benchmark
# folder per project memory `feedback_benchmark_output_folder.md`).
#
# TQ1 baseline (from `project_bitnet_live_bench.md`):
#   - 64-token: ~67 tok/s
#   - 1024-token: ~33 tok/s
#
# Sherry target: meet or beat both numbers — kernel bandwidth drops from
# 2 bpw to 1.25 bpw so upper-bound speedup is 1.6x.
#
# Usage:
#   benchmarks/sherry-bench.sh [--model PATH] [--iters N]
#
# The Sherry model defaults to /opt/halo-ai/models/halo-1bit-2b-sherry.h1b
# and can be overridden by $HALO_SHERRY_MODEL or the --model flag.
#
# This script only fails hard if `bitnet_decode` itself exits non-zero OR
# the model path is missing — the JSON is still written in both cases so
# CI can grep for the failure reason.

set -euo pipefail

OUT_DIR="/home/bcloud/claude output"
OUT_JSON="${OUT_DIR}/sherry-bench.json"
DEFAULT_MODEL="${HALO_SHERRY_MODEL:-/opt/halo-ai/models/halo-1bit-2b-sherry.h1b}"
DEFAULT_TQ1="${HALO_TQ1_MODEL:-/opt/halo-ai/models/halo-1bit-2b.h1b}"
DEFAULT_ITERS="${HALO_BENCH_ITERS:-64}"
BITNET_DECODE_BIN="${BITNET_DECODE:-/opt/halo-ai/bin/bitnet_decode}"

MODEL="${DEFAULT_MODEL}"
TQ1_MODEL="${DEFAULT_TQ1}"
ITERS="${DEFAULT_ITERS}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --tq1)
            TQ1_MODEL="$2"
            shift 2
            ;;
        --iters)
            ITERS="$2"
            shift 2
            ;;
        -h|--help)
            grep '^#' "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            echo "sherry-bench: unknown arg: $1" >&2
            exit 2
            ;;
    esac
done

mkdir -p "${OUT_DIR}"

run_ctx() {
    # run_ctx <model> <ctx>
    local model="$1"
    local ctx="$2"
    if [[ ! -f "${model}" ]]; then
        echo "MISSING"
        return
    fi
    if [[ ! -x "${BITNET_DECODE_BIN}" ]]; then
        echo "NO_BIN"
        return
    fi
    # bitnet_decode prints "X.XX tok/s" somewhere in its trailing summary;
    # this script doesn't mandate a parser — CI wraps and greps. We just
    # re-emit the raw last line for the JSON.
    "${BITNET_DECODE_BIN}" \
        --model "${model}" \
        --ctx "${ctx}" \
        --iters "${ITERS}" \
        2>&1 | tail -1 | tr -d '"\n'
}

TIMESTAMP="$(date -Is)"
SHERRY_64="$(run_ctx "${MODEL}" 64)"
SHERRY_1024="$(run_ctx "${MODEL}" 1024)"
TQ1_64="$(run_ctx "${TQ1_MODEL}" 64)"
TQ1_1024="$(run_ctx "${TQ1_MODEL}" 1024)"

cat > "${OUT_JSON}" <<JSON
{
  "harness": "bitnet_decode",
  "timestamp": "${TIMESTAMP}",
  "iters": ${ITERS},
  "baseline_tq1_tok_s_64":   "${TQ1_64}",
  "baseline_tq1_tok_s_1024": "${TQ1_1024}",
  "sherry_tok_s_64":         "${SHERRY_64}",
  "sherry_tok_s_1024":       "${SHERRY_1024}",
  "notes": "Raw last-line output from bitnet_decode. Baseline per project_bitnet_live_bench.md: 67 @64, 33 @1024."
}
JSON

echo "sherry-bench: wrote ${OUT_JSON}"
