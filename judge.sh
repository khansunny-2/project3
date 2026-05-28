#!/usr/bin/env bash
set -euo pipefail

# Parse bench.log -> compute score -> emit result.json for SOJ.
# Usage: judge.sh <bench.log> <result.json>
#
# Scoring tiers (linear interpolation between key points):
#   0x        -> 0%
#   1x        -> 20%
#   10x       -> 30%
#   20x       -> 40%
#   30x       -> 50%
#   40x       -> 60%
#   50x       -> 80%
#   85x+      -> 100%

LOG=${1:?bench.log path}
OUT=${2:?result.json path}

if grep -q "^GeoSpeedup[[:space:]]*:" "$LOG"; then
    speedup=$(awk '/^GeoSpeedup[[:space:]]*:/ {v=$3} END {print v}' "$LOG" | tr -d 'x')
    sizes=$(awk '/^Sizes[[:space:]]*:/ {
        sub(/^[^:]*:[[:space:]]*/, "");
        v=$0;
    } END {print v}' "$LOG")
    best_base=""
    best_opt=$(awk '/^  Optimized :/ {v=$3} END {print v}' "$LOG")
    metric="geomean"
else
    if ! grep -q "^  Speedup" "$LOG"; then
        printf '{"success":false,"score":0,"message":"bench.log missing GeoSpeedup or Speedup line","tag":""}' > "$OUT"
        exit 0
    fi

    best_base=$(awk '/^  Baseline  :/ {print $3}' "$LOG")
    best_opt=$(awk  '/^  Optimized :/ {print $3}' "$LOG")
    speedup=$(awk   '/^  Speedup   :/ {print $3}' "$LOG" | tr -d 'x')
    sizes=""
    metric="single"
fi

if [ -z "${speedup:-}" ]; then
    printf '{"success":false,"score":0,"message":"failed to parse bench.log","tag":""}' > "$OUT"
    exit 0
fi

best_opt=${best_opt:-0}

score=$(awk -v s="$speedup" 'BEGIN {
    # Piecewise linear interpolation
    if (s < 1) {
        score = 20 * s
    } else if (s < 10) {
        score = 20 + 10/9 * (s - 1)
    } else if (s < 40) {
        score = 30 + (s - 10)
    } else if (s < 50) {
        score = 60 + 2 * (s - 40)
    } else if (s < 85) {
        score = 80 + 20/35 * (s - 50)
    } else {
        score = 100
    }
    if (score > 100) score = 100
    printf("%.2f", score)
}')

time_ns=$(awk -v t="$best_opt" 'BEGIN { printf("%d", t * 1e9) }')

if [ "$metric" = "geomean" ]; then
    msg="geomean_speedup=${speedup}x sizes=${sizes}"
else
    msg="baseline=${best_base}s optimized=${best_opt}s speedup=${speedup}x"
fi

printf '{"success":true,"score":%s,"message":"%s","time":%s,"memory":0,"tag":"%sx"}\n' \
    "$score" "$msg" "$time_ns" "$speedup" > "$OUT"
