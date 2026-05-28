#!/usr/bin/env bash
set -euo pipefail

# Benchmark baseline vs optimized with process isolation.
# Each binary prints compute time (seconds) to stderr.
# Usage:
#   ./bench.sh                         # default multi-size benchmark
#   SIZES="4096 8192" ./bench.sh       # override benchmark sizes
#   ./bench.sh N [density] [seed] [runs]  # compatibility: single size
#
# By default, each nominal size is perturbed by +/- N_JITTER_PCT percent and
# the seed is read from the Linux random source. Set N_JITTER=0 and pass a seed
# explicitly for reproducible local debugging.

if [ $# -gt 0 ] && [ -z "${SIZES:-}" ]; then
  SIZES=$1
  shift
else
  SIZES=${SIZES:-"2000000 4000000 8000000"}
fi

DENSITY=${1:-0.5}
SEED=${2:-}
RUNS=${3:-5}
N_JITTER_PCT=${N_JITTER_PCT:-15}

BINDIR=bin
TMPDIR=tmp

random_u32() {
  local value
  if [ -r /dev/urandom ]; then
    value=$(od -An -N4 -tu4 /dev/urandom | tr -d ' ')
    if [ -n "$value" ]; then
      printf "%s\n" "$value"
      return
    fi
  fi

  value=$((((RANDOM << 16) ^ RANDOM ^ $$ ^ $(date +%s)) & 0xffffffff))
  printf "%s\n" "$value"
}

randomize_size() {
  local nominal=$1 pct=${N_JITTER:-$N_JITTER_PCT}

  if [ "$pct" = "0" ]; then
    printf "%s\n" "$nominal"
    return
  fi

  awk -v n="$nominal" -v pct="$pct" -v r="$(random_u32)" 'BEGIN {
        span = int(n * pct / 100.0);
        if (span < 1) {
            print n;
            exit;
        }
        offset = int(r % (2 * span + 1)) - span;
        randomized = n + offset;
        if (randomized < 1) randomized = 1;
        print randomized;
    }'
}

if [ -z "$SEED" ]; then
  SEED=$(random_u32)
fi

# BASELINE / OPTIMIZED are command prefixes (may contain spaces). Default to
# the bare binaries; if numactl_args.txt is present (produced by SOJ from the
# student's affinity.conf), prepend numactl + those args. To experiment locally,
# either set the env vars yourself or echo args into numactl_args.txt.
BASELINE="${BASELINE:-$BINDIR/nbody_baseline}"
if [ -z "${OPTIMIZED:-}" ]; then
  if [ -s numactl_args.txt ]; then
    OPTIMIZED="numactl $(cat numactl_args.txt) $BINDIR/nbody_optimized"
  else
    OPTIMIZED="$BINDIR/nbody_optimized"
  fi
fi

echo "=== N-body Benchmark ==="
echo "NominalSizes=$SIZES  density=$DENSITY  seed=$SEED  n_jitter_pct=${N_JITTER:-$N_JITTER_PCT}  optimized_runs=$RUNS (+ optimized warmup per size)"
echo ""

make -s all
mkdir -p "$TMPDIR"
SPEEDUPS_FILE="$TMPDIR/.speedups"
rm -f "$SPEEDUPS_FILE"
trap 'rm -f "$SPEEDUPS_FILE"' EXIT

# run_warmup <binary> <N> <density> <seed>
run_warmup() {
  local bin=$1 n=$2 density=$3 seed=$4

  # $bin is intentionally unquoted: it may be a multi-word prefix like
  # "numactl --physcpubind=0-3 ./bin/nbody_optimized". Values are platform-
  # controlled (validated by parse_affinity.sh on the SOJ side).
  $bin "$n" "$density" "$seed" /dev/null 2>/dev/null
}

# run_once <label> <binary> <result_file> <N> <density> <seed>
# Prints one measured run live, writes elapsed time to a temp file.
run_once() {
  local label=$1 bin=$2 result_file=$3 n=$4 density=$5 seed=$6
  local elapsed

  printf "[%s]\n" "$label"
  elapsed=$($bin "$n" "$density" "$seed" /dev/null 2>&1 >/dev/null)
  printf "  Run 1  : %s s\n" "$elapsed"
  printf "  Time   : %s s\n\n" "$elapsed"
  echo "$elapsed" >"$result_file"
}

# run_best <label> <binary> <runs> <result_file> <N> <density> <seed>
# Prints each line live, writes best time to a temp file.
run_best() {
  local label=$1 bin=$2 runs=$3 result_file=$4 n=$5 density=$6 seed=$7
  local best=999999 elapsed

  printf "[%s]\n" "$label"
  run_warmup "$bin" "$n" "$density" "$seed"
  printf "  Warmup : done\n"

  for ((r = 1; r <= runs; r++)); do
    elapsed=$($bin "$n" "$density" "$seed" /dev/null 2>&1 >/dev/null)
    printf "  Run %-2d : %s s\n" "$r" "$elapsed"
    if awk -v elapsed="$elapsed" -v best="$best" 'BEGIN { exit !(elapsed < best) }'; then
      best=$elapsed
    fi
  done
  printf "  Best   : %s s\n\n" "$best"
  echo "$best" >"$result_file"
}

bench_one_size() {
  local n=$1 density=$2 seed=$3 runs=$4
  local best_base_file="$TMPDIR/.best_base_$n"
  local best_opt_file="$TMPDIR/.best_opt_$n"
  local force_base="$TMPDIR/forces_baseline_$n.bin"
  local force_opt="$TMPDIR/forces_optimized_$n.bin"
  local best_base best_opt speedup_raw speedup

  echo "==============================="
  echo "N=$n  density=$density  seed=$seed  baseline_runs=1  optimized_runs=$runs (+ optimized warmup)"
  echo ""

  run_once "1 - Baseline" "$BASELINE" "$best_base_file" "$n" "$density" "$seed"
  run_best "2 - Optimized" "$OPTIMIZED" "$runs" "$best_opt_file" "$n" "$density" "$seed"

  best_base=$(cat "$best_base_file")
  best_opt=$(cat "$best_opt_file")
  rm -f "$best_base_file" "$best_opt_file"

  # --- Verify correctness ---
  echo "[3] Correctness"
  $BASELINE "$n" "$density" "$seed" "$force_base" 2>/dev/null
  $OPTIMIZED "$n" "$density" "$seed" "$force_opt" 2>/dev/null
  "$BINDIR/nbody_verify" "$force_base" "$force_opt"
  echo ""

  # --- Per-size summary ---
  # Keep the full-precision ratio for the final geomean; only round the
  # printed per-size Speedup line.
  speedup_raw=$(awk -v base="$best_base" -v opt="$best_opt" 'BEGIN {
        if (opt <= 0) exit 1;
        printf("%.17g", base / opt);
    }')
  speedup=$(awk -v s="$speedup_raw" 'BEGIN { printf("%.2f", s) }')
  printf "%s\n" "$speedup_raw" >>"$SPEEDUPS_FILE"

  echo "-------------------------------"
  echo "  N         : ${n}"
  echo "  Baseline  : ${best_base} s (single run)"
  echo "  Optimized : ${best_opt} s (best of ${runs})"
  echo "  Speedup   : ${speedup}x"
  echo "-------------------------------"
  echo ""
}

ACTUAL_SIZES=""
for size in $SIZES; do
  actual_size=$(randomize_size "$size")
  ACTUAL_SIZES="${ACTUAL_SIZES}${ACTUAL_SIZES:+ }${actual_size}"
  bench_one_size "$actual_size" "$DENSITY" "$SEED" "$RUNS"
done

# --- Overall summary ---
# GeoSpeedup = exp(avg(log(speedup_i))) over the full-precision per-size ratios.
geo_speedup=$(awk '
    {
        if ($1 <= 0) exit 1;
        sum += log($1);
        count++;
    }
    END {
        if (count == 0) exit 1;
        printf("%.2f", exp(sum / count));
    }
' "$SPEEDUPS_FILE")
rm -f "$SPEEDUPS_FILE"
echo "==============================="
echo "GeoSpeedup : ${geo_speedup}x"
echo "Sizes      : ${ACTUAL_SIZES}"
echo "Nominal    : ${SIZES}"
echo "Seed       : ${SEED}"
echo "==============================="
