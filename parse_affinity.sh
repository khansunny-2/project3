#!/bin/sh
# Parse a student-provided affinity.conf and emit a validated numactl
# argument string on stdout. Non-zero exit on any invalid input.
#
# Format (key=value, one per line; '#' comments and blank lines allowed):
#
#   physcpubind = 0-3        (mutually exclusive with cpunodebind)
#   cpunodebind = 0
#   membind     = 0          (mutually exclusive with the other 3)
#   interleave  = 0,1
#   preferred   = 0
#   localalloc  = true
#
# Values must match ^[0-9]+([,-][0-9]+)*$ (digits / commas / hyphens), or
# true/false for localalloc. Anything else → reject. Even if a value
# slips through, the cgroup AllowedCPUs / AllowedMemoryNodes wall still
# bounds what numactl can actually request.

set -eu

FILE=${1:?usage: parse_affinity.sh <affinity.conf>}

if [ ! -f "$FILE" ]; then
    echo "parse_affinity: $FILE not found" >&2
    exit 1
fi

cpu_count=0
mem_count=0
out=""

# Read with a sentinel so the last line without trailing newline still parses.
while IFS='=' read -r raw_k raw_v || [ -n "${raw_k:-}" ]; do
    k=$(printf '%s' "${raw_k:-}" | tr -d '[:space:]')
    v=$(printf '%s' "${raw_v:-}" | tr -d '[:space:]')

    # Skip blanks and comments.
    case "$k" in
        ''|\#*) continue ;;
    esac

    case "$k" in
        physcpubind|cpunodebind)
            echo "$v" | grep -Eq '^[0-9]+([,-][0-9]+)*$' \
                || { echo "parse_affinity: bad value for $k: '$v'" >&2; exit 1; }
            out="$out --$k=$v"
            cpu_count=$((cpu_count + 1))
            ;;
        membind|interleave|preferred)
            echo "$v" | grep -Eq '^[0-9]+([,-][0-9]+)*$' \
                || { echo "parse_affinity: bad value for $k: '$v'" >&2; exit 1; }
            out="$out --$k=$v"
            mem_count=$((mem_count + 1))
            ;;
        localalloc)
            case "$v" in
                true|1)
                    out="$out --localalloc"
                    mem_count=$((mem_count + 1))
                    ;;
                false|0|'')
                    ;;
                *)
                    echo "parse_affinity: bad value for localalloc: '$v'" >&2
                    exit 1
                    ;;
            esac
            ;;
        *)
            echo "parse_affinity: unknown key '$k'" >&2
            exit 1
            ;;
    esac
done < "$FILE"

if [ "$cpu_count" -gt 1 ]; then
    echo "parse_affinity: at most one CPU binding (physcpubind/cpunodebind)" >&2
    exit 1
fi
if [ "$mem_count" -gt 1 ]; then
    echo "parse_affinity: at most one memory binding (membind/interleave/preferred/localalloc)" >&2
    exit 1
fi

# Trim leading space; empty output is a valid "no binding" config.
printf '%s\n' "${out# }"
