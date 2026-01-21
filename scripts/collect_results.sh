#!/bin/bash
# Collect Criterion benchmark results from multiple baselines into markdown format
#
# Usage:
#   ./scripts/collect_results.sh                    # Compare all baselines found
#   ./scripts/collect_results.sh 1a_default 1b_default_native  # Compare specific baselines
#
# Output: Markdown table to stdout (redirect to file if desired)

set -e

CRITERION_DIR="target/criterion"

if [ ! -d "$CRITERION_DIR" ]; then
    echo "Error: $CRITERION_DIR not found. Run benchmarks first." >&2
    exit 1
fi

# Get baselines from args or auto-detect
if [ $# -gt 0 ]; then
    BASELINES=("$@")
else
    # Auto-detect baselines using glob expansion
    BASELINES=()
    for f in "$CRITERION_DIR"/*/*/*/*/estimates.json; do
        [ -f "$f" ] || continue
        baseline=$(basename "$(dirname "$f")")
        # Match pattern: starts with digit, then optional letter, then underscore
        if [[ "$baseline" =~ ^[0-9]+[a-z]?_ ]]; then
            # Check if already in array
            found=0
            for b in "${BASELINES[@]}"; do
                if [ "$b" = "$baseline" ]; then
                    found=1
                    break
                fi
            done
            if [ $found -eq 0 ]; then
                BASELINES+=("$baseline")
            fi
        fi
    done

    # Sort baselines
    IFS=$'\n' BASELINES=($(sort <<<"${BASELINES[*]}")); unset IFS

    if [ ${#BASELINES[@]} -eq 0 ]; then
        echo "Error: No baselines found. Run the benchmark script first." >&2
        exit 1
    fi
fi

echo "# Benchmark Results Comparison"
echo ""
echo "Generated: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "## Baselines"
echo ""
for baseline in "${BASELINES[@]}"; do
    echo "- \`$baseline\`"
done
echo ""

# Build header row
HEADER="| Benchmark | Size |"
SEPARATOR="|:----------|-----:|"
for baseline in "${BASELINES[@]}"; do
    HEADER="$HEADER $baseline |"
    SEPARATOR="$SEPARATOR--------:|"
done

# If we have 2+ baselines, add a diff column
if [ ${#BASELINES[@]} -ge 2 ]; then
    HEADER="$HEADER Δ% |"
    SEPARATOR="$SEPARATOR----:|"
fi

echo "## Results (time in nanoseconds)"
echo ""
echo "$HEADER"
echo "$SEPARATOR"

FIRST_BASELINE="${BASELINES[0]}"

# Use glob to find all benchmark paths with the first baseline
for ESTIMATES_FILE in "$CRITERION_DIR"/*/*/*/"$FIRST_BASELINE"/estimates.json; do
    [ -f "$ESTIMATES_FILE" ] || continue

    # Extract path components
    # e.g., target/criterion/arrays_vs_slices/hamming_bitwise_array/512b/1a_default/estimates.json
    REL_PATH="${ESTIMATES_FILE#$CRITERION_DIR/}"
    BENCH_PATH="${REL_PATH%/$FIRST_BASELINE/estimates.json}"

    SIZE=$(basename "$BENCH_PATH")
    BENCH_NAME=$(dirname "$BENCH_PATH")

    # Skip system directories
    if [[ "$SIZE" == "report" ]] || [[ "$BENCH_NAME" == "." ]]; then
        continue
    fi

    BASE_PATH="$CRITERION_DIR/$BENCH_PATH"

    # Collect times for all baselines
    TIME_VALUES=""
    ROW="| $BENCH_NAME | $SIZE |"

    for baseline in "${BASELINES[@]}"; do
        BASELINE_ESTIMATES="$BASE_PATH/$baseline/estimates.json"
        if [ -f "$BASELINE_ESTIMATES" ]; then
            # Extract mean point_estimate (first point_estimate in file is the mean's)
            MEAN=$(awk -F'"point_estimate":' '{print $2}' "$BASELINE_ESTIMATES" | awk -F',' '{print $1}')
            if [ -n "$MEAN" ]; then
                FORMATTED=$(printf "%.2f" "$MEAN")
                ROW="$ROW $FORMATTED |"
                TIME_VALUES="$TIME_VALUES $MEAN"
            else
                ROW="$ROW - |"
                TIME_VALUES="$TIME_VALUES -"
            fi
        else
            ROW="$ROW - |"
            TIME_VALUES="$TIME_VALUES -"
        fi
    done

    # Calculate diff if we have 2+ baselines
    if [ ${#BASELINES[@]} -ge 2 ]; then
        T1=$(echo "$TIME_VALUES" | awk '{print $1}')
        T2=$(echo "$TIME_VALUES" | awk '{print $2}')

        if [ "$T1" != "-" ] && [ "$T2" != "-" ] && [ -n "$T1" ] && [ -n "$T2" ]; then
            DIFF=$(echo "scale=2; ($T2 - $T1) / $T1 * 100" | bc 2>/dev/null || echo "")
            if [ -n "$DIFF" ]; then
                if [ "${DIFF:0:1}" != "-" ]; then
                    ROW="$ROW +${DIFF}% |"
                else
                    ROW="$ROW ${DIFF}% |"
                fi
            else
                ROW="$ROW - |"
            fi
        else
            ROW="$ROW - |"
        fi
    fi

    echo "$ROW"
done | sort -t'|' -k2,2 -k3,3

echo ""
echo "---"
echo ""
echo "*Δ% shows change from first baseline to second. Negative = faster.*"
