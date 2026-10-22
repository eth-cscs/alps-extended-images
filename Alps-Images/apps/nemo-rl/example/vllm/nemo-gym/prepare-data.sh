#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == -h || "${1:-}" == --help ]]; then
    echo 'Usage: bash prepare-data.sh {train|validation} OUTPUT_JSONL'
    echo 'Run in the example container. Uses HF_HUB_CACHE and adds Gym agent routing.'
    exit 0
fi
[[ "$#" == 2 ]] || { echo 'Expected a split and output JSONL path.' >&2; exit 2; }

split="$1"
output="$2"
dataset=nvidia/Nemotron-RL-math-OpenMathReasoning
revision=219bf0170cd87cd5db2bbf33715ce9ea43a2c59c
case "$split" in
    train) checksum=cdcf89edb4c9f3047687d0272f70b96cf84bb74dcd7966dad0cd7c721bd91823 ;;
    validation) checksum=5ef14d33dc83a18fac53d6676509434bc1627cb23f520890b712b3c9143118f0 ;;
    *) echo "Unknown dataset split: $split" >&2; exit 2 ;;
esac

printf 'Preparing %s %s.jsonl at revision %s\n' "$dataset" "$split" "$revision" >&2
raw="$(uv run --no-project --python /usr/bin/python3.12 hf download \
    "$dataset" \
    "$split.jsonl" \
    --repo-type=dataset \
    --revision="$revision" \
    --quiet)"
printf '%s  %s\n' "$checksum" "$raw" | sha256sum --check

mkdir -p "$(dirname -- "$output")"
temporary="$(mktemp "$output.tmp.XXXXXX")"
trap 'rm -f -- "$temporary"' EXIT
jq --compact-output \
    '.agent_ref = {name: "math_with_judge_simple_agent"}' \
    "$raw" > "$temporary"
mv -- "$temporary" "$output"
printf 'Prepared %s tasks in %s\n' "$(wc -l < "$output")" "$output" >&2
