#!/bin/bash
#SBATCH --job-name=vidMochi
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --qos normal

readarray -t PROMPTS < <(jq -r '.PROMPTS[]' prompts.json)

PREFIX='exp7_'
RUNS=5
WARMUP=2

i=1

for prompt in "${PROMPTS[@]}"; do
    echo "======================================="
    echo "Processing prompt #$i: $prompt"
    echo "======================================="

    MODEL="Mochi-1-preview"
    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_${NOW}.csv"
    OUT_VIDEO="${PREFIX}${MODEL}_output_prompt${i}_${NOW}.mp4"

    python mochi.py \
        --model_name "genmo/mochi-1-preview" \
        --prompt "$prompt" \
        --num_frames 84 \
        --runs "$RUNS" \
        --warmup "$WARMUP" \
        --fps 30 \
        --out_csv "$OUT_CSV" \
        --out_video "$OUT_VIDEO" \
        --output_path "./../data"

    ((i++))
done
