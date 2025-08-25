#!/bin/bash
#SBATCH --job-name=vidADiff
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

    MODEL="AnimateDiff"
    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_${NOW}.csv"
    OUT_GIF="${PREFIX}${MODEL}_output_prompt${i}_${NOW}.gif"

    python animateDiff.py \
        --adapter_repo "ByteDance/AnimateDiff-Lightning" \
        --adapter_ckpt "animatediff_lightning_4step_diffusers.safetensors" \
        --base_model "emilianJR/epiCRealism" \
        --prompt "$prompt" \
        --guidance_scale 1.0 \
        --steps 4 \
        --runs "$RUNS" \
        --warmup "$WARMUP" \
        --out_csv "$OUT_CSV" \
        --out_gif "$OUT_GIF" \
        --output_path "./../data"

    ((i++))
done
