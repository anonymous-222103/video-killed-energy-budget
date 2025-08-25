#!/bin/bash
#SBATCH --job-name=vidCogX2B
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --qos normal

readarray -t PROMPTS < <(jq -r '.PROMPTS[]' prompts.json)

PREFIX='exp7_'
RUNS=5
WARMUP=2

MODEL="CogVideoX-2b" # option : CogVideoX-2b # or CogVideoX-5b
MODEL_NAME="THUDM/CogVideoX-2b"  # option : THUDM/CogVideoX-2b # or THUDM/CogVideoX-5b

i=1

for prompt in "${PROMPTS[@]}"; do
    echo "======================================="
    echo "Processing prompt #$i: $prompt"
    echo "======================================="

    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_${NOW}.csv"
    OUT_VIDEO="${PREFIX}${MODEL}_output_prompt${i}_${NOW}.mp4"

    python cogX.py \
        --model_name "$MODEL_NAME" \
        --prompt "$prompt" \
        --num_frames 49 \
        --steps 50 \
        --guidance_scale 6.0 \
        --runs "$RUNS" \
        --seed 42 \
        --fps 8 \
        --warmup "$WARMUP" \
        --out_csv "$OUT_CSV" \
        --out_video "$OUT_VIDEO" \
        --output_path "./../data"

    ((i++))
done
