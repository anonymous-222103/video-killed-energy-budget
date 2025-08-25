#!/bin/bash
#SBATCH --job-name=vidLTX
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

    MODEL="LTX-Video-0.9.7-dev"
    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_${NOW}.csv"
    OUT_VIDEO="${PREFIX}${MODEL}_output_prompt${i}_${NOW}.mp4"

    python ltx.py \
        --model_name "Lightricks/LTX-Video-0.9.7-dev" \
        --upsample_model_name "Lightricks/ltxv-spatial-upscaler-0.9.7" \
        --prompt "$prompt" \
        --negative_prompt "worst quality, inconsistent motion, blurry, jittery, distorted" \
        --height 512 \
        --width 704 \
        --downscale_factor 0.6667 \
        --num_frames 121 \
        --generate_steps 30 \
        --denoise_steps 10 \
        --denoise_strength 0.4 \
        --decode_timestep 0.05 \
        --image_cond_noise_scale 0.025 \
        --runs "$RUNS" \
        --seed 0 \
        --fps 24 \
        --warmup "$WARMUP" \
        --out_csv "$OUT_CSV" \
        --out_video "$OUT_VIDEO" \
        --output_path "./../data"

    ((i++))
done
