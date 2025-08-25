#!/bin/bash -l
#SBATCH --job-name=WAN1.3-720p
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --qos normal

source ./../.venv/bin/activate
readarray -t PROMPTS < <(jq -r '.PROMPTS[]' prompts.json)

PREFIX='exp7_'
RUNS=5
WARMUP=2
MODEL="WAN2.1-T2V-14B-Diffusers"  # options : WAN2.1-T2V-1.3B-Diffusers # or WAN2.1-T2V-14B-Diffusers
MODEL_NAME="Wan-AI/WAN2.1-T2V-14B-Diffusers" # options : Wan-AI/Wan2.1-T2V-1.3B-Diffusers # or Wan-AI/Wan2.1-T2V-14B-Diffusers

i=1
for prompt in "${PROMPTS[@]}"; do
    echo "======================================="
    echo "Processing prompt #$i: $prompt"
    echo "======================================="

    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_${NOW}.csv"
    OUT_VIDEO="${PREFIX}${MODEL}_output_prompt${i}_${NOW}.mp4"

    ./../.venv/bin/python wan2_1.py \
        --model_name "$MODEL_NAME" \
        --prompt "$prompt" \
        --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
        --height 720 \
        --width 1280 \
        --num_frames 81 \
        --guidance_scale 5.0 \
        --runs "$RUNS" \
        --seed 0 \
        --fps 15 \
        --warmup "$WARMUP" \
        --flow_shift 5 \
        --out_csv "$OUT_CSV" \
        --out_video "$OUT_VIDEO" \
        --output_path "./../data"

    ((i++))
done
