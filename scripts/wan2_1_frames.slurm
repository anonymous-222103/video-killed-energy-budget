#!/bin/bash
#SBATCH --job-name=WAN_frames
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
#SBATCH --qos normal

PROMPTS=(
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
)

PREFIX='exp7_frames_'
RUNS=2
WARMUP=1
MODEL="Wan2.1-T2V-1.3B-Diffusers"
MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"

i=1

# Boucle sur les prompts
for prompt in "${PROMPTS[@]}"; do
    echo "======================================="
    echo "Processing prompt #$i: $prompt"
    echo "======================================="

    # Boucle sur les num_frames de 1 à 100
    for NUM_FRAMES in $(seq 1 100); do
        echo "Generating with num_frames=$NUM_FRAMES"

        NOW=$(date +"%Y-%m-%d_%H-%M-%S")
        OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_frames${NUM_FRAMES}_${NOW}.csv"
        OUT_VIDEO="${PREFIX}${MODEL}_output_prompt${i}_frames${NUM_FRAMES}_${NOW}.mp4"

        python wan2_1.py \
            --model_name "$MODEL_NAME" \
            --prompt "$prompt" \
            --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
            --height 720 \
            --width 1280 \
            --num_frames "$NUM_FRAMES" \
            --guidance_scale 5.0 \
            --runs "$RUNS" \
            --seed 0 \
            --fps 15 \
            --warmup "$WARMUP" \
            --flow_shift 5 \
            --out_csv "$OUT_CSV" \
            --out_video "$OUT_VIDEO" \
            --output_path "/fsx/jdelavande/benchlab/videos/data"
    done

    ((i++))
done
