#!/bin/bash
#SBATCH --job-name=WAN1.3-720p
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --qos normal

# Liste de prompts
PROMPTS=(
    "A futuristic cityscape at night, neon lights reflecting on wet streets"
    "A serene beach at sunrise with gentle waves"
    "A dense jungle with exotic animals and mist"
    "A majestic dragon flying over snowy mountains"
    "A cyberpunk hacker in a dark alleyway with holograms"
    "An underwater coral reef with colorful fish"
    "A medieval castle on a cliff during a thunderstorm"
    "A peaceful rural village in spring with blooming flowers"
    "A spaceship approaching a giant alien planet"
    "A mystical forest with glowing mushrooms and fairies"
)

PROMPTS=(
    "A realistic gorilla wearing a yellow Hugging Face t-shirt, filming itself in selfie mode while walking around Paris landmarks like the Eiffel Tower. The gorilla waves at tourists, points at monuments, and playfully scratches its head. Smooth camera motion, lively city background, daylight."
    "Cinematic ad: a shiny Hugging Face robot helps people, shares smiles, and waves in a bright urban park — uplifting music, smooth camera moves, friendly vibe."
)

PROMPTS=(
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
)

readarray -t PROMPTS < <(jq -r '.PROMPTS[]' prompts.json)

PREFIX='exp7_'
RUNS=5
WARMUP=2
MODEL="WAN2.1-T2V-1.3B-Diffusers"  # option : WAN2.1-T2V-1.3B-Diffusers # or WAN2.1-T2V-14B-Diffusers
MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers" # option : Wan-AI/Wan2.1-T2V-1.3B-Diffusers # or Wan-AI/Wan2.1-T2V-14B-Diffusers

# Compteur
i=1

# Boucle sur les 10 prompts
for prompt in "${PROMPTS[@]}"; do
    echo "======================================="
    echo "Processing prompt #$i: $prompt"
    echo "======================================="

    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_${NOW}.csv"
    OUT_VIDEO="${PREFIX}${MODEL}_output_prompt${i}_${NOW}.mp4"

    python wan2_1.py \
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
        --output_path "/fsx/jdelavande/benchlab/videos/data"

    ((i++))
done
