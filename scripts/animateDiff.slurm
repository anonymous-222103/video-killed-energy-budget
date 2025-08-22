#!/bin/bash
#SBATCH --job-name=vidADiff
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
    "A girl smiling")
readarray -t PROMPTS < <(jq -r '.PROMPTS[]' prompts.json)

PREFIX='exp7_'
RUNS=5
WARMUP=2

# Compteur
i=1

# Boucle sur les 10 prompts
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
        --output_path "/fsx/jdelavande/benchlab/videos/data"

    ((i++))
done
