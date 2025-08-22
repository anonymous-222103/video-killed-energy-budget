#!/bin/bash
#SBATCH --job-name=vidMochi
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --output=/fsx/jdelavande/logs/slurm-%j-%x.out
#SBATCH --mail-user=julien.delavande@huggingface.co
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
    "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k.")

PROMPTS=(
    "A realistic gorilla wearing a yellow Hugging Face t-shirt, filming itself in selfie mode while walking around Paris landmarks like the Eiffel Tower. The gorilla waves at tourists, points at monuments, and playfully scratches its head. Smooth camera motion, lively city background, daylight."
)
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
        --output_path "/fsx/jdelavande/benchlab/videos/data"

    ((i++))
done
