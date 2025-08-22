#!/bin/bash
#SBATCH --job-name=vidCosmos
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
PREFIX='exp_'

# Compteur
i=1

# Boucle sur les 10 prompts
for prompt in "${PROMPTS[@]}"; do
    echo "======================================="
    echo "Processing prompt #$i: $prompt"
    echo "======================================="

    MODEL="Cosmos-1.0-Diffusion-7B-Text2World"
    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_results_prompt${i}_${NOW}.csv"
    OUT_VIDEO="${PREFIX}${MODEL}_output_prompt${i}_${NOW}.mp4"

    python cosmos.py \
        --model_name "nvidia/Cosmos-1.0-Diffusion-7B-Text2World" \
        --prompt "$prompt" \
        --runs 5 \
        --warmup 2 \
        --fps 30 \
        --out_csv "$OUT_CSV" \
        --out_video "$OUT_VIDEO" \
        --output_path "/fsx/jdelavande/benchlab/videos/data"

    ((i++))
done
