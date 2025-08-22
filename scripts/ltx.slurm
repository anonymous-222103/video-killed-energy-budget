#!/bin/bash
#SBATCH --job-name=vidLTX
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
    "The video depicts a winding mountain road covered in snow, with a single vehicle traveling along it. The road is flanked by steep, rocky cliffs and sparse vegetation. The landscape is characterized by rugged terrain and a river visible in the distance. The scene captures the solitude and beauty of a winter drive through a mountainous region."
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

    # Génère des noms uniques
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
        --output_path "/fsx/jdelavande/benchlab/videos/data"

    ((i++))
done
