#!/bin/bash
#SBATCH --job-name=vidCogX2B
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
    "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance."
)
readarray -t PROMPTS < <(jq -r '.PROMPTS[]' prompts.json)

PREFIX='exp7_'
RUNS=5
WARMUP=2

MODEL="CogVideoX-2b" # option : CogVideoX-2b # or CogVideoX-5b
MODEL_NAME="THUDM/CogVideoX-2b"  # option : THUDM/CogVideoX-2b # or THUDM/CogVideoX-5b

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
        --output_path "/fsx/jdelavande/benchlab/videos/data"

    ((i++))
done
