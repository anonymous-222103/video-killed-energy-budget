#!/bin/bash
#SBATCH --job-name=WAN_res
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --qos normal

PROMPTS=(
    "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
)

PREFIX='exp8_res_'
RUNS=2
WARMUP=1
MODEL="Wan2.1-T2V-1.3B-Diffusers"
MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NUM_FRAMES=81
GUIDANCE=5.0
FPS=15

# Liste de résolutions (au moins 50), toutes multiples de 8
RESOLUTIONS=(
  "256x256" "320x240" "384x384" "448x256" "512x512"
  "576x320" "640x360" "704x384" "768x432" "832x480"
  "896x504" "960x540" "1024x576" "1088x608" "1152x648"
  "1216x684" "1280x720" "1344x756" "1408x792" "1472x828"
  "1536x864" "1600x900" "1664x936" "1728x972" "1792x1008"
  "1856x1044" "1920x1080" "1984x1116" "2048x1152" "2112x1188"
  "2176x1224" "2240x1260" "2304x1296" "2368x1332" "2432x1368"
  "2496x1404" "2560x1440" "2624x1476" "2688x1512" "2752x1548"
  "2816x1584" "2880x1620" "2944x1656" "3008x1692" "3072x1728"
  "3136x1764" "3200x1800" "3264x1836" "3328x1872" "3392x1908"
  "3456x1944" "3520x1980"
)

i=1
for prompt in "${PROMPTS[@]}"; do
    echo "======================================="
    echo "Processing prompt #$i: $prompt"
    echo "======================================="

    for res in "${RESOLUTIONS[@]}"; do
        WIDTH=$(echo $res | cut -d'x' -f1)
        HEIGHT=$(echo $res | cut -d'x' -f2)

        echo "Generating resolution ${WIDTH}x${HEIGHT}"

        NOW=$(date +"%Y-%m-%d_%H-%M-%S")
        OUT_CSV="${PREFIX}${MODEL}_prompt${i}_${WIDTH}x${HEIGHT}_${NOW}.csv"
        OUT_VIDEO="${PREFIX}${MODEL}_prompt${i}_${WIDTH}x${HEIGHT}_${NOW}.mp4"

        python wan2_1.py \
            --model_name "$MODEL_NAME" \
            --prompt "$prompt" \
            --negative_prompt "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards" \
            --height "$HEIGHT" \
            --width "$WIDTH" \
            --num_frames "$NUM_FRAMES" \
            --guidance_scale "$GUIDANCE" \
            --runs "$RUNS" \
            --seed 0 \
            --fps "$FPS" \
            --warmup "$WARMUP" \
            --flow_shift 5 \
            --out_csv "$OUT_CSV" \
            --out_video "$OUT_VIDEO" \
            --output_path "/fsx/jdelavande/benchlab/videos/data"
    done

    ((i++))
done
