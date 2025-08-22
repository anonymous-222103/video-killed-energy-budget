#!/bin/bash
#SBATCH --job-name=WAN_tokens
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=32G
#SBATCH --mail-type=ALL
#SBATCH --qos normal

# --- Config de base ---
BASE_PROMPT="A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
PADDING_PHRASE="with intricate details and cinematic lighting"
# Optionnel: forcer le tokenizer si différent
# export TEXT_TOKENIZER="google/t5-v1_1-large"

PREFIX='exp9_tokens_'
RUNS=2
WARMUP=1
MODEL="Wan2.1-T2V-1.3B-Diffusers"
MODEL_NAME="Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
NUM_FRAMES=81
GUIDANCE=5.0
FPS=15

# --- Résolution FIXE (au lieu de balayer) ---
WIDTH=832
HEIGHT=480

# --- Liste des cibles de tokens à tester ---
TOKENS=(16 32 64 96 128 192 256 384 512)

# --- Prompt négatif identique ---
NEG_PROMPT="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

# --- Helper Python pour fabriquer un prompt de longueur EXACTE N tokens ---
gen_prompt_python() {
python - "$BASE_PROMPT" "$1" "$PADDING_PHRASE" <<'PY'
import os, sys
from transformers import AutoTokenizer

base = sys.argv[1]
target = int(sys.argv[2])
pad_phrase = sys.argv[3]

tok_name = os.environ.get("TEXT_TOKENIZER", "google/t5-v1_1-large")
tok = AutoTokenizer.from_pretrained(tok_name)

# Construire un texte qui atteint (ou dépasse) target tokens
s = base.strip()
while True:
    ids = tok(s, add_special_tokens=False).input_ids
    if len(ids) >= target:
        break
    s = (s + " " + pad_phrase).strip()

# Ramener EXACTEMENT à target tokens
# (on recalcule ids pour découper proprement)
ids = tok(s, add_special_tokens=False).input_ids
s_exact = tok.decode(ids[:target], skip_special_tokens=True, clean_up_tokenization_spaces=True)

# Petit garde-fou: s'il est devenu trop court après decode/clean, on ré-étend
while len(tok(s_exact, add_special_tokens=False).input_ids) < target:
    s_exact = (s_exact + " " + pad_phrase).strip()

# Et on recoupe une dernière fois à target exact
ids = tok(s_exact, add_special_tokens=False).input_ids
s_exact = tok.decode(ids[:target], skip_special_tokens=True, clean_up_tokenization_spaces=True)

print(s_exact)
PY
}

i=1
echo "======================================="
echo "Base prompt: $BASE_PROMPT"
echo "Resolution fixed: ${WIDTH}x${HEIGHT}"
echo "======================================="

for ntok in "${TOKENS[@]}"; do
    echo "---- Target tokens: $ntok ----"

    # Génère un prompt exactly-ntok
    PROMPT_EXACT="$(gen_prompt_python "$ntok")"

    NOW=$(date +"%Y-%m-%d_%H-%M-%S")
    OUT_CSV="${PREFIX}${MODEL}_prompt${i}_${ntok}tok_${WIDTH}x${HEIGHT}_${NOW}.csv"
    OUT_VIDEO="${PREFIX}${MODEL}_prompt${i}_${ntok}tok_${WIDTH}x${HEIGHT}_${NOW}.mp4"

    python wan2_1.py \
        --model_name "$MODEL_NAME" \
        --prompt "$PROMPT_EXACT" \
        --negative_prompt "$NEG_PROMPT" \
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

    ((i++))
done
