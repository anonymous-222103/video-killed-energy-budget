import torch
from diffusers import MochiPipeline
from diffusers.utils import export_to_video

OUTPUT_PATH = "./../data"
OUT_VIDEO = "gorilla_mochi.mp4"

pipe = MochiPipeline.from_pretrained("genmo/mochi-1-preview")

# Enable memory savings
pipe.enable_model_cpu_offload()
pipe.enable_vae_tiling()

prompt = "Close-up of a chameleon's eye, with its scaly skin changing color. Ultra high resolution 4k."

with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
      frames = pipe(prompt, num_frames=84).frames[0]

export_to_video(frames, f"{OUTPUT_PATH}/{OUT_VIDEO}", fps=30)
