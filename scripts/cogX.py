"""
CogVideoX Benchmark Script
https://huggingface.co/THUDM/CogVideoX-5b
https://huggingface.co/THUDM/CogVideoX-2b
"""

import argparse
import torch
import time
import pandas as pd
from codecarbon import EmissionsTracker
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

def main(args):
    print("Starting CogVideoX benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 
    print(f"Using device: {device}, dtype: {dtype}")

    pipe = CogVideoXPipeline.from_pretrained(
        args.model_name,
        torch_dtype=dtype
    )
    pipe.to(device)
    pipe.enable_model_cpu_offload()
    pipe.vae.enable_tiling()

    results = []

    # Warmup
    print("Warmup run...")
    for _ in range(args.warmup):
        pipe(
            prompt=args.prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )

    # Main generation loop
    print("Starting main runs...")
    tracker = EmissionsTracker(
        gpu_ids=[0],
        log_level="warning",
        tracking_mode="machine",
        measure_power_secs=1,
        output_file=f"{args.output_path}/{args.out_csv.replace('.csv', '')}-codecarbon.csv",
        output_dir=args.output_path
    )
    torch.cuda.synchronize()
    tracker.start_task("generate")
    start_generate = time.time()
    for i in range(args.runs):
        output = pipe(
            prompt=args.prompt,
            num_videos_per_prompt=1,
            num_inference_steps=args.steps,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        )
    torch.cuda.synchronize()
    end_generate = time.time()
    emissions_generate = tracker.stop_task()
    tracker.stop()

    print("Generation completed.")
    print(f"Duration: {end_generate - start_generate:.2f} seconds")
    print(f"GPU Energy: {emissions_generate.gpu_energy*1000:.6f} Wh")
    print(f"CPU Energy: {emissions_generate.cpu_energy*1000:.6f} Wh")
    print(f"RAM Energy: {emissions_generate.ram_energy*1000:.6f} Wh")
    print(f"Total GPU memory used: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Total CPU memory used: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
    print(f"Total RAM memory used: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Total VRAM used: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")

    if not args.no_save_video:
        export_to_video(output.frames[0], f"{args.output_path}/{args.out_video}", fps=args.fps)
        print(f"Video saved to {args.output_path}/{args.out_video}")

    results.append({
        "model_name": args.model_name,
        "duration_generate": (end_generate - start_generate) / args.runs,
        "energy_generate": emissions_generate.gpu_energy / args.runs,
        "energy_generate_cpu": emissions_generate.cpu_energy / args.runs,
        "energy_generate_ram": emissions_generate.ram_energy / args.runs,
        "prompt": args.prompt,
        "num_frames": args.num_frames,
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "runs": args.runs,
        "out_video": args.out_video,
        "out_csv": args.out_csv,
        "fps": args.fps,
        "warmup": args.warmup,
        "output_path": args.output_path
    })

    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_path}/{args.out_csv}", index=False)
    print(f"Results saved in {args.output_path}/{args.out_csv}")

if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="THUDM/CogVideoX-5b")
    parser.add_argument("--prompt", type=str, default="A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical atmosphere of this unique musical performance.")
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=6.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_csv", type=str, default=f"cogvideox_results_{now}.csv")
    parser.add_argument("--out_video", type=str, default=f"cogvideox_video_{now}.mp4")
    parser.add_argument("--output_path", type=str, default="./../data")
    parser.add_argument("--no_save_video", action="store_true", help="Disable saving video")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    main(args)
