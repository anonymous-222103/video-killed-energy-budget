"""
Benchmark script for WAN2.1 T2V model using Diffusers library.
https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers
https://huggingface.co/Wan-AI/Wan2.1-T2V-14B-Diffusers
"""
import argparse
import torch
import time
import pandas as pd
from codecarbon import EmissionsTracker
from diffusers import AutoencoderKLWan, WanPipeline
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers.utils import export_to_video

def main(args):
    print("Starting WAN2.1 T2V benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 
    print(f"Using device: {device}, dtype: {dtype}")

    vae = AutoencoderKLWan.from_pretrained(args.model_name, subfolder="vae", torch_dtype=torch.float32)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True, num_train_timesteps=1000, flow_shift=args.flow_shift)
    pipe = WanPipeline.from_pretrained(args.model_name, vae=vae, torch_dtype=dtype)
    pipe.scheduler = scheduler
    pipe.to(device)

    results = []
    
    # Warmup
    print("Warmup run...")
    for _ in range(args.warmup):
        pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps
        )

    # Main generation loop
    print("Starting main runs...")
    tracker = EmissionsTracker(gpu_ids=[0], log_level="warning", tracking_mode="machine", measure_power_secs=1, output_file=f"{args.out_csv.replace('.csv', '')}-codecarbon.csv", output_dir=args.output_path)
    torch.cuda.synchronize()
    tracker.start_task("generate")
    start_generate = time.time()
    for i in range(args.runs):
        output = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=args.width,
            height=args.height,
            num_frames=args.num_frames,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps
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
        "negative_prompt": args.negative_prompt,
        "height": args.height,
        "width": args.width,
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
    print(f"results saved in {args.output_path}/{args.out_csv}")

if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Wan-AI/Wan2.1-T2V-1.3B-Diffusers") # Wan-AI/Wan2.1-T2V-1.3B-Diffusers # Wan-AI/Wan2.1-T2V-14B-Diffusers
    parser.add_argument("--prompt", type=str, default="A cat walks on the grass, realistic")
    parser.add_argument("--negative_prompt", type=str, default="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards")
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--width", type=int, default=832)
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str, default=f"wan2_results_{now}.csv")
    parser.add_argument("--out_video", type=str, default=f"wan2_video_{now}.mp4")
    parser.add_argument("--output_path", type=str, default="./../data")
    parser.add_argument("--no_save_video", action="store_true", help="Disable saving video")
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--flow_shift", type=int, default=5., help="5.0 for 720P, 3.0 for 480P")
    args = parser.parse_args()
    main(args)
    