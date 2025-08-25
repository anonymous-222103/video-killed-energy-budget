"""
AnimateDiff Benchmark Script
https://huggingface.co/ByteDance/AnimateDiff-Lightning
"""

import argparse
import torch
import time
import pandas as pd
from codecarbon import EmissionsTracker
from diffusers import AnimateDiffPipeline, MotionAdapter, EulerDiscreteScheduler
from diffusers.utils import export_to_gif
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

def main(args):
    print("Starting AnimateDiff benchmark...")
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    dtype = torch.float16
    print(f"Using device: {device}, dtype: {dtype}")

    # Load motion adapter
    adapter = MotionAdapter().to(device, dtype)
    adapter.load_state_dict(load_file(hf_hub_download(args.adapter_repo, args.adapter_ckpt), device=device_str))

    # Load pipeline
    pipe = AnimateDiffPipeline.from_pretrained(args.base_model, motion_adapter=adapter, torch_dtype=dtype).to(device)
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing", beta_schedule="linear")

    results = []

    # Warmup
    print("Warmup run...")
    for _ in range(args.warmup):
        pipe(
            prompt=args.prompt,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps
        )

    # Main generation loop
    print("Starting main runs...")
    tracker = EmissionsTracker(gpu_ids=[0], log_level="warning", tracking_mode="machine", measure_power_secs=1,
                                output_file=f"{args.output_path}/{args.out_csv.replace('.csv', '')}-codecarbon.csv",
                                output_dir=args.output_path)
    torch.cuda.synchronize()
    tracker.start_task("generate")
    start_generate = time.time()
    for i in range(args.runs):
        output = pipe(
            prompt=args.prompt,
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

    if not args.no_save_gif:
        export_to_gif(output.frames[0], f"{args.output_path}/{args.out_gif}")
        print(f"GIF saved to {args.output_path}/{args.out_gif}")

    results.append({
        "adapter_repo": args.adapter_repo,
        "adapter_ckpt": args.adapter_ckpt,
        "base_model": args.base_model,
        "duration_generate": (end_generate - start_generate) / args.runs,
        "energy_generate": emissions_generate.gpu_energy / args.runs,
        "energy_generate_cpu": emissions_generate.cpu_energy / args.runs,
        "energy_generate_ram": emissions_generate.ram_energy / args.runs,
        "prompt": args.prompt,
        "guidance_scale": args.guidance_scale,
        "steps": args.steps,
        "runs": args.runs,
        "out_gif": args.out_gif,
        "out_csv": args.out_csv,
        "warmup": args.warmup,
        "output_path": args.output_path
    })

    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_path}/{args.out_csv}", index=False)
    print(f"Results saved in {args.output_path}/{args.out_csv}")

if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_repo", type=str, default="ByteDance/AnimateDiff-Lightning")
    parser.add_argument("--adapter_ckpt", type=str, default="animatediff_lightning_4step_diffusers.safetensors")
    parser.add_argument("--base_model", type=str, default="emilianJR/epiCRealism")
    parser.add_argument("--prompt", type=str, default="A girl smiling")
    parser.add_argument("--guidance_scale", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--out_csv", type=str, default=f"animatediff_results_{now}.csv")
    parser.add_argument("--out_gif", type=str, default=f"animatediff_gif_{now}.gif")
    parser.add_argument("--output_path", type=str, default="./../data")
    parser.add_argument("--no_save_gif", action="store_true", help="Disable saving GIF")
    parser.add_argument("--warmup", type=int, default=1)
    args = parser.parse_args()
    main(args)
