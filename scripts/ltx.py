""""
Benchmark script for Lightricks LTX-Video model using Diffusers library.
https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev
"""
import argparse
import torch
import time
import pandas as pd
from codecarbon import EmissionsTracker
from diffusers import LTXConditionPipeline, LTXLatentUpsamplePipeline
from diffusers.utils import export_to_video

def round_to_nearest_resolution_acceptable_by_vae(height, width, ratio):
    height = height - (height % ratio)
    width = width - (width % ratio)
    return height, width

def main(args):
    print("Starting LTX video benchmark...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the LTX pipeline and upsample pipeline
    print(f"Loading model {args.model_name} and upsample model {args.upsample_model_name}...")
    pipe = LTXConditionPipeline.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)
    pipe_upsample = LTXLatentUpsamplePipeline.from_pretrained(args.upsample_model_name, vae=pipe.vae, torch_dtype=torch.bfloat16)
    pipe.to(device)
    pipe_upsample.to(device)
    pipe.vae.enable_tiling()

    expected_height, expected_width = args.height, args.width
    downscale_factor = args.downscale_factor
    num_frames = args.num_frames

    downscaled_height = int(expected_height * downscale_factor)
    downscaled_width = int(expected_width * downscale_factor)
    downscaled_height, downscaled_width = round_to_nearest_resolution_acceptable_by_vae(
        downscaled_height, downscaled_width, pipe.vae_spatial_compression_ratio)

    results = []

    #### STEP 1: Low resolution generation ####
    print(f"Running low resolution generation with {downscaled_height}x{downscaled_width}...")
    for _ in range(args.warmup):
        pipe(
            conditions=None,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=args.generate_steps,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="latent",
        )

    tracker = EmissionsTracker(gpu_ids=[0], log_level="warning", tracking_mode="machine", measure_power_secs=1, output_file=f"{args.output_path}/{args.out_csv.replace('.csv', '')}-codecarbon.csv", output_dir=args.output_path)
    torch.cuda.synchronize()
    tracker.start_task("generate")
    start_generate = time.time()

    for _ in range(args.runs):
        latents = pipe(
            conditions=None,
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=downscaled_width,
            height=downscaled_height,
            num_frames=num_frames,
            num_inference_steps=args.generate_steps,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="latent",
        ).frames

    torch.cuda.synchronize()
    end_generate = time.time()
    emissions_generate = tracker.stop_task()

    #### STEP 2: Upscaling latent ####
    print(f"Running upscaling with {downscaled_height}x{downscaled_width} to {expected_height}x{expected_width}...")
    for _ in range(args.warmup):
        _ = pipe_upsample(latents=latents, output_type="latent").frames

    torch.cuda.synchronize()
    tracker.start_task("upsample")
    start_upsample = time.time()
    for _ in range(args.runs):
        upscaled_latents = pipe_upsample(latents=latents, output_type="latent").frames

    torch.cuda.synchronize()
    end_upsample = time.time()
    emissions_upsample = tracker.stop_task()

    #### STEP 3: Denoising and final generation ####
    print(f"Running denoising and final generation with {expected_height}x{expected_width}...")
    upscaled_height = downscaled_height * 2
    upscaled_width = downscaled_width * 2
    for _ in range(args.warmup):
        _ = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=args.denoise_strength,
            num_inference_steps=args.denoise_steps,
            latents=upscaled_latents,
            decode_timestep=args.decode_timestep,
            image_cond_noise_scale=args.image_cond_noise_scale,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="pil",
        ).frames[0]

    torch.cuda.synchronize()
    tracker.start_task("denoise")
    start_denoise = time.time()
    for _ in range(args.runs):
        video = pipe(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            width=upscaled_width,
            height=upscaled_height,
            num_frames=num_frames,
            denoise_strength=args.denoise_strength,
            num_inference_steps=args.denoise_steps,
            latents=upscaled_latents,
            decode_timestep=args.decode_timestep,
            image_cond_noise_scale=args.image_cond_noise_scale,
            generator=torch.Generator().manual_seed(args.seed),
            output_type="pil",
        ).frames[0]
    torch.cuda.synchronize()
    end_denoise = time.time()
    emissions_denoise = tracker.stop_task()
    tracker.stop()

    print("Generation complete!")
    print(f"Generated {num_frames} frames at {expected_height}x{expected_width} resolution.")
    print(f"Total time for generation: {(end_generate - start_generate) / args.runs:.6f} seconds")
    print(f"Total time for upscaling: {(end_upsample - start_upsample) / args.runs:.6f} seconds")
    print(f"Total time for denoising: {(end_denoise - start_denoise) / args.runs:.6f} seconds")
    print(f"Total GPU energy for generation: {emissions_generate.gpu_energy / args.runs:.6f} kWh")
    print(f"Total GPU energy for upscaling: {emissions_upsample.gpu_energy / args.runs:.6f} kWh")
    print(f"Total GPU energy for denoising: {emissions_denoise.gpu_energy / args.runs:.6f} kWh")
    print(f"Total CPU energy for generation: {emissions_generate.cpu_energy / args.runs:.6f} kWh")
    print(f"Total CPU energy for upscaling: {emissions_upsample.cpu_energy / args.runs:.6f} kWh")
    print(f"Total RAM energy for generation: {emissions_generate.ram_energy / args.runs:.6f} kWh")
    print(f"Total RAM energy for upscaling: {emissions_upsample.ram_energy / args.runs:.6f} kWh")
    print(f"Total RAM energy for denoising: {emissions_denoise.ram_energy / args.runs:.6f} kWh")
    print(f"Total GPU memory allocated: {torch.cuda.max_memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Total GPU memory reserved: {torch.cuda.max_memory_reserved() / (1024 ** 3):.2f} GB")
    print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / (1024 ** 3):.2f} GB")
    print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / (1024 ** 3):.2f} GB")


    # Resize final
    video = [frame.resize((expected_width, expected_height)) for frame in video]

    if not args.no_save_video:
        export_to_video(video, f"{args.output_path}/{args.out_video}", fps=args.fps)
        print(f"Video saved to {args.output_path}/{args.out_video}")

    results.append({
        "model_name": args.model_name,
        "duration_generate": (end_generate - start_generate)/ args.runs,
        "duration_upsample": (end_upsample - start_upsample) / args.runs,
        "duration_denoise": (end_denoise - start_denoise) / args.runs,
        "energy_generate_gpu": emissions_generate.gpu_energy / args.runs,
        "energy_upsample_gpu": emissions_upsample.gpu_energy / args.runs,
        "energy_denoise_gpu": emissions_denoise.gpu_energy / args.runs,
        "energy_generate_cpu": emissions_generate.cpu_energy / args.runs,
        "energy_upsample_cpu": emissions_upsample.cpu_energy / args.runs,
        "energy_denoise_cpu": emissions_denoise.cpu_energy / args.runs,
        "energy_generate_ram": emissions_generate.ram_energy / args.runs,
        "energy_upsample_ram": emissions_upsample.ram_energy / args.runs,
        "energy_denoise_ram": emissions_denoise.ram_energy / args.runs,
        "upsample_model_name": args.upsample_model_name,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "height": expected_height,
        "width": expected_width,
        "downscaled_height": downscaled_height,
        "downscaled_width": downscaled_width,
        "num_frames": num_frames,
        "generate_steps": args.generate_steps,
        "denoise_steps": args.denoise_steps,
        "denoise_strength": args.denoise_strength,
        "decode_timestep": args.decode_timestep,
        "image_cond_noise_scale": args.image_cond_noise_scale,
        "runs": args.runs,
        "out_video": args.out_video,
        "out_csv": args.out_csv,
        "fps": args.fps,
        "warmup": args.warmup,
        "output_path": args.output_path
    })

    df = pd.DataFrame(results)
    df.to_csv(f"{args.output_path}/{args.out_csv}", index=False)
    print(f"Results saved to {args.output_path}/{args.out_csv}")

if __name__ == "__main__":
    now = time.strftime("%Y-%m-%d_%H-%M-%S")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Lightricks/LTX-Video-0.9.7-dev")
    parser.add_argument("--upsample_model_name", type=str, default="Lightricks/ltxv-spatial-upscaler-0.9.7")
    parser.add_argument("--prompt", type=str, default="A beautiful landscape with mountains and a river, high quality, detailed, cinematic")
    parser.add_argument("--negative_prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=704)
    parser.add_argument("--downscale_factor", type=float, default=2/3)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--generate_steps", type=int, default=30)
    parser.add_argument("--denoise_steps", type=int, default=10)
    parser.add_argument("--denoise_strength", type=float, default=0.4)
    parser.add_argument("--decode_timestep", type=float, default=0.05)
    parser.add_argument("--image_cond_noise_scale", type=float, default=0.025)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_csv", type=str   , default=f"results_Lightricks-LTX-Video_{now}.csv")
    parser.add_argument("--out_video", type=str, default=f"output_Lightricks-LTX-Video_{now}.mp4")
    parser.add_argument("--output_path", type=str, default="./../data")
    parser.add_argument("--no_save_video", action="store_true", help="Disable saving video")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs to skip before measuring performance")

    args = parser.parse_args()
    main(args)
