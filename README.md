# Benchlab Videos – Text-to-Video Energy Benchmark

This folder contains all scripts, SLURM job files, and results related to the energy benchmarking of state-of-the-art text-to-video open generation models (June 2025).

## Overview

The goal of this benchmark is to evaluate and compare the energy consumption of popular text-to-video models available on the Hugging Face Hub. All experiments were conducted on a controlled hardware setup, and the results are provided in CSV format for reproducibility and further analysis.

## Contents

- **scripts/**: Python scripts and SLURM job files to run energy measurements for each model.
- **data/**: Output CSV files containing energy consumption and performance metrics for each run.
- **plot.ipynb**: Jupyter notebook for aggregating and visualizing the results.
- **results.ipynb**: Example notebook for loading and analyzing the combined results.
- **notes.md**: Additional notes on the experimental setup and reproducibility.

## Experimental Setup

- All models were run on a single NVIDIA H100 GPU and 8-core AMD EPYC 7R13 CPU.
- Each script measures GPU, CPU, and RAM energy consumption using [CodeCarbon](https://mlco2.github.io/codecarbon/).
- For each model, 10 prompts were tested. Each prompt was run with 2 warm-up passes and 5 measurement passes.
- The reported metrics are the mean across these runs.

## Models Benchmarked

- [ByteDance/ContentV-8B](https://huggingface.co/ByteDance/ContentV-8B)
- [THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- [AnimateDiff](https://huggingface.co/ByteDance/AnimateDiff-Lightning)
- [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview)
- [Lightricks/LTX-Video-0.9.7-dev](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev)
- ...and others

## How to Reproduce

1. Choose a script from the `scripts/` folder corresponding to the model you want to benchmark.
2. Submit the SLURM job or run the script directly, adjusting parameters as needed.
3. Results will be saved in the `data/` folder as CSV files.
4. Use `plot.ipynb` to visualize and compare results.

## Citation

If you use these scripts or results in your research, please cite the repository or link to the [Benchlab project](https://github.com/JulienDelavande/benchlab).

---

For more details on parameters and model usage, refer to each model’s Hugging Face page or the script headers.