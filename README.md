# Video Killed the Energy Budget: Text-to-Video Energy Benchmark

This repository accompanies the article *"Video Killed the Energy Budget"* and provides all the necessary scripts, data, and instructions to reproduce the experiments described in the paper. The study focuses on the energy consumption and computational costs of state-of-the-art text-to-video (T2V) generation models.

## Introduction

Recent advances in text-to-video (T2V) generation have enabled the creation of high-fidelity, temporally coherent video clips from natural language prompts. However, these systems come with significant computational costs, and their energy demands remain poorly understood. 

In our paper, we present:
- A compute-bound analytical model predicting scaling laws for spatial resolution, temporal length, and denoising steps.
- Experimental validation of these predictions using WAN2.1-T2V, showing quadratic growth with spatial and temporal dimensions, and linear scaling with denoising steps.
- A comparative analysis of six diverse T2V models, evaluating their runtime and energy profiles under default settings.

This repository provides a benchmark reference and practical tools for designing and deploying more sustainable generative video systems.

## Repository Contents

- **`scripts/`**: Python scripts and SLURM job files to run energy measurements for each model.
- **`data/`**: Output CSV files containing energy consumption and performance metrics for each run.
- **`plot.ipynb`**: Jupyter notebook for aggregating and visualizing the results.
- **`results.ipynb`**: Example notebook for loading and analyzing the combined results.
- **`notes.md`**: Additional notes on the experimental setup and reproducibility.

## Experimental Setup

- **Hardware**: All experiments were conducted on a single NVIDIA H100 GPU and an 8-core AMD EPYC 7R13 CPU.
- **Energy Measurement**: Energy consumption for GPU, CPU, and RAM was measured using [CodeCarbon](https://mlco2.github.io/codecarbon/).
- **Protocol**: For each model, 10 prompts were tested. Each prompt was run with 2 warm-up passes and 5 measurement passes. The reported metrics are the mean across these runs.

## Models Benchmarked

The following models were evaluated in this benchmark:
- [ByteDance/ContentV-8B](https://huggingface.co/ByteDance/ContentV-8B)
- [THUDM/CogVideoX-5b](https://huggingface.co/THUDM/CogVideoX-5b)
- [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers)
- [AnimateDiff](https://huggingface.co/ByteDance/AnimateDiff-Lightning)
- [genmo/mochi-1-preview](https://huggingface.co/genmo/mochi-1-preview)
- [Lightricks/LTX-Video-0.9.7-dev](https://huggingface.co/Lightricks/LTX-Video-0.9.7-dev)
- ...and others.

## How to Reproduce the Experiments

1. **Select a Model**: Choose a script from the `scripts/` folder corresponding to the model you want to benchmark.
2. **Run the Benchmark**:
   - Submit the SLURM job using the provided `.slurm` files.
   - Alternatively, run the script directly, adjusting parameters as needed.
3. **Analyze Results**:
   - Results will be saved in the `data/` folder as CSV files.
   - Explore `results.ipynb` for detailed analysis.

For more details on parameters and model usage, refer to each model’s Hugging Face page or the script headers.