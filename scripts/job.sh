#!/bin/bash
#SBATCH --partition=iiser_gpu
#SBATCH --cpus-per-task=16
#SBATCH --gpus 1
#SBATCH --time=1-12:00:00            #specifies maximum duration of run
#SBATCH --job-name=GQML
#SBATCH --error=logs/job.%J.err
#SBATCH --output=logs/job.%J.out

julia --project -e 'using Pkg; Pkg.precompile()'
julia --project notebooks/main.jl
