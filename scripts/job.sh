#!/bin/bash
#SBATCH -G 1                       #specifies number of GPUs
#SBATCH --time=12:00:00            #specifies maximum duration of run
#SBATCH --job-name=QML
#SBATCH --error=job.%J.err
#SBATCH --output=job.%J.out

srun --unbuffered time julia main.jl
