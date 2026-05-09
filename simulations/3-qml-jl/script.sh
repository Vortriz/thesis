#!/bin/bash
#SBATCH -G 1  #specifies number of GPUs
#SBATCH --time=12:00:00 #specifies maximum duration of run
#SBATCH --job-name=QML #specifies job name
#SBATCH --error=job.%J.err #specifies error file name
#SBATCH --output=job.%J.out #specifies output file name

srun --unbuffered time julia -t auto notebook.jl
