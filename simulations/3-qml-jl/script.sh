#!/bin/bash
#SBATCH -N 1  #specifies number of nodes
#SBATCH --ntasks-per-node=48 #specifies core per node
#SBATCH --mem=128G
#SBATCH --time=02:00:00 #specifies maximum duration of run
#SBATCH --job-name=QML_sweep #specifies job name
#SBATCH --error=job.%J.err #specifies error file name
#SBATCH --output=job.%J.out #specifies output file name

time julia -p auto sweep.jl
