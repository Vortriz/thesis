#!/bin/bash
#SBATCH -N 1  #specifies number of nodes
#SBATCH --ntasks-per-node=24 #specifies core per node
#SBATCH --time=00:15:00 #specifies maximum duration of run
#SBATCH --job-name=julia_precompile #specifies job name
#SBATCH --error=job.err #specifies error file name
#SBATCH --output=job.out #specifies output file name

julia --project=. -e "using Pkg; Pkg.precompile()"
