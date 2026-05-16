#!/bin/bash

PARTITION="iiser_gpu"
CPUS_PER_TASK=24
GPUS=1

salloc --partition=$PARTITION \
       --cpus-per-task=$CPUS_PER_TASK \
       --gpus=$GPUS \
       --nice \
       srun --pty bash
