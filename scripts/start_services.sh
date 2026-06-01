#!/bin/bash

LOG_DIR_TB="data/"

echo "=== Starting Tmux Services on $(hostname) ==="

# 1. Kill old sessions if they exist to prevent port conflicts
tmux kill-session -t pluto 2>/dev/null
tmux kill-session -t tensorboard 2>/dev/null

# 2. Launch Pluto in a detached tmux session
echo "[+] Launching Pluto..."
tmux new-session -d -s pluto "julia -t $SLURM_JOB_CPUS_PER_NODE scripts/pluto.jl 2>&1"

# 3. Launch TensorBoard in a detached tmux session
echo "[+] Launching TensorBoard..."
tmux new-session -d -s tensorboard "uv tool run tensorboard --logdir $LOG_DIR_TB --host 0.0.0.0 --port 6006 2>&1"

echo "==================================================="
echo "[!] Both services are running in the background."
echo "[!] To view Pluto live:  tmux attach -t pluto"
echo "[!] To view TB live:     tmux attach -t tensorboard"
echo "==================================================="

# 4. Attach to the Pluto session by default
tmux attach -t pluto
