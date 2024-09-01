#!/bin/zsh
qrsh -l node_q=1 -l h_rt=6:00:00 -g tgh-24IAT
module load cuda/12.3.2 cudnn/9.0.0
source .venv/bin/activate
export HF_HOME=/gs/bs/tgh-24IAT
