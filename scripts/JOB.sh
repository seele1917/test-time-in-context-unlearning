#!/bin/sh
#$ -l node_q=1
#$ -l h_rt=6:00:00
#$ -j y
#$ -cwd
#$ -p -3

module load cuda/12.3.2 cudnn/9.0.0 code-server/4.22.1
source .venv/bin/activate
echo https://ood.t4.gsic.titech.ac.jp/rnode/`hostname`/8888
PASSWORD=tsubame code-server --bind-addr 0.0.0.0:8888
# qsub -g tgh-24IAT scripts/JOB.sh
