#!/bin/sh
#SBATCH -N 1
#SBATCH --job-name=VSLNet
#SBATCH --time=5:59:00
#SBATCH --cpus-per-task=6
#SBATCH --mem=90GB
#SBATCH --gres=gpu:1
#SBATCH --chdir=/ibex/ai/home/soldanm/projects/nips22/episodic-memory/NLQ/VSLNet/
#SBATCH -o logs/%x_%A_%a.out
#SBATCH -e logs/%x_%A_%a.err
#SBATCH --array=0

module load gcc

echo "######################### SLURM JOB ########################"
echo HOST NAME
echo `hostname`
echo "############################################################"

environment=ego
conda_root=$HOME/anaconda3

source $conda_root/etc/profile.d/conda.sh
conda activate $environment

# ------------------------ need not change -----------------------------------
LANG=bert
VISUAL=official
VISUALSIZE=2304
MAXPOSLEN=128

python main.py                            \
--task nlq_official_v1                    \
--predictor $LANG                         \
--mode train                              \
--video_feature_dim $VISUALSIZE           \
--max_pos_len $MAXPOSLEN                  \
--epochs 200                              \
--fv $VISUAL                              \
--num_workers 6                           \
--model_dir checkpoints/slowfast_bert_batch32_lr_0001 \
--eval_gt_json "/ibex/ai/home/soldanm/projects/nips22/episodic-memory/NLQ/VSLNet/data/nlq_val.json"        \
--batch_size 32                           \
--init_lr 0.0001  