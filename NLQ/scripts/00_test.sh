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
LANG=EgoVLP
VISUAL=egovlp_egonce
VISUALSIZE=256
MAXPOSLEN=256
DIR=./checkpoints/best_model/egovlp_egonce_batch_32_lr_0.0005_POS_256/

python main.py                           \
--task nlq_official_v1                   \
--predictor EgoVLP                       \
--mode test                              \
--video_feature_dim $VISUALSIZE          \
--max_pos_len $MAXPOSLEN                 \
--fv $VISUAL                             \
--model_dir $DIR                 


LANG=EgoVLP
VISUAL=ht100m
VISUALSIZE=256
MAXPOSLEN=64
DIR=./checkpoints/best_model/ht100m_batch_8_lr_0.0005_POS_64/



python utils/evaluate_ego4d_nlq.py \
    --ground_truth_json data/nlq_val.json \
    --model_prediction_json /home/soldanm/Documents/projects/nips22/episodic-memory/NLQ/VSLNet/checkpoints/best_model/cc3mwebvid_batch_16_lr_0.0005_POS_128/vslnet_nlq_official_v1_cc3mwebvid_128_EgoVLP/model/vslnet_39_28240_preds.json \
    --thresholds 0.3 0.5 \
    --topK 1 5

    