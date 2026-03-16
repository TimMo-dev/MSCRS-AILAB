#!/bin/bash
#SBATCH --job-name=mscrs_pretrain
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G


WORKDIR="/ceph/project/rtm-p10/MSCRS-main"
cd "$WORKDIR"

# Automatically find the next available pretrain_run directory
base_run_dir="train_runs/pretrain_run"
mkdir -p "$base_run_dir"

# Find next run number
current_iteration=1
while [ -d "$base_run_dir/pretrain_inspired${current_iteration}" ]; do
  current_iteration=$((current_iteration + 1))
done

pretrain_run_dir="$base_run_dir/pretrain_inspired${current_iteration}"
mkdir -p "$pretrain_run_dir"

outfile="${pretrain_run_dir}/${SLURM_JOB_ID}_pretrain_training.out"
errfile="${pretrain_run_dir}/${SLURM_JOB_ID}_pretrain_training.err"


# Launch training
srun --output="${outfile}" --error="${errfile}" \
  singularity exec --nv --cleanenv \
    --bind "$(dirname "$WORKDIR"):/home/weiyibiao" \
    pytorch38_cu111_pyg.sif \
    python /home/weiyibiao/MSCRS-main/rec/src/train_pre_inspired.py \
      --dataset /home/weiyibiao/MSCRS-main/data/rec_data/inspired \
      --tokenizer /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --model     /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --text_tokenizer /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --text_encoder   /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --output_dir "${pretrain_run_dir}" \
      --num_train_epochs 10 \
      --fp16 \
      --per_device_train_batch_size 2 \
      --per_device_eval_batch_size 2 \
      --gradient_accumulation_steps 16


