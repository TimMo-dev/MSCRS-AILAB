#!/bin/bash
#SBATCH --job-name=train_pre_redial
#SBATCH --output=train_pre_redial.out
#SBATCH --error=train_pre_redial.err
#SBATCH --mem=24G
#SBATCH --cpus-per-task=15
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00

set -euo pipefail

WORKDIR="/ceph/project/rtm-p10/MSCRS-main"
cd "$WORKDIR"

srun bash -lc '
  echo "HOST: $(hostname)"
  echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
  echo "SLURM_STEP_GPUS=$SLURM_STEP_GPUS"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  nvidia-smi
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
'

srun singularity exec --nv --cleanenv \
  --bind "$(dirname "$WORKDIR"):/home/weiyibiao" \
  --env CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  --env TRANSFORMERS_OFFLINE=1 \
  pytorch38_cu111_pyg.sif \
  python /home/weiyibiao/MSCRS-main/rec/src/train_pre_redial.py \
    --dataset /home/weiyibiao/MSCRS-main/data/rec_data/redial \
    --tokenizer /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
    --model     /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
    --text_tokenizer /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
    --text_encoder   /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 6

