#!/bin/bash
#SBATCH --job-name=mscrs_rec_train
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@600      # 10 minutes before walltime (AAU example uses SIGTERM@30) :contentReference[oaicite:2]{index=2}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G

WORKDIR="/ceph/project/rtm-p10/MSCRS-main"
cd "$WORKDIR"

base_run_dir="train_runs/inspired_run"
mkdir -p "$base_run_dir"

current_iteration=1
while [ -d "$base_run_dir/inspired_rec${current_iteration}" ]; do
  current_iteration=$((current_iteration + 1))
done

inspired_rec_run_dir="$base_run_dir/inspired_rec${current_iteration}"
mkdir -p "$inspired_rec_run_dir"

outfile="${inspired_rec_run_dir}/${SLURM_JOB_ID}_inspired_rec_training.out"
errfile="${inspired_rec_run_dir}/${SLURM_JOB_ID}_inspired_rec_training.err"

# Launch training
srun --output="${outfile}" --error="${errfile}" \
  singularity exec --nv --cleanenv \
    --bind "$(dirname "$WORKDIR"):/home/weiyibiao" \
    pytorch38_cu111_pyg.sif \
    python /home/weiyibiao/MSCRS-main/rec/src/train_rec_inspired.py \
      --dataset /home/weiyibiao/MSCRS-main/data/rec_data/inspired \
      --tokenizer /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --model     /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --text_tokenizer /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --text_encoder   /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --prompt_encoder /home/weiyibiao/MSCRS-main/train_runs/pretrain_run/pretrain_inspired6/best \
      --output_dir "${inspired_rec_run_dir}" \
      --num_train_epochs 10

