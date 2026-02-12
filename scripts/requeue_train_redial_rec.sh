#!/bin/bash
#SBATCH --job-name=mscrs_rec_train
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@600      # 10 minutes before walltime (AAU example uses SIGTERM@30) :contentReference[oaicite:2]{index=2}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G

WORKDIR="/ceph/project/rtm-p10/MSCRS-main"
cd "$WORKDIR"

outfile="${SLURM_JOB_ID}_rec_training.out"
errfile="${SLURM_JOB_ID}_.rec_training.err"

# Launch training
srun --output="${outfile}" --error="${errfile}" \
  singularity exec --nv --cleanenv \
    --bind "$(dirname "$WORKDIR"):/home/weiyibiao" \
    pytorch38_cu111_pyg.sif \
    python /home/weiyibiao/MSCRS-main/rec/src/train_rec_redial.py \
      --dataset /home/weiyibiao/MSCRS-main/data/rec_data/redial \
      --tokenizer /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --model     /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --text_tokenizer /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --text_encoder   /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --output_dir "$OUTDIR" \
      --resume_from latest \
      --save_steps 1000 \
      --save_total_limit 3 \
      --num_train_epochs 10 \
