#!/bin/bash
#SBATCH --job-name=mscrs_pretrain
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@600      # 10 minutes before walltime (AAU example uses SIGTERM@30) :contentReference[oaicite:2]{index=2}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G
#SBATCH --requeue                   # allow requeueing

# tweak this
max_restarts=10

# figure out restart count (AAU guide pattern) :contentReference[oaicite:3]{index=3}
scontext=$(scontrol show job ${SLURM_JOB_ID})
restarts=$(echo ${scontext} | grep -o 'Restarts=[0-9]*' | cut -d= -f2)
iteration=${restarts:-0}

OUTDIR="/ceph/project/rtm-p10/MSCRS-main/pretrain_run1"   # persistent
mkdir -p "$OUTDIR"

outfile="${SLURM_JOB_ID}_${iteration}.out"
errfile="${SLURM_JOB_ID}_${iteration}.err"

echo "Output file: ${outfile}"
echo "Error file: ${errfile}"
echo "Checkpoint dir: ${OUTDIR}"

term_handler() {
  echo "Executing term handler at $(date)"

  if [[ ${iteration} -lt ${max_restarts} ]]; then
    echo "Requesting requeue..."
    scontrol requeue ${SLURM_JOB_ID}
    echo "Requeue requested. Waiting for training process to finish checkpointing..."
    wait ${SRUN_PID}
    exit 0
  else
    echo "Maximum restarts reached; not requeueing."
    exit 1
  fi
}

trap 'term_handler' SIGTERM

WORKDIR="/ceph/project/rtm-p10/MSCRS-main"
cd "$WORKDIR"

# Launch training
srun --output="${outfile}" --error="${errfile}" \
  singularity exec --nv --cleanenv \
    --bind "$(dirname "$WORKDIR"):/home/weiyibiao" \
    pytorch38_cu111_pyg.sif \
    python /home/weiyibiao/MSCRS-main/rec/src/train_pre_redial.py \
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
  &

SRUN_PID=$!
wait ${SRUN_PID}

