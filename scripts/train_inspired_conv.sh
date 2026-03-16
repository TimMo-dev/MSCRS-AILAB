#!/bin/bash
#SBATCH --job-name=mscrs_conv_train
#SBATCH --time=12:00:00
#SBATCH --signal=B:SIGTERM@600      # 10 minutes before walltime (AAU example uses SIGTERM@30) :contentReference[oaicite:2]{index=2}
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=15
#SBATCH --mem=24G

WORKDIR="/ceph/project/rtm-p10/MSCRS-main"
cd "$WORKDIR"

# Automatically find the next available conv_trainN directory
base_run_dir="train_runs/inspired_conv_run"
mkdir -p "$base_run_dir"

# Find next run number
current_iteration=1
while [ -d "$base_run_dir/conv_train${current_iteration}" ]; do
  current_iteration=$((current_iteration + 1))
done

conv_run_dir="$base_run_dir/conv_train${current_iteration}"
mkdir -p "$conv_run_dir"

outfile="${conv_run_dir}/${SLURM_JOB_ID}_conv_training.out"
errfile="${conv_run_dir}/${SLURM_JOB_ID}_conv_training.err"



# Launch training
# REMEMBER TO CHECK THAT THE OUTPUT DIRS AND PROMPT ENCODER IS CORRECT!!!!!
srun --output="${outfile}" --error="${errfile}" \
  singularity exec --nv --cleanenv \
    --bind "$(dirname "$WORKDIR"):/home/weiyibiao" \
    --env WANDB_DISABLED=true \
    pytorch38_cu111_pyg.sif \
    python /home/weiyibiao/MSCRS-main/conv/src/train_conv.py \
      --dataset /home/weiyibiao/MSCRS-main/data/conv_data/inspired \
      --tokenizer /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --model     /home/weiyibiao/MSCRS-main/hf_models/DialoGPT-small \
      --text_tokenizer /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --text_encoder   /home/weiyibiao/MSCRS-main/hf_models/roberta-base \
      --output_dir /home/weiyibiao/MSCRS-main/${conv_run_dir} \
      --save_each_epoch \
      --num_train_epochs 10 \
      --per_device_train_batch_size 4 \
      --per_device_eval_batch_size 4 \
      --gradient_accumulation_steps 8 \
      --n_examples 1 \
      --prompt_max_length 10 \
      --context_max_length 128 \
      --resp_max_length 50 \
      --fp16 
