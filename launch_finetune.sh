#!/bin/bash
#SBATCH --account=your_account
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=training-jepa
#SBATCH --time=08:00:00
#SBATCH --nodes=4
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:4

module load python
module load cuda/12.2

source #path environment

export HF_TOKEN="Your_HF_key"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Starting JEPA Llama training..." >> output.txt

# Distributed args
nodes=( $(scontrol show hostnames $SLURM_JOB_NODELIST) )
nodes_array=("${nodes[@]}")
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address | awk '{print $1}')
export NNODES=${#nodes_array[@]}
export GPUS_PER_NODE=4
export WORLD_SIZE=$NNODES*$GPUS_PER_NODE
echo "###NNODES:$NNODES\n"

export DISTRIBUTED_ARGS="--rdzv_id=$RANDOM \
 --rdzv_backend=c10d \
 --rdzv_endpoint=$head_node_ip:29505 \
 --nnodes=$NNODES \
 --nproc_per_node=$GPUS_PER_NODE"

export OUTPUT_DIR="output_${SLURM_JOB_ID}"

mkdir -p "$OUTPUT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True 

srun torchrun $DISTRIBUTED_ARGS --nproc_per_node=4 finetune.py \
  --train_file synth_train.jsonl \
  --model_name #model_path \
  --output_dir $OUTPUT_DIR \
  --num_epochs 6 \
  --learning_rate 2e-5 \
  --finetune_seed 82 \
  --last_token -1 \
  --predictors 2 \
  --lbd 20 \


echo "Training completed. Evaluating..." >> output.txt