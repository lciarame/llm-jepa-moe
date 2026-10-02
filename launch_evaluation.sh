#!/bin/bash
#SBATCH --account=Your_account
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --partition=boost_usr_prod
##SBATCH --qos=boost_qos_lprod
##SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=training-jepa
#SBATCH --time=12:15:00
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --cpus-per-task=32
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1

module load python
module load cuda/12.2

source #path environment

export HF_TOKEN="Your_HF_key"

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Starting evaluation..." >> output.txt

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
 --rdzv_backend=c10d"

export EVAL_OUT="eval_${SLURM_JOB_ID}"

srun torchrun $DISTRIBUTED_ARGS --nproc_per_node=1 evaluate.py \
  --model_name #MODEL_PATH
  --input_file  rotten_tomatoes_test.jsonl\
  --output_file $EVAL_OUT.jsonl \
  --device_map "auto" \
  --split_tune_untune \
  --original_model_name #MODEL_NAME \
  --nosplit_data | tee -a output.txt

echo "JEPA pipeline COMPLETA!" >> output.txt