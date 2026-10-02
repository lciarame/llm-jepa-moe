#!/bin/bash
#SBATCH --account=PHD_ciaramel
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
##SBATCH --exclude=lrdn[2371-3456]
#SBATCH --gres=gpu:4

module load python
module load cuda/12.2

source /leonardo_work/PHD_ciaramel/new_jepa_env/bin/activate

export HF_TOKEN="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"

##source ./run.sh #per ottenere le funzioni del run

export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

echo "Starting JEPA Llama training..." >> output.txt

#export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
#export MASTER_PORT=29500

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

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True #per migliorare l'uso della memoria
#finetunejepa.py se voglio vedere se viene calcolata jepa losg
srun torchrun $DISTRIBUTED_ARGS --nproc_per_node=4 finetune_exp13.py \
  --train_file synth_train.jsonl \
  --model_name /leonardo_scratch/large/userexternal/lciarame/hf_cache/hub/models--allenai--OLMoE-1B-7B-0924-Instruct \
  --output_dir $OUTPUT_DIR \
  --num_epochs 6 \
  --learning_rate 2e-5 \
  --finetune_seed 82 \
  --last_token -1 \
  --predictors 2 \
  --lbd 20 \
  #--regular \
  #--grad_accum 32 \
  #--max_length 4096 \
  #--batch_size 2 \
  #--track_flop \
  #--batch_size 1 \
  #--grad_accum 64 \
  #--max_length 128 \

echo "Training completed. Evaluating..." >> output.txt

#./run_jepa meta-llama/Llama-3.2-1B 1e-5 4 -2 1 82 0.5 gsm8k
