# LLM-JEPA-MoE

Code and experiments for integrating JEPA with Large Language Models and Mixture-of-Experts architectures.

This repository contains the code used for the JEPA-related experiments presented in the master's thesis.

## Reference

This work is based on the original [LLM-JEPA](https://github.com/galilai-group/llm-jepa) implementation by the Galilai group. Please refer to the original repository for additional details on the LLM-JEPA architecture and implementation.

## Repository Structure

```text
.
├── evaluate.py
├── finetune.py
├── pretrain.py
├── launch_evaluation.sh
├── launch_finetune.sh
├── launch_pretrain.sh
├── requirements.txt
└── LICENSE
```

The Python scripts contain the implementations for pre-training, fine-tuning and evaluation, while the corresponding `launch_*.sh` scripts are used to submit the experiments to an HPC cluster.

## Installation

Clone the repository:

```bash
git clone https://github.com/lciarame/llm-jepa-moe.git
cd llm-jepa-moe
```

Create a Python virtual environment and install the dependencies using the provided `requirements.txt` file:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The experiments were run with CUDA 12.2. Python 3.10 is recommended. The exact Python and CUDA setup may depend on the target HPC system.

## Models

The models used in the experiments are downloaded from [Hugging Face](https://huggingface.co/).

- If a model is gated (e.g. Llama), request access on its Hugging Face page and authenticate with `huggingface-cli login`.
- Compute nodes on HPC clusters often have no internet access: download the models beforehand from the login node.
- The path to the model must be set in the corresponding launch script before submitting a job.

## Running Experiments

The experiments are designed to run on an HPC cluster using SLURM. Each launch script is associated with a Python script, which must be kept in the same directory:

| Launch script          | Python script | Purpose           |
| ---------------------- | ------------- | ----------------- |
| `launch_pretrain.sh`   | `pretrain.py` | JEPA pre-training |
| `launch_finetune.sh`   | `finetune.py` | JEPA fine-tuning  |
| `launch_evaluation.sh` | `evaluate.py` | Model evaluation  |

The launch scripts are provided as templates containing the SLURM configuration and the arguments passed to the Python scripts. Before submitting a job, complete and adapt the corresponding `launch_*.sh` to your cluster (account, partition, paths, etc.).

Two sets of experiments are supported:

1. **Fine-tuning and evaluation**: fine-tune a pretrained model, then evaluate it.

```bash
   sbatch launch_finetune.sh
   sbatch launch_evaluation.sh
```

2. **Pre-training, fine-tuning and evaluation**: run JEPA pre-training first, then fine-tune the resulting model and evaluate it.

```bash
   sbatch launch_pretrain.sh
   sbatch launch_finetune.sh
   sbatch launch_evaluation.sh
```

Each step depends on the output of the previous one, so submit a job only after the preceding one has completed (or chain them with `sbatch --dependency=afterok:<JOBID>`). Make sure the paths in each launch script point to the outputs of the previous step.

## Citation

If you use this code, please cite the original LLM-JEPA work:

```bibtex
@misc{huang2025llmjepalargelanguagemodels,
      title={LLM-JEPA: Large Language Models Meet Joint Embedding Predictive Architectures},
      author={Hai Huang and Yann LeCun and Randall Balestriero},
      year={2025},
      eprint={2509.14252},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.14252},
}
```

Original implementation: https://github.com/galilai-group/llm-jepa

## License

This repository contains code derived from the original LLM-JEPA implementation and is released under the Apache License 2.0 (see the [LICENSE](LICENSE) file), the same license as the original project.
