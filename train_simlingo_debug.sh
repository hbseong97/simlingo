#!/bin/bash
#SBATCH --job-name=slv2_debug
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --output=logs/slv2_debug_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=logs/slv2_debug_%a_%A.err   # File to which STDERR will be written
#SBATCH --partition=h100

# print info about current job
scontrol show job $SLURM_JOB_ID

conda env list

pwd
export CARLA_ROOT=~/software/carla0915/
export WORK_DIR=/mnt/raid12/scratch/simlingo
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export SCENARIO_RUNNER_ROOT=${WORK_DIR}/scenario_runner
export LEADERBOARD_ROOT=${WORK_DIR}/leaderboard
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":${PYTHONPATH}

export PYTHONPATH=$PYTHONPATH:${WORK_DIR}

export MASTER_ADDR=localhost
export NCCL_DEBUG=INFO

export OMP_NUM_THREADS=64 # Limits pytorch to spawn at most num cpus cores threads
export OPENBLAS_NUM_THREADS=1  # Shuts off numpy multithreading, to avoid threads spawning other threads.
WANDB__SERVICE_WAIT=300 HYDRA_FULL_ERROR=1 python simlingo_training/train.py experiment=debug data_module.batch_size=8 gpus=1 name=simlingo_seed1