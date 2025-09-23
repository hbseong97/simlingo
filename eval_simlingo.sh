#!/bin/bash
#SBATCH --job-name=eval_slv2
#SBATCH --nodes=1
#SBATCH --time=3-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=50G
#SBATCH --output=logs/eval_slv2_%a_%A.out  # File to which STDOUT will be written
#SBATCH --error=logs/eval_slv2_%a_%A.err   # File to which STDERR will be written
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


python -m simlingo_training.eval