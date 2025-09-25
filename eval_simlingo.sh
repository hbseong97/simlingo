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
#SBATCH --nodelist=DGX-H100-11


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



# load_path = '/mnt/raid12/cache/huggingface/hub/models--RenzKa--simlingo/snapshots/26c7c89e797d4e25bbf640013317af8da26a5454/simlingo/checkpoints/epoch=013.ckpt'
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/05-45-21/checkpoints/epoch=004.ckpt' # Internvl2
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/12-13-14/checkpoints/epoch=002.ckpt' # Internvl3
# # 3 batch
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/13-12-28/checkpoints/last.ckpt' # /mnt/harbor/projects/owa/checkpoints/InternVL3-1B-hf-no_seminit
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/13-31-03/checkpoints/last.ckpt' # /mnt/harbor/users/jyjung/checkpoints/iclr_agent/InternVL3-1B-HF_0ms/checkpoint-12892
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/13-45-51/checkpoints/last.ckpt' # /mnt/harbor/users/jyjung/checkpoints/iclr_agent/InternVL3-1B-HF_0ms-PT-FT/checkpoint-9669
# # 1 epoch
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/15-49-38/checkpoints/last.ckpt' # /mnt/harbor/projects/owa/checkpoints/InternVL3-1B-hf-no_seminit
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/15-49-41/checkpoints/last.ckpt' # /mnt/harbor/users/jyjung/checkpoints/iclr_agent/InternVL3-1B-HF_0ms/checkpoint-12892
# load_path = '/mnt/raid12/scratch/simlingo/outputs/2025-09-23/15-49-43/checkpoints/last.ckpt' # /mnt/harbor/users/jyjung/checkpoints/iclr_agent/InternVL3-1B-HF_0ms-PT-FT/checkpoint-9669




# Evaluate checkpoints from step 1000 to 3500 with 500 stride
for STEP in 003000; do
    echo "Evaluating step ${STEP}..."

    # # Model 1: /mnt/harbor/projects/owa/checkpoints/InternVL3-1B-hf-no_seminit
    # LOAD_PATH="/mnt/home/haebin/raid12/scratch/simlingo/outputs/2025-09-24/01-08-01/checkpoints/epoch=000-step=${STEP}.ckpt"
    # echo "Evaluating: ${LOAD_PATH}"
    # python -m simlingo_training.eval load_path=${LOAD_PATH//=/\\=}

    # # Model 2: /mnt/harbor/users/jyjung/checkpoints/iclr_agent/InternVL3-1B-HF_0ms/checkpoint-12892
    # LOAD_PATH="/mnt/home/haebin/raid12/scratch/simlingo/outputs/2025-09-24/01-09-48/checkpoints/epoch=000-step=${STEP}.ckpt"
    # echo "Evaluating: ${LOAD_PATH}"
    # python -m simlingo_training.eval load_path=${LOAD_PATH//=/\\=}

    # # Model 3: /mnt/harbor/users/jyjung/checkpoints/iclr_agent/InternVL3-1B-HF_0ms-PT-FT/checkpoint-9669
    # LOAD_PATH="/mnt/home/haebin/raid12/scratch/simlingo/outputs/2025-09-24/01-11-21/checkpoints/epoch=000-step=${STEP}.ckpt"
    # echo "Evaluating: ${LOAD_PATH}"
    # python -m simlingo_training.eval load_path=${LOAD_PATH//=/\\=}

    # exp 9
    LOAD_PATH="/mnt/raid12/scratch/simlingo/outputs/2025-09-24/22-25-50/checkpoints/epoch=000-step=${STEP}.ckpt"
    echo "Evaluating: ${LOAD_PATH}"
    HYDRA_FULL_ERROR=1 python -m simlingo_training.eval load_path=${LOAD_PATH//=/\\=}

    # exp 10
    LOAD_PATH="/mnt/raid12/scratch/simlingo/outputs/2025-09-24/22-25-43/checkpoints/epoch=000-step=${STEP}.ckpt"
    echo "Evaluating: ${LOAD_PATH}"
    HYDRA_FULL_ERROR=1 python -m simlingo_training.eval load_path=${LOAD_PATH//=/\\=}

    echo "Completed step ${STEP}"
    echo "----------------------------------------"
done