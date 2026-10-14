#!/bin/bash

# Allocation account name
#SBATCH --account=ucb821_asc1
#SBATCH --job-name=desat_test3_charge_penalty3_origSpeeds
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=11:59:00
# Jobs to run, inclusive
#SBATCH --array=0-0
#SBATCH --partition=acpu
#SBATCH --qos=cpu-normal
#SBATCH --output=/scratch/alpine/chbe6195/job_output/job_%a-%j.out
#SBATCH --mail-type=ALL
#SBATCH --dependency=singleton
#SBATCH --mem=28G

module purge

echo "Loading modules"
module load python/3.10.2
module load gcc

echo "Activating virtual environment"
source /projects/chbe6195/.venv/bin/activate
export TMPDIR=/scratch/alpine/chbe6195/temp_dir

# Edit these settings directly for each run.
SETUP_ENV_MODULE="dynamic_imaging_rewardFlexible_flexible_specs_charge"
OBSERVATION_SPEC="charge_desat_area2_reward_32"
ACTION_SPEC="charge_desat_image_32"
REWARD_FUNCTION="multiply"
REWARD_WEIGHT="1.0"
FILTER_VALUE="10000.0"
N_CASES="1"
N_ENVS="24"

echo "Running training script"
echo "Setup env module: $SETUP_ENV_MODULE"
echo "Observation spec: $OBSERVATION_SPEC"
echo "Action spec: $ACTION_SPEC"
echo "Reward function: $REWARD_FUNCTION"
echo "Reward weight: $REWARD_WEIGHT"
echo "Filter value: $FILTER_VALUE"
echo "N cases: $N_CASES"
echo "N envs: $N_ENVS"

python3 -u /projects/chbe6195/DLA-Project-AVS/moving_target_training/dynamic_targets_time_uncertain_charge_train.py \
    --setup-env-module "$SETUP_ENV_MODULE" \
    --observation-spec "$OBSERVATION_SPEC" \
    --action-spec "$ACTION_SPEC" \
    --reward-function "$REWARD_FUNCTION" \
    --reward-weight "$REWARD_WEIGHT" \
    --filter-value "$FILTER_VALUE" \
    --n-cases "$N_CASES" \
    --n-envs "$N_ENVS" 

echo "== End of Job =="
