#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output="/mnt/home/%u/joblog/%j"
#SBATCH --mail-user=mawni4ah2o@pomail.net
#SBATCH --mail-type=FAIL,TIME_LIMIT,ARRAY_TASKS
#SBATCH --requeue
#SBATCH --array=0-999

set -euo pipefail

echo "configuration ==========================================================="
JOBDATE="$(date '+%Y-%m-%d')"
echo "JOBDATE ${JOBDATE}"

JOBNAME="$(basename -s .sh "$0")"
echo "JOBNAME ${JOBNAME}"

JOBPROJECT="$(basename -s .git "$(git remote get-url origin)")"
echo "JOBPROJECT ${JOBPROJECT}"

SOURCE_REVISION="$(git rev-parse HEAD)"
echo "SOURCE_REVISION ${SOURCE_REVISION}"
SOURCE_REMOTE_URL="$(git config --get remote.origin.url)"
echo "SOURCE_REMOTE_URL ${SOURCE_REMOTE_URL}"

echo "initialization telemetry ==============================================="
echo "date $(date)"
echo "hostname $(hostname)"
echo "PWD ${PWD}"
echo "SLURM_JOB_ID ${SLURM_JOB_ID:-nojid}"
echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID:-notid}"
module purge || :
module load Python/3.10.8 || :
echo "python3.10 $(which python3.10)"
echo "python3.10 --version $(python3.10 --version)"

echo "setup HOME dirs ========================================================"
mkdir -p "${HOME}/joblatest"
mkdir -p "${HOME}/joblog"
mkdir -p "${HOME}/jobscript"
if ! [ -e "${HOME}/scratch" ]; then
    if [ -e "/mnt/scratch/${USER}" ]; then
        ln -s "/mnt/scratch/${USER}" "${HOME}/scratch" || :
    else
        mkdir -p "${HOME}/scratch" || :
    fi
fi

echo "launch job ============================================================="
SLURM_ARRAY_TASK_ID_="$SLURM_ARRAY_TASK_ID"
echo "SLURM_ARRAY_TASK_ID_ ${SLURM_ARRAY_TASK_ID_}"
for i in 0 1 2 3 4; do
    echo "subtask i=${i}"
    export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID_}${i}"
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
    singularity exec docker://ghcr.io/mmore500/hstrat-synthesis@sha256:70445f4dc3b41a137d4f79284feaa045f7ff8f0d802518cbdf53a05bbab86b74 python3 -m pylib.trafficsim_msprime "16"
done

for i in 5 6 7 8 9; do
    echo "subtask i=${i}"
    export SLURM_ARRAY_TASK_ID="${SLURM_ARRAY_TASK_ID_}${i}"
    echo "SLURM_ARRAY_TASK_ID ${SLURM_ARRAY_TASK_ID}"
    singularity exec docker://ghcr.io/mmore500/hstrat-synthesis@sha256:70445f4dc3b41a137d4f79284feaa045f7ff8f0d802518cbdf53a05bbab86b74 python3 -m pylib.trafficsim_msprime "32"
done
