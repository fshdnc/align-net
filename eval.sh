#!/bin/bash
#SBATCH -J eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:10:00
#SBATCH --partition=gputest
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks-per-node=1
#SBATCH --account=project_2002820
#SBATCH --output=/scratch/project_2002820/lihsin/align-lang/log-eval-%j.out
#SBATCH --error=/scratch/project_2002820/lihsin/align-lang/log-eval-%j.err

echo "START $SLURM_JOBID: $(date)"

function on_exit {
seff $SLURM_JOBID
gpuseff $SLURM_JOBID
echo "END $SLURM_JOBID: $(date)"
}
trap on_exit EXIT

#CKPT_PATH="/scratch/project_2002820/lihsin/align-lang/model_20201223-221953.pt"
CKPT_PATH="/scratch/project_2002820/lihsin/align-lang/model_20201228-174218.pt"

module purge
module load pytorch/1.3.1
python3 evaluation.py $CKPT_PATH

