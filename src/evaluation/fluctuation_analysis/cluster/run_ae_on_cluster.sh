#!/bin/bash
#SBATCH --partition=clara
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=40GB
#SBATCH --gres=gpu:v100
#SBATCH --mail-type=FAIL
#SBATCH -o /work/users/$USER/final/logs/tf/job_eval_%A_%a.log

export IDS_ON_CLUSTER=1

module load matplotlib/3.4.3-foss-2021b
module load CUDA/11.3.1
module load Python/3.9.6-GCCcore-11.2.0
module load SciPy-bundle/2021.10-foss-2021b
module load PyTorch/1.10.0-foss-2021a-CUDA-11.3.1
module load tensorboard/2.8.0-foss-2021a

export PYTHONPATH="$HOME/lidds_wt/thesis/LID-DS:$PYTHONPATH"

# parameters:
# 1: -bp base_path
# 2: -ds dataset
# 3: -s scenario_name
# 4: -c checkpoint_dir
# 5: -n ngram_length
# 6: -cs custom_split
# 7:-eal eval_after_load
# 8: -do dropout
# 9: -e evaluate
python fluctuation_analysis_ae.py -bp "$1" -ds "$2" -s "$3" -c "$4" -n "$5" -cs "$6" -eal "$7" -do "$8" -e "$9"
