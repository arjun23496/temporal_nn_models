#!/bin/bash
#SBATCH --partition=titanx-long    # Partition to submit to <m40-short|m40-long|teslax-short|teslax-long>
#SBATCH --job-name=emhop
#SBATCH -o /home/akaruvally/work_dir/temporal_nnmodels/sims_convLSTM_%A_%a.txt            # output file
#SBATCH -e /home/akaruvally/work_dir/temporal_nnmodels/sims_convLSTM_%A_%a.err            # File to which STDERR will be written
#SBATCH --ntasks=1
#SBATCH --time=1-00:00:00          # D-HH:MM:SS
#SBATCH --mem=10GB
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-3
#SBATCH --mail-user=akaruvally@cs.umass.edu

source /home/akaruvally/.bashrc

hostname

conda activate cache_analysis

cd /home/akaruvally/temporal_nn_models

LEARNING_RATES=(0.1 0.01 0.001 0.0001)

comb_id=0

for learning_rate in "${LEARNING_RATES[@]}"
do
  if (( $comb_id % 4 == $SLURM_ARRAY_TASK_ID ))
  then
    python ./train.py --data_dir /home/akaruvally/work_dir/binds_data/Sims4ActionVideos \
                      --model CDNA \
                      --output_dir /home/akaruvally/scratch_dir/experiments/temporal_nn_LR=${learning_rate} \
                      --context_frames 2 \
                      --batch_size 32 \
                      --learning_rate ${learning_rate} \
                      --epochs 30 \
                      --print_interval 10 \
                      --device cuda:0 \
                      --use_state \
                      --height 64 \
                      --width 128
  fi
  ((comb_id=comb_id+1))
done
