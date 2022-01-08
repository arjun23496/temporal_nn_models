MODEL_DIR="/home/akaruvally/scratch_dir/experiments/temporal_nn_LR=0.001_TASK=medium_classification"
python ./test.py --data_dir /home/akaruvally/work_dir/binds_data/Sims4ActionVideos \
                  --environment_config /home/akaruvally/temporal_nn_models/environment_configs/task_long.yaml \
                  --model CDNA \
                  --output_dir ${MODEL_DIR}/test_results \
                  --pretrained_model ${MODEL_DIR}/net_epoch_29.pth \
                  --context_frames 10 \
                  --num_actions 8 \
                  --batch_size 1 \
                  --device cuda \
                  --height 100 \
                  --width 100
