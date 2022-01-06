python ./test.py --data_dir /home/akaruvally/work_dir/binds_data/Sims4ActionVideos \
                  --environment_config /home/akaruvally/temporal_nn_models/environment_configs/task_long.yaml \
                  --model CDNA \
                  --output_dir /home/akaruvally/scratch_dir/experiments/temporal_nn_LR=0.001_TASK=long/test_results \
                  --pretrained_model /home/akaruvally/scratch_dir/experiments/temporal_nn_LR=0.001_TASK=long/net_epoch_9.pth \
                  --context_frames 10 \
                  --batch_size 1 \
                  --device cuda \
                  --height 100 \
                  --width 100
