python ./test.py --data_dir /home/akaruvally/work_dir/binds_data/Sims4ActionVideos \
                  --model CDNA \
                  --output_dir /home/akaruvally/scratch_dir/experiments/temporal_nn_test/test_results \
                  --pretrained_model /home/akaruvally/scratch_dir/experiments/temporal_nn_LR=0.0001/net_epoch_29.pth \
                  --context_frames 2 \
                  --batch_size 1 \
                  --device cuda \
                  --height 64 \
                  --width 128
