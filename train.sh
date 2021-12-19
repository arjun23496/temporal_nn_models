python ./train.py --data_dir /media/arjun/Shared/research_projects/temporal_weight_gating/Sims4ActionVideos \
                  --model CDNA \
                  --output_dir ./weights \
                  --context_frames 2 \
                  --batch_size 3 \
                  --learning_rate 0.001 \
                  --epochs 10 \
                  --print_interval 10 \
                  --device cpu \
                  --use_state \
                  --height 64 \
                  --width 128
#                  --pretrained_model model