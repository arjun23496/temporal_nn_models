python ./train.py --data_dir /media/arjun/Shared/research_projects/temporal_weight_gating/Sims4ActionVideos \
                  --model CDNA \
                  --output_dir ./weights \
                  --sequence_length 10 \
                  --context_frames 2 \
                  --num_masks 10 \
                  --schedsamp_k 900.0 \
                  --batch_size 32 \
                  --learning_rate 0.001 \
                  --epochs 10 \
                  --print_interval 10 \
                  --device cpu \
                  --use_state \
#                  --pretrained_model model