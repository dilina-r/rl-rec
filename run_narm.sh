conda activate pytorch

python main.py \
    --train_file "Data/Yoochoose/gru_protocol/yoochoose_train_full.txt" \
    --test_file "Data/Yoochoose/gru_protocol/yoochoose_test.txt" \
    --batch_size 144 --epoch 500 --lr 0.0006 --embed_dim 160 \
    --n_items 17435 --gpu_id 0 --dataset yoochoose_gru_protocol
