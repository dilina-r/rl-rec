conda activate pytorch

# create directory to save trained models
mkdir -p trained_models/NARM

# Yoochoose dataset - GRU Protocol
python main.py --train_file "Data/Yoochoose/gru_protocol/yoochoose_train_full.txt" --test_file "Data/Yoochoose/gru_protocol/yoochoose_test.txt" \
    --batch_size=144 --epoch=50 --lr=0.0006 --embed_dim=160 --n_items=17435 --gpu_id=0 --model_path=trained_models/NARM/yoochoose_gru_protocol.pt

# RetailRocket dataset - GRU Protocol
python main.py --train_file "Data/RetailRocket/gru_protocol/retailrocket_train_full.txt" --test_file "Data/RetailRocket/gru_protocol/retailrocket_test.txt" \
    --batch_size=144 --epoch=50 --lr=0.0001 --embed_dim=384 --n_items=48958 --gpu_id=0 --model_path=trained_models/NARM/retailrocket_gru_protocol.pt

# Diginetica dataset - GRU Protocol
python main.py --train_file "Data/Diginetica/gru_protocol/diginetica_train_full.txt" --test_file "Data/Diginetica/gru_protocol/diginetica_test.txt" \
    --batch_size=128 --epoch=50 --lr=0.0006 --embed_dim=96 --n_items=43105 --gpu_id=0 --model_path=trained_models/NARM/diginetica_gru_protocol.pt


# Yoochoose dataset - SQN Protocol
python main.py --train_file "Data/Yoochoose/sqn_protocol/yoochoose_train_full.txt" --test_file "Data/Yoochoose/sqn_protocol/yoochoose_test.txt" \
    --batch_size=96 --epoch=50 --lr=0.0006 --embed_dim=224 --n_items=25613 --gpu_id=0 --model_path=trained_models/NARM/yoochoose_sqn_protocol.pt

# RetailRocket dataset - SQN Protocol
python main.py --train_file "Data/RetailRocket/sqn_protocol/retailrocket_train_full.txt" --test_file "Data/RetailRocket/sqn_protocol/retailrocket_test.txt" \
    --batch_size=144 --epoch=50 --lr=0.0001 --embed_dim=256 --n_items=69370 --gpu_id=0 --model_path=trained_models/NARM/retailrocket_sqn_protocol.pt

# Diginetica dataset - SQN Protocol
python main.py --train_file "Data/Diginetica/sqn_protocol/diginetica_train_full.txt" --test_file "Data/Diginetica/sqn_protocol/diginetica_test.txt" \
    --batch_size=176 --epoch=50 --lr=0.0001 --embed_dim=512 --n_items=58313 --gpu_id=0 --model_path=trained_models/NARM/diginetica_sqn_protocol.pt