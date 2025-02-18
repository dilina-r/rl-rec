conda activate tf1

python RL_RS/SASRec.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df \
    --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df \
    --lr=0.0055 --hidden_factor=256 --epoch=30 --dropout=0.3 --batch_size=244 \
    --save_file=trained_models/gru_protocol/yoochoose_sasrec.pkt

python RL_RS/SQN.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df \
    --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df \
    --model=SASRec --lr=0.0055 --hidden_factor=256 --epoch=30 \
    --discount=0.6 --dropout=0.3 --batch_size=244 \
    --save_file=trained_models/gru_protocol/yoochoose_sas_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df \
    --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df \
    --model=SASRec --lr=0.0055 --hidden_factor=256 --epoch=30 \
    --discount=0.6 --dropout=0.3 --batch_size=244 \
    --save_file=trained_models/gru_protocol/yoochoose_eval_sqn.pkt