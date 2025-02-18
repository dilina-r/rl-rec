conda activate tf1

############################## SQN PROTOCOL Scripts ####################################

# create directory to save trained models
mkdir -p trained_models/RL/gru_protocol

# Yoochoose Dataset - SASRec baseline, SQN-SAS and EVAL-SAS
python RL_RS/SASRec.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df --lr=0.0055 --hidden_factor=256 --epoch=40 --dropout=0.3 --batch_size=244 \
    --save_file=trained_models/RL/gru_protocol/yoochoose_sasrec.pkt

python RL_RS/SQN.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df --model=SASRec --lr=0.0055 --hidden_factor=256 --epoch=40 \
    --discount=0.6 --dropout=0.3 --batch_size=244 --save_file=trained_models/RL/gru_protocol/yoochoose_sas_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df --model=SASRec --lr=0.0055 --hidden_factor=256 --epoch=40 \
    --discount=0.6 --dropout=0.3 --batch_size=244 --save_file=trained_models/RL/gru_protocol/yoochoose_sas_eval.pkt 

# Yoochoose Dataset - GRU baseline, SQN-GRU and EVAL-GRU
python RL_RS/GRU.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df --lr=0.003 --hidden_factor=64 --epoch=40 --batch_size=512 \
    --save_file=trained_models/RL/gru_protocol/yoochoose_gru.pkt

python RL_RS/SQN.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df --model=GRU --lr=0.003 --hidden_factor=64 --epoch=40 --discount=0 --batch_size=512 \
    --save_file=trained_models/RL/gru_protocol/yoochoose_gru_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/Yoochoose/gru_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/gru_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/gru_protocol/data_statis.df --model=GRU --lr=0.003 --hidden_factor=64 --epoch=40 --discount=0 --batch_size=512 \
    --save_file=trained_models/RL/gru_protocol/yoochoose_gru_eval.pkt 


# Retailrocket Dataset - SASRec baseline, SQN-SAS and EVAL-SAS
python RL_RS/SASRec.py --replay_buffer=Data/RetailRocket/gru_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/gru_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/gru_protocol/data_statis.df --lr=0.0013 --hidden_factor=384 --epoch=40 --dropout=0.5 --batch_size=128 \
    --save_file=trained_models/RL/gru_protocol/retailrocket_sasrec.pkt

python RL_RS/SQN.py --replay_buffer=Data/RetailRocket/gru_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/gru_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/gru_protocol/data_statis.df --model=SASRec --lr=0.0013 --hidden_factor=384 --epoch=40 --discount=0.7 --dropout=0.5 --batch_size=128 \
    --save_file=trained_models/RL/gru_protocol/retailrocket_sas_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/RetailRocket/gru_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/gru_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/gru_protocol/data_statis.df --model=SASRec --lr=0.0013 --hidden_factor=384 --epoch=40 --discount=0.7 --dropout=0.5 --batch_size=128 \
    --save_file=trained_models/RL/gru_protocol/retailrocket_sas_eval.pkt

# Retailrocket Dataset - GRU baseline, SQN-GRU and EVAL-GRU
python RL_RS/GRU.py --replay_buffer=Data/RetailRocket/gru_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/gru_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/gru_protocol/data_statis.df --lr=0.000475 --hidden_factor=512 --epoch=40 --batch_size=512 \
    --save_file=trained_models/RL/gru_protocol/retailrocket_gru.pkt

python RL_RS/SQN.py --replay_buffer=Data/RetailRocket/gru_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/gru_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/gru_protocol/data_statis.df --model=GRU --lr=0.000475 --hidden_factor=512 --epoch=40 --batch_size=512 --discount=0.2 \
    --save_file=trained_models/RL/gru_protocol/retailrocket_gru_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/RetailRocket/gru_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/gru_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/gru_protocol/data_statis.df --model=GRU --lr=0.000475 --hidden_factor=512 --epoch=40 --batch_size=512 --discount=0.2 \
    --save_file=trained_models/RL/gru_protocol/retailrocket_gru_eval.pkt 


# Diginetica Dataset - SASRec baseline, SQN-SAS and EVAL-SAS
python RL_RS/SASRec.py --replay_buffer=Data/Diginetica/gru_protocol/replay_buffer_full.df --test_file=Data/Diginetica/gru_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/gru_protocol/data_statis.df --lr=0.0055 --hidden_factor=256 --epoch=40 --dropout=0.3 --batch_size=244 \
    --save_file=trained_models/RL/gru_protocol/diginetica_sasrec.pkt 

python RL_RS/SQN.py --replay_buffer=Data/Diginetica/gru_protocol/replay_buffer_full.df --test_file=Data/Diginetica/gru_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/gru_protocol/data_statis.df --model=SASRec --lr=0.0055 --hidden_factor=256 --epoch=40 --dropout=0.3 --batch_size=244 --discount=0.6 \
    --save_file=trained_models/RL/gru_protocol/diginetica_sas_sqn.pkt 

python RL_RS/EVAL.py --replay_buffer=Data/Diginetica/gru_protocol/replay_buffer_full.df --test_file=Data/Diginetica/gru_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/gru_protocol/data_statis.df --model=SASRec --lr=0.0055 --hidden_factor=256 --epoch=40 --dropout=0.3 --batch_size=244 --discount=0.6 \
    --save_file=trained_models/RL/gru_protocol/diginetica_sas_eval.pkt 

# Diginetica Dataset - GRU baseline, SQN-GRU and EVAL-GRU
python RL_RS/GRU.py --replay_buffer=Data/Diginetica/gru_protocol/replay_buffer_full.df --test_file=Data/Diginetica/gru_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/gru_protocol/data_statis.df --lr=0.001 --hidden_factor=512 --epoch=40 --batch_size=160 \
    --save_file=trained_models/RL/gru_protocol/diginetica_gru.pkt 

python RL_RS/SQN.py --replay_buffer=Data/Diginetica/gru_protocol/replay_buffer_full.df --test_file=Data/Diginetica/gru_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/gru_protocol/data_statis.df --model=GRU --lr=0.001 --hidden_factor=512 --epoch=40 --batch_size=160 --discount=0.2 \
    --save_file=trained_models/RL/gru_protocol/diginetica_gru_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/Diginetica/gru_protocol/replay_buffer_full.df --test_file=Data/Diginetica/gru_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/gru_protocol/data_statis.df --model=GRU --lr=0.001 --hidden_factor=512 --epoch=40 --batch_size=160 --discount=0.2 \
    --save_file=trained_models/RL/gru_protocol/diginetica_gru_eval.pkt



############################## SQN PROTOCOL Scripts ####################################

# create directory to save trained models
mkdir -p trained_models/RL/sqn_protocol

# Yoochoose Dataset - SASRec baseline, SQN-SAS and EVAL-SAS
python RL_RS/SASRec.py --replay_buffer=Data/Yoochoose/sqn_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/sqn_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/sqn_protocol/data_statis.df --lr=0.001 --hidden_factor=384 --epoch=40 --dropout=0.7 --batch_size=512 \
    --save_file=trained_models/RL/sqn_protocol/yoochoose_sasrec.pkt 

python RL_RS/SQN.py --replay_buffer=Data/Yoochoose/sqn_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/sqn_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/sqn_protocol/data_statis.df --model=SASRec --lr=0.001 --hidden_factor=384 --epoch=40 --dropout=0.7 --batch_size=512 --discount=0.2 \
    --save_file=trained_models/RL/sqn_protocol/yoochoose_sas_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/Yoochoose/sqn_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/sqn_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/sqn_protocol/data_statis.df --model=SASRec --lr=0.001 --hidden_factor=384 --epoch=40 --dropout=0.7 --batch_size=512 --discount=0.2 \
    --save_file=trained_models/RL/sqn_protocol/yoochoose_sas_eval.pkt


# Yoochoose Dataset - GRU baseline, SQN-GRU and EVAL-GRU
python RL_RS/GRU.py --replay_buffer=Data/Yoochoose/sqn_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/sqn_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/sqn_protocol/data_statis.df --lr=0.001 --hidden_factor=64 --epoch=40 --batch_size=64 \
    --save_file=trained_models/RL/sqn_protocol/yoochoose_gru.pkt
    
python RL_RS/SQN.py --replay_buffer=Data/Yoochoose/sqn_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/sqn_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/sqn_protocol/data_statis.df --model=GRU --lr=0.001 --hidden_factor=64 --epoch=40 --batch_size=64 --discount=0.5 \
    --save_file=trained_models/RL/sqn_protocol/yoochoose_gru_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/Yoochoose/sqn_protocol/replay_buffer_full.df --test_file=Data/Yoochoose/sqn_protocol/sampled_test.df \
    --stats_file=Data/Yoochoose/sqn_protocol/data_statis.df --model=GRU --lr=0.001 --hidden_factor=64 --epoch=40 --batch_size=64 --discount=0.5 \
    --save_file=trained_models/RL/sqn_protocol/yoochoose_gru_eval.pkt 


# Retailrocket Dataset - SASRec baseline, SQN-SAS and EVAL-SAS
python RL_RS/SASRec.py --replay_buffer=Data/RetailRocket/sqn_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/sqn_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/sqn_protocol/data_statis.df --lr=0.0001 --hidden_factor=256 --epoch=40 --dropout=0.6 --batch_size=512 \
    --save_file=trained_models/RL/sqn_protocol/retailrocket_sasrec.pkt

python RL_RS/SQN.py --replay_buffer=Data/RetailRocket/sqn_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/sqn_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/sqn_protocol/data_statis.df --model=SASRec --lr=0.0001 --hidden_factor=256 --epoch=40 --dropout=0.6 --batch_size=512 --discount=0.5 \
    --save_file=trained_models/RL/sqn_protocol/retailrocket_sas_sqn.pkt


python RL_RS/EVAL.py --replay_buffer=Data/RetailRocket/sqn_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/sqn_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/sqn_protocol/data_statis.df --model=SASRec --lr=0.0001 --hidden_factor=256 --epoch=40 --dropout=0.6 --batch_size=512 --discount=0.5 \
    --save_file=trained_models/RL/sqn_protocol/retailrocket_sas_eval.pkt

# Retailrocket Dataset - GRU baseline, SQN-GRU and EVAL-GRU
python RL_RS/GRU.py --replay_buffer=Data/RetailRocket/sqn_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/sqn_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/sqn_protocol/data_statis.df --lr=0.0015 --hidden_factor=256 --epoch=40 --batch_size=384 \
    --save_file=trained_models/RL/sqn_protocol/retailrocket_gru.pkt

python RL_RS/SQN.py --replay_buffer=Data/RetailRocket/sqn_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/sqn_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/sqn_protocol/data_statis.df --model=GRU --lr=0.0015 --hidden_factor=256 --epoch=40 --batch_size=384 --discount=0.4 \
    --save_file=trained_models/RL/sqn_protocol/retailrocket_gru_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/RetailRocket/sqn_protocol/replay_buffer_full.df --test_file=Data/RetailRocket/sqn_protocol/sampled_test.df \
    --stats_file=Data/RetailRocket/sqn_protocol/data_statis.df --model=GRU --lr=0.0015 --hidden_factor=256 --epoch=40 --batch_size=384 --discount=0.4 \
    --save_file=trained_models/RL/sqn_protocol/retailrocket_gru_eval.pkt 


# Diginetica Dataset - SASRec baseline, SQN-SAS and EVAL-SAS
python RL_RS/SASRec.py --replay_buffer=Data/Diginetica/sqn_protocol/replay_buffer_full.df --test_file=Data/Diginetica/sqn_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/sqn_protocol/data_statis.df --lr=0.0001 --hidden_factor=320 --epoch=40 --dropout=0.6 --batch_size=384 \
    --save_file=trained_models/RL/sqn_protocol/diginetica_sasrec.pkt

python RL_RS/SQN.py --replay_buffer=Data/Diginetica/sqn_protocol/replay_buffer_full.df --test_file=Data/Diginetica/sqn_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/sqn_protocol/data_statis.df --model=SASRec --lr=0.0001 --hidden_factor=320 --epoch=40 --dropout=0.6 --batch_size=384 --discount=0.2 \
    --save_file=trained_models/RL/sqn_protocol/diginetica_sas_sqn.pkt

python RL_RS/EVAL.py --replay_buffer=Data/Diginetica/sqn_protocol/replay_buffer_full.df --test_file=Data/Diginetica/sqn_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/sqn_protocol/data_statis.df --model=SASRec --lr=0.0001 --hidden_factor=320 --epoch=40 --dropout=0.6 --batch_size=384 --discount=0.2 \
    --save_file=trained_models/RL/sqn_protocol/diginetica_sas_eval.pkt 

# Diginetica Dataset - GRU baseline, SQN-GRU and EVAL-GRU
python RL_RS/GRU.py --replay_buffer=Data/Diginetica/sqn_protocol/replay_buffer_full.df --test_file=Data/Diginetica/sqn_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/sqn_protocol/data_statis.df --lr=0.005 --hidden_factor=128 --epoch=40 --batch_size=512 \
    --save_file=trained_models/RL/sqn_protocol/diginetica_gru.pkt 

python RL_RS/SQN.py --replay_buffer=Data/Diginetica/sqn_protocol/replay_buffer_full.df --test_file=Data/Diginetica/sqn_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/sqn_protocol/data_statis.df --model=GRU --lr=0.005 --hidden_factor=128 --epoch=40 --batch_size=512 --discount=0 \
    --save_file=trained_models/RL/sqn_protocol/diginetica_gru_sqn.pkt 

python RL_RS/EVAL.py --replay_buffer=Data/Diginetica/sqn_protocol/replay_buffer_full.df --test_file=Data/Diginetica/sqn_protocol/sampled_test.df \
    --stats_file=Data/Diginetica/sqn_protocol/data_statis.df --model=GRU --lr=0.005 --hidden_factor=128 --epoch=40 --batch_size=512 --discount=0 \
    --save_file=trained_models/RL/sqn_protocol/diginetica_gru_eval.pkt  