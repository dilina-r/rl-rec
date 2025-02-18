#Provide the path to the directories of the raw datasets
YOOCHOOSE_DIR=""
RETAILROCKET_DIR=""
DIGINETICA_DIR=""

# GRU4Rec Protocol Data scripts
python preprocess/gru_protocol/preprocess_yoochoose.py --src=$YOOCHOOSE_DIR --dst=Data/Yoochoose/gru_protocol/ &&
python preprocess/replay_buffer.py --path=Data/Yoochoose/gru_protocol/ &&

python preprocess/gru_protocol/preprocess_retailrocket.py --src=$RETAILROCKET_DIR --dst=Data/RetailRocket/gru_protocol/ &&
python preprocess/replay_buffer.py --path=Data/RetailRocket/gru_protocol/ &&

python preprocess/gru_protocol/preprocess_diginetica.py --src=$DIGINETICA_DIR --dst=Data/Diginetica/gru_protocol/ &&
python preprocess/replay_buffer.py --path=Data/Diginetica/gru_protocol/ &&

# SQN Protocol Data scripts
python preprocess/sqn_protocol/preprocess_yoochoose.py --src=$YOOCHOOSE_DIR --dst=Data/Yoochoose/sqn_protocol/ &&
python preprocess/replay_buffer.py --path=Data/Yoochoose/sqn_protocol/ &&

python preprocess/sqn_protocol/preprocess_retailrocket.py --src=$RETAILROCKET_DIR --dst=Data/RetailRocket/sqn_protocol/ &&
python preprocess/replay_buffer.py --path=Data/RetailRocket/sqn_protocol/ &&

python preprocess/sqn_protocol/preprocess_diginetica.py --src=$DIGINETICA_DIR --dst=Data/Diginetica/sqn_protocol/ &&
python preprocess/replay_buffer.py --path=Data/Diginetica/sqn_protocol/ 