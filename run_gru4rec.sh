conda activate pytorch

# create directory to save trained models
mkdir -p trained_models/GRU4Rec

# GRU4Rec Protocol Data

python GRU4Rec/run.py Data/Yoochoose/gru_protocol/yoochoose_train_full.txt -t Data/Yoochoose/gru_protocol/yoochoose_test.txt \
    -pf GRU4Rec/paramfiles/gru_protocol/yoochoose.py -s trained_models/GRU4Rec/yoochoose_gru_protocol.pt -d cuda:0 -m 5 10 20 

python GRU4Rec/run.py Data/RetailRocket/gru_protocol/retailrocket_train_full.txt -t Data/RetailRocket/gru_protocol/retailrocket_test.txt \
    -pf GRU4Rec/paramfiles/gru_protocol/retailrocket.py -s trained_models/GRU4Rec/retailrocket_gru_protocol.pt -d cuda:0 -m 5 10 20 

python GRU4Rec/run.py Data/Diginetica/gru_protocol/diginetica_train_full.txt -t Data/Diginetica/gru_protocol/diginetica_test.txt \
    -pf GRU4Rec/paramfiles/gru_protocol/diginetica.py -s trained_models/GRU4Rec/diginetica_gru_protocol.pt -d cuda:0 -m 5 10 20 


# SQN Protocol Data

python GRU4Rec/run.py Data/Yoochoose/sqn_protocol/yoochoose_train_full.txt -t Data/Yoochoose/sqn_protocol/yoochoose_test.txt \
    -pf GRU4Rec/paramfiles/sqn_protocol/yoochoose.py -s trained_models/GRU4Rec/yoochoose_sqn_protocol.pt -d cuda:0 -m 5 10 20 

python GRU4Rec/run.py Data/RetailRocket/sqn_protocol/retailrocket_train_full.txt -t Data/RetailRocket/sqn_protocol/retailrocket_test.txt \
    -pf GRU4Rec/paramfiles/sqn_protocol/retailrocket.py -s trained_models/GRU4Rec/retailrocket_sqn_protocol.pt -d cuda:0 -m 5 10 20 

python GRU4Rec/run.py Data/Diginetica/sqn_protocol/diginetica_train_full.txt -t Data/Diginetica/sqn_protocol/diginetica_test.txt \
    -pf GRU4Rec/paramfiles/sqn_protocol/diginetica.py -s trained_models/GRU4Rec/diginetica_sqn_protocol.pt -d cuda:0 -m 5 10 20 