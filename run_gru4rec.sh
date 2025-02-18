conda activate pytorch

python GRU4Rec_pytorch/run.py Data/Yoochoose/gru_protocol/yoochoose_train_full.txt \
    -t Data/Yoochoose/gru_protocol/yoochoose_test.txt \
    -pf GRU4Rec_pytorch/paramfiles/gru_protocol/yoochoose.py \
    -s GRU4Rec_pytorch/saved_models/yoochoose_gru_protocol.pt \
    -d cuda:0 -m 5 10 20 