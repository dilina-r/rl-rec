
import os
import numpy as np
import pandas as pd
import datetime as dt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='Path to directory containing the raw Yoochoose data files')
parser.add_argument('--dst', help='Path to save the processed data')
args = parser.parse_args()

def rename_columns(df, columns):
    # if pd.__version__ < '1.4.3':
    #     df.rename(columns = columns, axis = 1, inplace = True)
    # else:
    #     df.rename(columns, inplace=True)
    new_cols = []
    for c in df.columns:
        if c in columns: new_cols.append(columns[c])
        else: new_cols.append(c)
    df.columns = new_cols
    return df



def main():
    split = 64
    num_test_days = 1
    tday = 86400 * num_test_days
    dataset = 'yoochoose'


    PATH_TO_ORIGINAL_DATA = args.src
    PATH_TO_PROCESSED_DATA = args.dst


    if not(os.path.exists(PATH_TO_PROCESSED_DATA)):
        os.makedirs(PATH_TO_PROCESSED_DATA)


    data = pd.read_csv( os.path.join(PATH_TO_ORIGINAL_DATA, 'yoochoose-clicks.dat'), sep=',', header=None, usecols=[0,1,2,3], dtype={0:np.int32, 1:str, 2:np.int64, 3:str})
    data.columns = ['SessionId', 'TimeStr', 'ItemId', 'Cate']
    data['Time'] = data.TimeStr.apply(lambda x: dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()) #This is not UTC. It does not really matter.
    del(data['TimeStr'])



    data['is_buy'] = 0
    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>1].index)]


    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]

    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=2].index)]

    tmax = data.Time.max()
    session_max_times = data.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-tday].index
    session_test = session_max_times[session_max_times > tmax-tday].index
    train = data[np.in1d(data.SessionId, session_train)]
    test = data[np.in1d(data.SessionId, session_test)]


    ##### Split RC15 fractions
    if split is not None:
        train.sort_values(["Time"], inplace=True)
        n = len(train) // split
        train = train[-n:]
        session_lengths = train.groupby('SessionId').size()
        train = train[np.in1d(train.SessionId, session_lengths[session_lengths>=2].index)]


    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=2].index)]


    ##### Map ItemIds 
    ItemId = train['ItemId'].unique()
    n_items = len(ItemId)  # Define n_items
    itemid_map = pd.Series(data=np.arange(n_items, dtype='int32'), index=ItemId, name='ItemIdx')
    train['ItemIdx'] = train['ItemId'].map(itemid_map)
    test['ItemIdx'] = test['ItemId'].map(itemid_map)
    train['ItemId'] = train['ItemIdx']
    test['ItemId'] = test['ItemIdx']
    del(train['ItemIdx']); del(test['ItemIdx'])

    tmax = train.Time.max()
    session_max_times = train.groupby('SessionId').Time.max()
    session_train = session_max_times[session_max_times < tmax-tday].index
    session_val = session_max_times[session_max_times > tmax-tday].index
    train_tr = train[np.in1d(train.SessionId, session_train)]
    valid = train[np.in1d(train.SessionId, session_val)]


    ###### Save Processed Sessions
    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, f'{dataset}_train_full.txt'), sep='\t', index=False)
    
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, f'{dataset}_test.txt'), sep='\t', index=False)

    print('Train split\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, f'{dataset}_train_tr.txt'), sep='\t', index=False)
    
    print('Val split\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
    valid.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, f'{dataset}_train_valid.txt'), sep='\t', index=False)

    # Rename columns names for fit SQN format and save 
    columns = {'SessionId' : 'session_id', 'ItemId' : 'item_id', 'Time' : 'timestamp'}
    train = rename_columns(train, columns)
    train.to_pickle(os.path.join(PATH_TO_PROCESSED_DATA, 'sampled_train_full.df'))

    test = rename_columns(test, columns)
    test.to_pickle(os.path.join(PATH_TO_PROCESSED_DATA, 'sampled_test.df'))

    train_tr = rename_columns(train_tr, columns)
    train_tr.to_pickle(os.path.join(PATH_TO_PROCESSED_DATA, 'sampled_train.df'))

    valid = rename_columns(valid, columns)
    valid.to_pickle(os.path.join(PATH_TO_PROCESSED_DATA, 'sampled_val.df'))



if __name__ == '__main__':
    main()
