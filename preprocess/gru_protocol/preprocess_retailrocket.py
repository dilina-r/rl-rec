import os
import numpy as np
import pandas as pd
import datetime as dt
from datetime import datetime, timezone, timedelta
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src', help='Path to directory containing the raw Retailrocket data files')
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
    num_test_days = 7
    num_val_days = 7
    tsplit = 1442545200

    dataset = 'retailrocket'
    min_session_len = 2
    SESSION_LENGTH = 30 * 60 ## 30 mins

    PATH_TO_ORIGINAL_DATA = args.src
    PATH_TO_PROCESSED_DATA = args.dst

    if not(os.path.exists(PATH_TO_PROCESSED_DATA)):
        os.makedirs(PATH_TO_PROCESSED_DATA)

    data = pd.read_csv( os.path.join(PATH_TO_ORIGINAL_DATA, 'events.csv'), sep=',', header=0, usecols=[0,1,2,3], dtype={0:np.int64, 1:np.int32, 2:str, 3:np.int32})
    #specify header names
    data.columns = ['Time','UserId','Type','ItemId']
    data['Time'] = (data.Time / 1000).astype( int )
    #sessionize
    data.sort_values(by=['UserId', 'Time'], ascending=True, inplace=True)
    # compute the time difference between queries
    tdiff = np.diff(data['Time'].values)
    # check which of them are bigger then session_th
    split_session = tdiff > SESSION_LENGTH
    split_session = np.r_[True, split_session]
    # check when the user chenges is data
    new_user = data['UserId'].values[1:] != data['UserId'].values[:-1]
    new_user = np.r_[True, new_user]
    # a new sessions stars when at least one of the two conditions is verified
    new_session = np.logical_or(new_user, split_session)
    # compute the session ids
    session_ids = np.cumsum(new_session)
    data['SessionId'] = session_ids
    data.sort_values( ['SessionId','Time'], ascending=True, inplace=True )


    views = data[data.Type == 'view']
    views['is_buy'] = 0
    # cart = data[data.Type == 'addtocart']
    # cart['is_buy'] = 1
    # trans = data[data.Type == 'transaction']
    # trans['is_buy'] = 2
    
    # # data = pd.concat([views, cart])
    
    data = views
    del data['Type']




    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=min_session_len].index)]

    item_supports = data.groupby('ItemId').size()
    data = data[np.in1d(data.ItemId, item_supports[item_supports>=5].index)]

    session_lengths = data.groupby('SessionId').size()
    data = data[np.in1d(data.SessionId, session_lengths[session_lengths>=min_session_len].index)]

    sbeg = data.groupby("SessionId").Time.min()

    print(dt.datetime.utcfromtimestamp(tsplit))
    tday = 86400 * num_test_days
    test = data[data.SessionId.isin(sbeg[sbeg >= tsplit - tday ].index)]
    train = data[data.Time < tsplit - tday ]

    session_length = train.groupby("SessionId").size()
    train = train[train.SessionId.isin(session_length[session_length >= min_session_len].index)]


    ##### Map ItemIds 
    ItemId = train['ItemId'].unique()
    n_items = len(ItemId)  # Define n_items
    itemid_map = pd.Series(data=np.arange(n_items, dtype='int32'), index=ItemId, name='ItemIdx')
    train['ItemIdx'] = train['ItemId'].map(itemid_map)
    test['ItemIdx'] = test['ItemId'].map(itemid_map)
    train['ItemId'] = train['ItemIdx']
    test['ItemId'] = test['ItemIdx']
    del(train['ItemIdx']); del(test['ItemIdx'])


    tday = 86400 * (num_test_days + num_val_days)
    train_tr = train[train.Time < tsplit - tday]
    valid = train[train.SessionId.isin(sbeg[sbeg >= tsplit - tday].index)]
    session_length2 = train_tr.groupby("SessionId").size()
    train_tr = train_tr[train_tr.SessionId.isin(session_length2[session_length2 >= min_session_len].index)]

    test = test[np.in1d(test.ItemId, train.ItemId)]
    tslength = test.groupby('SessionId').size()
    test = test[np.in1d(test.SessionId, tslength[tslength>=min_session_len].index)]


    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train), train.SessionId.nunique(), train.ItemId.nunique()))
    train.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, f'{dataset}_train_full.txt'), sep='\t', index=False)
    
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test), test.SessionId.nunique(), test.ItemId.nunique()))
    test.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, f'{dataset}_test.txt'), sep='\t', index=False)
    
    print('Train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_tr), train_tr.SessionId.nunique(), train_tr.ItemId.nunique()))
    train_tr.to_csv(os.path.join(PATH_TO_PROCESSED_DATA, f'{dataset}_train_tr.txt'), sep='\t', index=False)
    
    print('Val set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(valid), valid.SessionId.nunique(), valid.ItemId.nunique()))
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