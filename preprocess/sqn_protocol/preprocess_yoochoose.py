import os

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
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

def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

if __name__ == '__main__':
    dataset = 'yoochoose'

    PATH_TO_ORIGINAL = args.src
    PATH_TO_PROCESSED = args.dst

    if not(os.path.exists(PATH_TO_PROCESSED)):
        os.makedirs(PATH_TO_PROCESSED)

    click_df = pd.read_csv(os.path.join(PATH_TO_ORIGINAL, 'yoochoose-clicks.dat'), header=None)
    click_df.columns = ['session_id', 'time', 'item_id','category']
    click_df['valid_session'] = click_df.session_id.map(click_df.groupby('session_id')['item_id'].size() > 2)
    click_df = click_df.loc[click_df.valid_session].drop('valid_session', axis=1)

    buy_df = pd.read_csv(os.path.join(PATH_TO_ORIGINAL, 'yoochoose-buys.dat'), header=None)
    buy_df.columns = ['session_id', 'time', 'item_id', 'price', 'quantity']

    sampled_session_id = np.random.choice(click_df.session_id.unique(), 200000, replace=False)
    sampled_click_df = click_df.loc[click_df.session_id.isin(sampled_session_id)]

    item_encoder = LabelEncoder()
    sampled_click_df['item_id'] = item_encoder.fit_transform(sampled_click_df.item_id)

    sampled_buy_df = buy_df.loc[buy_df.session_id.isin(sampled_click_df.session_id)]
    sampled_buy_df['item_id'] = item_encoder.transform(sampled_buy_df.item_id)

    to_pickled_df(PATH_TO_PROCESSED,sampled_clicks=sampled_click_df)
    to_pickled_df(PATH_TO_PROCESSED,sampled_buys=sampled_buy_df)

    sampled_clicks=sampled_click_df.drop(columns=['category'])
    sampled_buys=sampled_buy_df.drop(columns=['price','quantity'])

    sampled_clicks['is_buy']=0
    sampled_buys['is_buy']=1

    sorted_events=pd.concat([sampled_clicks, sampled_buys], ignore_index=True)
    sorted_events['timestamp'] = sorted_events.time.apply(lambda x: int(dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S.%fZ').timestamp()))
    del(sorted_events['time'])
    sorted_events=sorted_events.sort_values(by=['session_id','timestamp'])

    # merge_session.to_csv(os.path.join(PATH_TO_PROCESSED, 'sampled_sessions.csv'), index = None, header=True)
    sorted_events.to_csv(os.path.join(PATH_TO_PROCESSED, 'sorted_events.csv'), index=None, header=True)

    to_pickled_df(PATH_TO_PROCESSED, sorted_events=sorted_events)



    print("total: ", len(sorted_events))
    print("Clicks: ", len(sorted_events[sorted_events.is_buy==0]))
    print("Purchases: ", len(sorted_events[sorted_events.is_buy==1]))
    print(sorted_events.is_buy.unique())

    max_timestamps = sorted_events.groupby('session_id')['timestamp'].max()
    sorted_max_timestamps = max_timestamps.sort_values()
    # print(sorted_max_timestamps.head(10))
    T = sorted_max_timestamps.quantile(0.9)
    sessions_test = max_timestamps[max_timestamps > T].index
    test_sessions = sorted_events[sorted_events['session_id'].isin(sessions_test)]
    train_full = sorted_events[~sorted_events['session_id'].isin(sessions_test)]

    # Remove items seen only in test data, not in training 
    test_sessions = test_sessions[test_sessions.item_id.isin(train_full.item_id.unique())]

    # Encode and Map ItemIds
    from sklearn.preprocessing import LabelEncoder
    item_encoder = LabelEncoder()
    train_full['item_id'] = item_encoder.fit_transform(train_full.item_id)
    test_sessions['item_id'] = item_encoder.transform(test_sessions.item_id)

    session_lengths = test_sessions.groupby('session_id').size()
    test_sessions = test_sessions[np.in1d(test_sessions.session_id, session_lengths[session_lengths>1].index)]


    T_val = sorted_max_timestamps.quantile(0.8)
    sessions_val = max_timestamps[max_timestamps > T_val].index
    val_sessions = train_full[train_full['session_id'].isin(sessions_val)]
    train_sessions = train_full[~train_full['session_id'].isin(sessions_val)]


    to_pickled_df(PATH_TO_PROCESSED,sampled_train_full=train_full)
    to_pickled_df(PATH_TO_PROCESSED, sampled_train=train_sessions)
    to_pickled_df(PATH_TO_PROCESSED, sampled_val=val_sessions)
    to_pickled_df(PATH_TO_PROCESSED,sampled_test=test_sessions)


    ### Rename columns for GRU data format
    columns = {'session_id' : 'SessionId', 'item_id' : 'ItemId', 'timestamp' : 'Time'}

    train_full = rename_columns(train_full, columns)
    train_full.to_csv(os.path.join(PATH_TO_PROCESSED, f'{dataset}_train_full.txt'), sep='\t', index=False)

    train_sessions = rename_columns(train_sessions, columns)
    train_sessions.to_csv(os.path.join(PATH_TO_PROCESSED, f'{dataset}_train_tr.txt'), sep='\t', index=False)

    val_sessions = rename_columns(val_sessions, columns)
    val_sessions.to_csv(os.path.join(PATH_TO_PROCESSED, f'{dataset}_train_valid.txt'), sep='\t', index=False)

    test_sessions = rename_columns(test_sessions, columns)
    test_sessions.to_csv(os.path.join(PATH_TO_PROCESSED, f'{dataset}_test.txt'), sep='\t', index=False)

    print('Full train set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_full), train_full.SessionId.nunique(), train_full.ItemId.nunique()))
    print('Test set\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(test_sessions), test_sessions.SessionId.nunique(), test_sessions.ItemId.nunique()))
    print('Train split\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(train_sessions), train_sessions.SessionId.nunique(), train_sessions.ItemId.nunique()))
    print('Val split\n\tEvents: {}\n\tSessions: {}\n\tItems: {}'.format(len(val_sessions), val_sessions.SessionId.nunique(), val_sessions.ItemId.nunique()))

