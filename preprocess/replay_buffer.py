import os
import pandas as pd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--path', help='Path to directory containing the pre-processed data files')
args = parser.parse_args()

def to_pickled_df(data_directory, **kwargs):
    for name, df in kwargs.items():
        df.to_pickle(os.path.join(data_directory, name + '.df'))

def pad_history(itemlist,length,pad_item):
    if len(itemlist)>=length:
        return itemlist[-length:]
    if len(itemlist)<length:
        temp = [pad_item] * (length-len(itemlist))
        itemlist.extend(temp)
        return itemlist
    

def get_rb_dict(session_data, length, pad_item):
    groups=session_data.groupby('session_id')
    ids=session_data.session_id.unique()

    state, len_state, action, is_buy, next_state, next_action, len_next_state, is_done = [], [], [], [], [],[],[], []

    for id in ids:
        group=groups.get_group(id)
        history=[]
        enter=False
        for index, row in group.iterrows():
            s=list(history)
            len_state.append(length if len(s)>=length else 1 if len(s)==0 else len(s))
            s=pad_history(s,length,pad_item)
            a=row['item_id']
            is_b=row['is_buy']
            state.append(s)
            action.append(a)
            if enter:
                next_action.append(a)
            is_buy.append(is_b)
            history.append(row['item_id'])
            next_s=list(history)
            len_next_state.append(length if len(next_s)>=length else 1 if len(next_s)==0 else len(next_s))
            next_s=pad_history(next_s,length,pad_item)
            next_state.append(next_s)
            is_done.append(False)
            enter = True
        next_action.append(0)
        is_done[-1]=True

    dic={'state':state,'len_state':len_state,'action':action,'is_buy':is_buy,'next_state':next_state,'len_next_states':len_next_state,
          'next_action': next_action, 'is_done':is_done}

    return dic


if __name__ == '__main__':
    length=10
    data_directory = args.path
    train_full = pd.read_pickle(os.path.join(data_directory, 'sampled_train_full.df'))
    item_ids=train_full.item_id.unique()
    pad_item=len(item_ids)

    ##### Create Replay Buffer for full training data
    dic = get_rb_dict(train_full, length, pad_item)
    replay_buffer_full=pd.DataFrame(data=dic)
    to_pickled_df(data_directory, replay_buffer_full=replay_buffer_full)

    #### Create Replay Buffer for only the train split
    train_tr = pd.read_pickle(os.path.join(data_directory, 'sampled_train.df'))
    dic = get_rb_dict(train_tr, length, pad_item)
    replay_buffer=pd.DataFrame(data=dic)
    to_pickled_df(data_directory, replay_buffer=replay_buffer)

    dic={'state_size':[length],'item_num':[pad_item]}
    data_statis=pd.DataFrame(data=dic)
    to_pickled_df(data_directory,data_statis=data_statis)