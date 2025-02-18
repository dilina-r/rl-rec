# -*- coding: utf-8 -*-
"""
create on 18 Sep, 2019

@author: wangshuo

Reference: https://github.com/lijingsdu/sessionRec_NARM/blob/master/data_process.py
"""

import pickle
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pandas as pd




def load_data(train_file, test_file, item_idmap=None):
    if train_file.endswith('.df'):
        train_data = load_from_df(train_file)
    else:
        train_data, item_idmap = load_from_txt(train_file)
    
    if test_file.endswith('.df'):
        test_data = load_from_df(test_file)
    else:
        test_data, item_idmap = load_from_txt(test_file, itemid_map=item_idmap)
    
    return train_data, test_data


def load_from_txt(filepath, session_key='SessionId', item_key='ItemId', itemid_map=None):
    data = pd.read_csv(filepath, sep='\t')
    groups=data.groupby(session_key)
    ids=data[session_key].unique()

    if itemid_map is None:
        itemids = data[item_key].unique()
        n_items = len(itemids)  # Define n_items
        itemid_map = pd.Series(data=np.arange(n_items, dtype='int32'), index=itemids, name='ItemIdx')
        data['MappedItemId'] = data[item_key].map(itemid_map)
        data[item_key] = data['MappedItemId']
    else:
        data['MappedItemId'] = data[item_key].map(itemid_map)
        data[item_key] = data['MappedItemId']
        
    seqs, labels, events = [], [], []

    for id in ids:
        group=groups.get_group(id)
        history=[]
        count = 0
        for index, row in group.iterrows():
            item_id = int(row['ItemId'])
            is_buy = int(row['is_buy'])
            if count > 0:
                seqs.append(list(history))
                labels.append(item_id)
                events.append(is_buy)
            history.append(item_id)
            count+=1
    
    return (seqs, labels, events), itemid_map


def load_from_df(filepath, session_key='session_id', item_key='item_id', itemid_map=None):
    data = pd.read_pickle(filepath)
    groups=data.groupby(session_key)
    ids=data[session_key].unique()
        
    seqs, labels, events = [], [], []

    for id in ids:
        group=groups.get_group(id)
        history=[]
        count = 0
        for index, row in group.iterrows():
            item_id = int(row[item_key])
            is_buy = int(row['is_buy'])
            if count > 0:
                seqs.append(list(history))
                labels.append(item_id)
                events.append(is_buy)
            history.append(item_id)
            count+=1
    
    return (seqs, labels, events)



# def load_from_files(filepath, test_file):
#     train_data = pd.read_csv(train_file, sep='\t')
#     test_data = pd.read_csv(test_data, sep='\t')
#     # print(data); exit()


#     itemids = train_data['ItemId'].unique()
#     n_items = len(itemids)  # Define n_items
#     itemid_map = pd.Series(data=np.arange(n_items, dtype='int32'), index=itemids, name='ItemIdx')
#     train_data['ItemIdx'] = train_data['ItemId'].map(itemid_map)
#     test_data['ItemIdx'] = test_data['ItemId'].map(itemid_map)
#     # del(train['ItemIds']); del(test['ItemIds'])


#     groups=train_data.groupby('SessionId')
#     ids=train_data.SessionId.unique()

#     seqs, labels, events = [], [], []

#     for id in ids:
#         group=groups.get_group(id)
#         history=[]
#         count = 0
#         for index, row in group.iterrows():
#             item_id = int(row['ItemIdx'])
#             is_buy = int(row['is_buy'])
#             if count > 0:
#                 seqs.append(list(history))
#                 labels.append(item_id)
#                 events.append(is_buy)
#             history.append(item_id)
#             count+=1
    
#     train = (seqs, labels, events)

class RecSysDataset(Dataset):
    """define the pytorch Dataset class for yoochoose and diginetica datasets.
    """
    def __init__(self, data):
        self.data = data
        print('-'*50)
        print('Dataset info:')
        print('Number of sessions: {}'.format(len(data[0])))
        print('-'*50)
        
    def __getitem__(self, index):
        session_items = self.data[0][index]
        target_item = self.data[1][index]
        event_type = self.data[2][index]
        return session_items, target_item, event_type

    def __len__(self):
        return len(self.data[0])
    


def collate_fn(data):
    """This function will be used to pad the sessions to max length
       in the batch and transpose the batch from 
       batch_size x max_seq_len to max_seq_len x batch_size.
       It will return padded vectors, labels and lengths of each session (before padding)
       It will be used in the Dataloader
    """
    data.sort(key=lambda x: len(x[0]), reverse=True)
    lens = [len(sess) for sess, target, event_type in data]
    labels = []
    events = []
    padded_sesss = torch.zeros(len(data), max(lens)).long()
    for i, (sess, label, event) in enumerate(data):
        padded_sesss[i,:lens[i]] = torch.LongTensor(sess)
        labels.append(label)
        events.append(event)
    
    padded_sesss = padded_sesss.transpose(0,1)
    return padded_sesss, torch.tensor(labels).long(), events, lens

