#!/usr/bin/env python37
# -*- coding: utf-8 -*-
"""
Created on 19 Sep, 2019

@author: wangshuo
"""


import os
import time
import random
import argparse
import pickle
import numpy as np
from tqdm import tqdm
from os.path import join

import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
from torch.backends import cudnn

from narm import NARM
from dataset import *
from results_handler import ResultHandler

parser = argparse.ArgumentParser()
parser.add_argument('--train_file', help='Train file path')
parser.add_argument('--test_file', default=None, help='Test or Validation file')
parser.add_argument('--n_items', type=int, default=8174, help='number of items for each dataset')
parser.add_argument('--batch_size', type=int, default=512, help='input batch size')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden state size of gru module')
parser.add_argument('--embed_dim', type=int, default=50, help='the dimension of item embedding')
parser.add_argument('--epoch', type=int, default=100, help='the number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  
parser.add_argument('--lr_dc', type=float, default=0.1, help='learning rate decay rate')
parser.add_argument('--lr_dc_step', type=int, default=80, help='the number of steps after which the learning rate decay') 
parser.add_argument('--test', action='store_true', help='test')
parser.add_argument('--topk', type=int, default=20, help='number of top score items selected for calculating recall and mrr metrics')
parser.add_argument('--valid_portion', type=float, default=0.1, help='split the portion of training set as validation set')
parser.add_argument('--gpu_id', type=int, default=0, help='GPU ID for cuda')
parser.add_argument('--model_path', type=str,default=None, help='Model save path')


args = parser.parse_args()
print(args)
os.environ["CUDA_VISIBLE_DEVICES"]=f"{args.gpu_id}"

here = os.path.dirname(os.path.abspath(__file__))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

event_types =['clicks', 'purchases']

def main():    

    train_data, test_data  = load_data(args.train_file, args.test_file)
    train_loader = DataLoader(RecSysDataset(train_data), batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)
    test_loader = DataLoader(RecSysDataset(test_data), batch_size = args.batch_size, shuffle = True, collate_fn = collate_fn)

    n_items = args.n_items   
    model = NARM(n_items, args.hidden_size, args.embed_dim, args.batch_size).to(device)

    if args.test:
        print("EVALUTING THE MODEL ONLY!!!")
        ckpt = torch.load(args.model_path)
        model.load_state_dict(ckpt['state_dict'])
        evaluate(test_loader, model)
        return

    optimizer = optim.Adam(model.parameters(), args.lr)
    criterion = nn.CrossEntropyLoss()
    scheduler = StepLR(optimizer, step_size = args.lr_dc_step, gamma = args.lr_dc)

    for epoch in range(args.epoch):
        scheduler.step(epoch = epoch)
        loss = trainForEpoch(train_loader, model, optimizer, criterion)
        print(f"Epoch {epoch+1}/{args.epoch} -- Loss: {loss}")

    
    ckpt_dict = {'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict()
                }
    torch.save(ckpt_dict, args.model_path)
    
    results = evaluate(test_loader, model, topk=[5,10,20])
    results.print_summary()



def trainForEpoch(train_loader, model, optimizer, criterion):
    model.train()

    sum_epoch_loss = 0
    
    losses = []
    for i, (seq, target, events, lens) in tqdm(enumerate(train_loader), total=len(train_loader)):
        seq = seq.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        outputs = model(seq, lens)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step() 
        losses.append(loss.cpu().item())
    
    return np.mean(losses)


def evaluate(data_loader, model, topk=[5,10,20]):
    model.eval()
    results = ResultHandler(topk=topk, event_types=event_types)
    with torch.no_grad():
        for seq, target, events, lens in data_loader:
            seq = seq.to(device)
            outputs = model(seq, lens)
            logits = F.softmax(outputs, dim = 1)
            sorted_list = np.argsort(np.squeeze(logits.cpu().numpy()))
            results.update(sorted_list=sorted_list, actions=target.numpy(), events=events)
    return results

if __name__ == '__main__':
    main()
