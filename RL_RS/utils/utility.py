import os
import numpy as np
import pandas as pd
from collections import deque
import tensorflow as tf

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
    
def calculate_off(sorted_list,true_items,rewards,reward_click,off_click_ng,off_purchase_ng,off_prob_click,off_prob_purchase,pop_dict,topk=10):
    rec_list = sorted_list[:, -topk:]
    for j in range(len(true_items)):
        prob = pop_dict[true_items[j]]
        #off_prob_total[0]+=1.0/prob
        if rewards[j] == reward_click:
            off_prob_click[0]+=1.0/prob
        else:
            off_prob_purchase[0]+=1.0/prob

        if true_items[j] in rec_list[j]:
            # off_prob_total[0] += 1.0 / prob
            rank = topk - np.argwhere(rec_list[j] == true_items[j])
            # recap[0]+=(rewards[j]/ np.log2(rank + 1))/prob

            if rewards[j] == reward_click:
                # off_prob_click[0] += 1.0 / prob
                off_click_ng[0]+=(1.0 / np.log2(rank + 1))/prob
            else:
                # off_prob_purchase[0] += 1.0 / prob
                off_purchase_ng[0]+=(1.0 / np.log2(rank + 1))/prob



def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor.
    :param data: Tensorflow tensor that will be subsetted.
    :param ind: Indices to take (one for each element along axis 0 of data).
    :return: Subsetted tensor.
    """

    batch_range = tf.range(tf.shape(data)[0])
    indices = tf.stack([batch_range, ind], axis=1)
    res = tf.gather_nd(data, indices)

    return res


def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):
    '''Applies layer normalization.

    Args:
      inputs: A tensor with 2 or more dimensions, where the first dimension has
        `batch_size`.
      epsilon: A floating number. A very small number for preventing ZeroDivision Error.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def calculate_hit(sorted_list,topk,true_items,rewards,r_click,total_reward,hit_click,ndcg_click,hit_purchase,ndcg_purchase):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            if true_items[j] in rec_list[j]:
                rank = int(topk[i] - np.squeeze(np.argwhere(rec_list[j] == true_items[j])))
                total_reward[i] += rewards[j]
                if rewards[j] == r_click:
                    hit_click[i] += 1.0
                    ndcg_click[i] += 1.0 / np.log2(rank + 1)
                else:
                    hit_purchase[i] += 1.0
                    ndcg_purchase[i] += 1.0 / np.log2(rank + 1)


def calculate_hit2(sorted_list,topk,true_items,events,hitrate,ndcg,mrr):
    for i in range(len(topk)):
        rec_list = sorted_list[:, -topk[i]:]
        for j in range(len(true_items)):
            event = events[j]
            if true_items[j] in rec_list[j]:
                # rank = topk[i] - np.argwhere(rec_list[j] == true_items[j])
                rank = int(topk[i] - np.squeeze(np.argwhere(rec_list[j] == true_items[j])))
                # print(i, topk[i], event)
                hitrate[i][event] += 1
                ndcg[i][event] += 1.0 / np.log2(rank + 1)
                mrr[i][event] += 1.0 / rank
                # print(i, topk[i], event); print(hitrate);print(mrr);print(ndcg)
    return hitrate, ndcg, mrr


# class Memory():
#     def __init__(self):
#         self.buffer = deque()
#
#     def add(self, experience):
#         self.buffer.append(experience)
#
#     def sample(self, batch_size):
#         idx = np.random.choice(np.arange(len(self.buffer)),
#                                size=batch_size,
#                                replace=False)
#         return [self.buffer[ii] for ii in idx]