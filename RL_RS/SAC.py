import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from collections import deque
from utils.utility import pad_history,calculate_hit
from sklearn.decomposition import PCA
import pickle as pkl
from utils.NextItNetModules import *
from utils.SASRecModules import *

import trfl
from trfl import indexing_ops

def parse_args():
    parser = argparse.ArgumentParser(description="Run double q learning.")

    parser.add_argument('--epoch', type=int, default=50,
                        help='Number of max epochs.')
    parser.add_argument('--data', nargs='?', default='/mnt/data/RetailRocket/data',
                        help='data directory')
    # parser.add_argument('--pretrain', type=int, default=1,
    #                     help='flag for pretrain. 1: initialize from pretrain; 0: randomly initialize; -1: save the model to pretrain file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--r_click', type=float, default=0.2,
                        help='reward for the click behavior.')
    parser.add_argument('--r_buy', type=float, default=1.0,
                        help='reward for the purchase behavior.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--discount', type=float, default=0.5,
                        help='Discount factor for RL.')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='Weight value for loss calculation.')
    parser.add_argument('--model', type=str, default=None,
                        help='the base recommendation models, including GRU,Caser,NItNet and SASRec')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs the model will be run in each recommendation model if model is not given.')
    parser.add_argument('--num_filters', type=int, default=16,
                        help='Number of filters per filter size (default: 16) (for Caser)')
    parser.add_argument('--filter_sizes', nargs='?', default='[2,3,4]',
                        help='Specify the filter_size (for Caser)')
    parser.add_argument('--num_heads', default=1, type=int,help='number heads (for SASRec)')
    parser.add_argument('--num_blocks', default=1, type=int, help='number heads (for SASRec)')
    parser.add_argument('--dropout_rate', default=0.1, type=float)
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--device', type=int, default=-1)


    return parser.parse_args()


class QNetwork:
    def __init__(self, hidden_size, learning_rate, item_num, state_size, pretrain, model='GRU', name='DQNetwork'):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size = hidden_size
        self.item_num = int(item_num)
        self.pretrain = pretrain
        self.weight=args.weight
        self.model=model
        self.is_training = tf.placeholder(tf.bool, shape=())
        # self.save_file = save_file
        self.name = name
        with tf.variable_scope(self.name):
            self.all_embeddings=self.initialize_embeddings()
            self.inputs = tf.placeholder(tf.int32, [None, state_size])  # sequence of history, [batchsize,state_size]
            self.len_state = tf.placeholder(tf.int32, [
                None])  # the length of valid positions, because short sesssions need to be padded

            self.input_emb = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'], self.inputs)

            if self.model=='GRU':
                gru_out, self.states_hidden = tf.nn.dynamic_rnn(
                    tf.contrib.rnn.GRUCell(self.hidden_size),
                    self.input_emb,
                    dtype=tf.float32,
                    sequence_length=self.len_state,
                )

            if self.model=='Caser':
                mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

                self.input_emb *= mask
                self.embedded_chars_expanded = tf.expand_dims(self.input_emb, -1)

                # Create a convolution + maxpool layer for each filter size
                pooled_outputs = []
                num_filters = args.num_filters
                filter_sizes = eval(args.filter_sizes)
                for i, filter_size in enumerate(filter_sizes):
                    with tf.name_scope("conv-maxpool-%s" % filter_size):
                        # Convolution Layer
                        filter_shape = [filter_size, self.hidden_size, 1, num_filters]
                        W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                        b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")

                        conv = tf.nn.conv2d(
                            self.embedded_chars_expanded,
                            W,
                            strides=[1, 1, 1, 1],
                            padding="VALID",
                            name="conv")
                        # Apply nonlinearity
                        h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                        # Maxpooling over the outputs
                        # new shape after max_pool[?, 1, 1, num_filters]
                        # be carefyul, the  new_sequence_length has changed because of wholesession[:, 0:-1]
                        pooled = tf.nn.max_pool(
                            h,
                            ksize=[1, state_size - filter_size + 1, 1, 1],
                            strides=[1, 1, 1, 1],
                            padding='VALID',
                            name="pool")
                        pooled_outputs.append(pooled)

                # Combine all the pooled features
                num_filters_total = num_filters * len(filter_sizes)
                self.h_pool = tf.concat(pooled_outputs, 3)
                self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])  # shape=[batch_size, 384]
                # design the veritcal cnn
                with tf.name_scope("conv-verical"):
                    filter_shape = [self.state_size, 1, 1, 1]
                    W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                    b = tf.Variable(tf.constant(0.1, shape=[1]), name="b")
                    conv = tf.nn.conv2d(
                        self.embedded_chars_expanded,
                        W,
                        strides=[1, 1, 1, 1],
                        padding="VALID",
                        name="conv")
                    h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
                self.vcnn_flat = tf.reshape(h, [-1, self.hidden_size])
                self.final = tf.concat([self.h_pool_flat, self.vcnn_flat], 1)  # shape=[batch_size, 384+100]

                # Add dropout
                with tf.name_scope("dropout"):
                    self.states_hidden = tf.layers.dropout(self.final,
                                                          rate=args.dropout_rate,
                                                          training=tf.convert_to_tensor(self.is_training))
            if self.model=='NItNet':

                mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)

                self.model_para = {
                    'dilated_channels': 64,  # larger is better until 512 or 1024
                    'dilations': [1, 2, 1, 2, 1, 2, ],  # YOU should tune this hyper-parameter, refer to the paper.
                    'kernel_size': 3,
                }

                context_embedding = tf.nn.embedding_lookup(self.all_embeddings['state_embeddings'],
                                                           self.inputs)
                context_embedding *= mask

                dilate_output = context_embedding
                for layer_id, dilation in enumerate(self.model_para['dilations']):
                    dilate_output = nextitnet_residual_block(dilate_output, dilation,
                                                             layer_id, self.model_para['dilated_channels'],
                                                             self.model_para['kernel_size'], causal=True,
                                                             train=self.is_training)
                    dilate_output *= mask

                self.states_hidden = extract_axis_1(dilate_output, self.len_state - 1)

            if self.model=='SASRec':
                pos_emb = tf.nn.embedding_lookup(self.all_embeddings['pos_embeddings'],
                                                 tf.tile(tf.expand_dims(tf.range(tf.shape(self.inputs)[1]), 0),
                                                         [tf.shape(self.inputs)[0], 1]))
                self.seq = self.input_emb + pos_emb

                mask = tf.expand_dims(tf.to_float(tf.not_equal(self.inputs, item_num)), -1)
                # Dropout
                self.seq = tf.layers.dropout(self.seq,
                                             rate=args.dropout_rate,
                                             training=tf.convert_to_tensor(self.is_training))
                self.seq *= mask

                # Build blocks

                for i in range(args.num_blocks):
                    with tf.variable_scope("num_blocks_%d" % i):
                        # Self-attention
                        self.seq = multihead_attention(queries=normalize(self.seq),
                                                       keys=self.seq,
                                                       num_units=self.hidden_size,
                                                       num_heads=args.num_heads,
                                                       dropout_rate=args.dropout_rate,
                                                       is_training=self.is_training,
                                                       causality=True,
                                                       scope="self_attention")

                        # Feed forward
                        self.seq = feedforward(normalize(self.seq), num_units=[self.hidden_size, self.hidden_size],
                                               dropout_rate=args.dropout_rate,
                                               is_training=self.is_training)
                        self.seq *= mask

                self.seq = normalize(self.seq)
                self.states_hidden = extract_axis_1(self.seq, self.len_state - 1)

            self.output1 = tf.contrib.layers.fully_connected(self.states_hidden, self.item_num,
                                                            activation_fn=None)  # all q-values

            self.output2= tf.contrib.layers.fully_connected(self.states_hidden, self.item_num,
                                                             activation_fn=None, scope="ce-logits")  # all ce logits

            # TRFL way
            self.actions = tf.placeholder(tf.int32, [None])


            self.targetQs_ = tf.placeholder(tf.float32, [None, item_num])
            self.targetQs_selector = tf.placeholder(tf.float32, [None,
                                                                 item_num])  # used for select best action for double q learning
            self.reward = tf.placeholder(tf.float32, [None])
            self.discount = tf.placeholder(tf.float32, [None])

            # TRFL double qlearning
            qloss, q_learning = trfl.double_qlearning(self.output1, self.actions, self.reward, self.discount,
                                                      self.targetQs_, self.targetQs_selector)
            self.qa = tf.stop_gradient(indexing_ops.batched_index(self.output1, self.actions))

            celoss1 = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions, logits=self.output2)

            celoss2 = tf.multiply(self.qa, tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.actions,
                                                                                            logits=self.output2))
            self.loss1 = tf.reduce_mean(celoss1 + qloss)
            self.loss2 = tf.reduce_mean(celoss2 + qloss)
            self.opt1 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss1)
            self.opt2 = tf.train.AdamOptimizer(learning_rate).minimize(self.loss2)

    def initialize_embeddings(self):
        all_embeddings = dict()
        if self.pretrain == False:
            with tf.variable_scope(self.name):
                state_embeddings = tf.Variable(tf.random_normal([self.item_num + 1, self.hidden_size], 0.0, 0.01),
                                           name='state_embeddings')
                pos_embeddings = tf.Variable(tf.random_normal([self.state_size, self.hidden_size], 0.0, 0.01),
                                             name='pos_embeddings')
                all_embeddings['state_embeddings'] = state_embeddings
                all_embeddings['pos_embeddings'] = pos_embeddings
        # else:
        #     weight_saver = tf.train.import_meta_graph(self.save_file + '.meta')
        #     pretrain_graph = tf.get_default_graph()
        #     state_embeddings = pretrain_graph.get_tensor_by_name('state_embeddings:0')
        #     with tf.Session() as sess:
        #         weight_saver.restore(sess, self.save_file)
        #         se = sess.run([state_embeddings])[0]
        #     with tf.variable_scope(self.name):
        #         all_embeddings['state_embeddings'] = tf.Variable(se, dtype=tf.float32)
        #     print("load!")
        return all_embeddings

def evaluate(sess,topk, test=False, model='GRU'):
    if not test:
        eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_val.df'))
    else:
        eval_sessions=pd.read_pickle(os.path.join(data_directory, 'sampled_test.df'))
    event_types = ['click', 'cart', 'purchase']
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated=0
    
    from metrics.results_handler import ResultHandler
    res_handler = ResultHandler(topk=topk, event_types=event_types)

    while evaluated<len(eval_ids):
        states, len_states, actions, events = [], [], [], []
        for i in range(batch):
            if evaluated==len(eval_ids):
                break
            id=eval_ids[evaluated]
            group=groups.get_group(id)
            history=[]
            count = 0
            for index, row in group.iterrows():
                if count > 0:
                    state=list(history)
                    len_states.append(state_size if len(state)>=state_size else 1 if len(state)==0 else len(state))
                    state=pad_history(state,state_size,item_num)
                    states.append(state)
                    action=row['item_id']
                    is_buy=int(row['is_buy'])
                    actions.append(action)
                    events.append(is_buy)

                history.append(row['item_id'])
                count = count + 1
            evaluated+=1
            if evaluated > len(eval_ids):
                break
        prediction=sess.run(QN_1.output2, feed_dict={QN_1.inputs: states,QN_1.len_state:len_states,QN_1.is_training:False,QN_1.actions:actions})
        sorted_list=np.argsort(np.squeeze(prediction))
        res_handler.update(sorted_list, actions, events)
   
    if test:
        res_handler.print_summary()

    hr, mrr, ndcg, _ = res_handler.get_total(k=20)

    return hr, mrr, ndcg





if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    if args.device > -1:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    data_directory = args.data
    data_statis = pd.read_pickle(
        os.path.join(data_directory, 'data_statis.df'))  # read data statistics, includeing state_size and item_num
    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items
    reward_click = args.r_click
    reward_buy = args.r_buy
    # topk=[5,10,15,20]
    # save_file = 'pretrain-GRU/%d' % (hidden_size)
    if args.model:
        mods = [args.model]
    else:
        mods = ['GRU', 'Caser', 'NItNet', 'SASRec'] * args.runs
    # save_file = 'pretrain-GRU/%d' % (hidden_size)
    event_rewards = [1.0, 4.0, 5.0]
    for mod in mods:

        tf.reset_default_graph()

        QN_1 = QNetwork(name='QN_1', hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                        state_size=state_size, model=mod, pretrain=False)
        QN_2 = QNetwork(name='QN_2', hidden_size=args.hidden_factor, learning_rate=args.lr, item_num=item_num,
                        state_size=state_size, model=mod, pretrain=False)

        replay_buffer = pd.read_pickle(os.path.join(data_directory, 'replay_buffer.df'))
        saver = tf.train.Saver()
        # save_dir = "trained_models/RetailRocket/SQN"
        # if not(os.path.exists(save_dir)):
        #     os.makedirs(save_dir)
        # save_file = os.path.join(save_dir, f"{mod}_full_all_embd{args.hidden_factor}.pkt")
        save_file = args.save_file
        print(f'Model save path: {save_file}')

        total_step=0
        best_mrr = 0.0
        losses = []
        early_stopping = 0
        with tf.Session() as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())
            # evaluate(sess)
            num_rows=replay_buffer.shape[0]
            num_batches=int(num_rows/args.batch_size)
            print(f'Num training batches: {num_batches}')

            for i in range(args.epoch):
                for j in range(num_batches):
                    batch = replay_buffer.sample(n=args.batch_size).to_dict()

                    next_state = list(batch['next_state'].values())
                    len_next_state = list(batch['len_next_states'].values())
                    # double q learning, pointer is for selecting which network  is target and which is main
                    pointer = np.random.randint(0, 2)
                    if pointer == 0:
                        mainQN = QN_1
                        target_QN = QN_2
                    else:
                        mainQN = QN_2
                        target_QN = QN_1
                    target_Qs = sess.run(target_QN.output1,
                                        feed_dict={target_QN.inputs: next_state,
                                                    target_QN.len_state: len_next_state,
                                                    target_QN.is_training:True})
                    target_Qs_selector = sess.run(mainQN.output1,
                                                feed_dict={mainQN.inputs: next_state,
                                                            mainQN.len_state: len_next_state,
                                                            mainQN.is_training:True})
                    # Set target_Qs to 0 for states where episode ends
                    is_done = list(batch['is_done'].values())
                    for index in range(target_Qs.shape[0]):
                        if is_done[index]:
                            target_Qs[index] = np.zeros([item_num])

                    state = list(batch['state'].values())
                    len_state = list(batch['len_state'].values())
                    target_Q_current = sess.run(target_QN.output1,
                                                feed_dict={target_QN.inputs: state,
                                                        target_QN.len_state: len_state,
                                                        target_QN.is_training:True})
                    target_Q__current_selector = sess.run(mainQN.output1,
                                                        feed_dict={mainQN.inputs: state,
                                                                    mainQN.len_state: len_state,
                                                                    mainQN.is_training:True})
                    action = list(batch['action'].values())

                    is_buy=list(batch['is_buy'].values())
                    reward=[]
                    for k in range(len(is_buy)):
                        reward.append(reward_buy if is_buy[k] == 1 else reward_click)
                    discount = [args.discount] * len(action)


                    if total_step < 15000:

                        loss, _ = sess.run([mainQN.loss1, mainQN.opt1],
                                        feed_dict={mainQN.inputs: state,
                                                    mainQN.len_state: len_state,
                                                    mainQN.targetQs_: target_Qs,
                                                    mainQN.reward: reward,
                                                    mainQN.discount: discount,
                                                    mainQN.actions: action,
                                                    mainQN.targetQs_selector: target_Qs_selector,
                                                    mainQN.is_training:True
                                                    })
                        total_step += 1
                        # if total_step % 200 == 0:
                        #     print("the loss in %dth batch is: %f" % (total_step, loss))
                        # if total_step % 4000 == 0:
                        #     evaluate(sess)
                    else:


                        loss, _ = sess.run([mainQN.loss2, mainQN.opt2],
                                        feed_dict={mainQN.inputs: state,
                                                    mainQN.len_state: len_state,
                                                    mainQN.targetQs_: target_Qs,
                                                    mainQN.reward: reward,
                                                    mainQN.discount: discount,
                                                    mainQN.actions: action,
                                                    mainQN.targetQs_selector: target_Qs_selector,
                                                    mainQN.is_training:True
                                                    })
                        total_step += 1
                    losses.append(loss)
                    # if total_step % 100 == 0:
                    #     print("the loss in %d/%d batch is: %f" % (total_step, num_batches*args.epoch, loss))
                    # if total_step % 1000 == 0:
                    #     hr, mrr, ndcg = evaluate(sess, topk=[20])
                print("the avg loss in %d/%d batch is: %f" % (total_step, num_batches*args.epoch, np.mean(losses)))
                losses = []
                hr, mrr, ndcg = evaluate(sess, topk=[20])
                early_stopping+=1
                if mrr > best_mrr:
                    best_mrr = mrr
                    saver.save(sess, save_file)
                    early_stopping=0
                # hr, mrr, ndcg = evaluate(sess, topk=[20])
                # print(f'Epoch {i} iteration {total_step}: hr@20:{np.round(hr, 4)}, mrr@20:{np.round(mrr, 4)}, ndcg@20:{np.round(ndcg, 4)}')
                print('Epoch {} validation: Recall@20: {:.4f}, MRR@20: {:.4f}, Best MRR@20: {:.4f}'.format(i, hr, mrr, best_mrr))

            print('*'*50)
            print("Evaluating on Test Data...", mod, args.discount)
            evaluate(sess, topk=[5,10,20], model=mod, test=True)
            saver.restore(sess, save_file)
            evaluate(sess, topk=[5,10,20], model=mod, test=True)