import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf
import numpy as np
import pandas as pd
import argparse
from collections import deque
from utils.utility import pad_history
import pickle as pkl
from utils.NextItNetModules import *
from utils.SASRecModules import *
from utils.results_handler import ResultHandler
import trfl

def parse_args():
    parser = argparse.ArgumentParser(description="Run double q learning.")

    parser.add_argument('--epoch', type=int, default=30,
                        help='Number of max epochs.')
    parser.add_argument('--replay_buffer', nargs='?',
                        help='path to replay_buffer file for training sequences')
    parser.add_argument('--test_file', nargs='?',
                        help='path to file containing the sessions for testing/validation')
    parser.add_argument('--stats_file', nargs='?',
                        help='path to data_statis.df file')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size.')
    parser.add_argument('--hidden_factor', type=int, default=64,
                        help='Number of hidden factors, i.e., embedding size.')
    parser.add_argument('--lr', type=float, default=0.005,
                        help='Learning rate.')
    parser.add_argument('--weight', type=float, default=1.0,
                        help='Weight value for loss calculation.')
    parser.add_argument('--model', type=str, default=None,
                        help='the base recommendation models, including GRU,Caser,NItNet and SASRec')
    parser.add_argument('--runs', type=int, default=1,
                        help='Number of runs the model will be run in each recommendation model if model is not given.')
    parser.add_argument('--save_file', type=str, default=None)
    parser.add_argument('--device', type=int, default=-1)

    return parser.parse_args()



class GRUnetwork:
    def __init__(self, hidden_size,learning_rate,item_num,state_size):
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.hidden_size=hidden_size
        self.item_num=int(item_num)

        all_embeddings=self.initialize_embeddings()

        self.inputs = tf.placeholder(tf.int32, [None, state_size],name='inputs')
        self.len_state=tf.placeholder(tf.int32, [None],name='len_state')
        self.target= tf.placeholder(tf.int32, [None],name='target') # target item, to calculate ce loss

        self.input_emb=tf.nn.embedding_lookup(all_embeddings['state_embeddings'],self.inputs)

        gru_out, self.states_hidden= tf.nn.dynamic_rnn(
            tf.contrib.rnn.GRUCell(self.hidden_size),
            self.input_emb,
            dtype=tf.float32,
            sequence_length=self.len_state,
        )

        self.output = tf.contrib.layers.fully_connected(self.states_hidden,self.item_num,activation_fn=None,scope='fc')

        self.loss=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target,logits=self.output)
        self.loss = tf.reduce_mean(self.loss)
        self.opt = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def initialize_embeddings(self):
        all_embeddings = dict()
        state_embeddings= tf.Variable(tf.random_normal([self.item_num+1, self.hidden_size], 0.0, 0.01),
            name='state_embeddings')
        all_embeddings['state_embeddings']=state_embeddings
        return all_embeddings



def evaluate(sess,eval_sessions, topk=[20]):
    
    event_types = ['click', 'purchase']
    eval_ids = eval_sessions.session_id.unique()
    groups = eval_sessions.groupby('session_id')
    batch = 100
    evaluated=0
    
    results = ResultHandler(topk=topk, event_types=event_types)

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
        prediction=sess.run(GRUnet.output, feed_dict={GRUnet.inputs: states,GRUnet.len_state:len_states})
        sorted_list=np.argsort(np.squeeze(prediction))
        results.update(sorted_list, actions, events)

    return results





if __name__ == '__main__':
    # Network parameters
    args = parse_args()
    if args.device > -1:
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.device)

    data_statis = pd.read_pickle(args.stats_file)  # read data statistics, includeing state_size and item_num
    replay_buffer = pd.read_pickle(args.replay_buffer)
    test_sessions = pd.read_pickle(args.test_file)

    event_rewards = [1.0, 5.0] # r_click=1.0, r_buy=5.0

    state_size = data_statis['state_size'][0]  # the length of history to define the state
    item_num = data_statis['item_num'][0]  # total number of items

    tf.reset_default_graph()
    GRUnet = GRUnetwork(hidden_size=args.hidden_factor, learning_rate=args.lr,item_num=item_num,state_size=state_size)

    saver = tf.train.Saver()
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

                state = list(batch['state'].values())
                len_state = list(batch['len_state'].values())
                target=list(batch['action'].values())
                loss, _ = sess.run([GRUnet.loss, GRUnet.opt],
                                   feed_dict={GRUnet.inputs: state,
                                              GRUnet.len_state: len_state,
                                              GRUnet.target: target})
                total_step+=1
                losses.append(loss)
            print(f"Epoch {i+1}/{args.epoch} -- Loss: {np.mean(losses)}")
            losses = []

        # Save trained model
        saver.save(sess, save_file)
        print('*'*50)
        print("Evaluating on Test Data...")
        results = evaluate(sess, test_sessions, topk=[5, 10, 20])
        results.print_summary()