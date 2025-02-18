import os
import numpy as np
import pandas as pd


class ResultHandler:

    def __init__(self, topk, event_types):
        self.topk = topk
        self.event_types = event_types

        self.reset()

    def reset(self):
        self.metrics = {}
        for e in self.event_types:
            self.metrics[e] = {}
        for key, value in self.metrics.items():
            self.metrics[key]['count'] = 0
            for k in self.topk:
                self.metrics[key][f'MRR@{k}'] = 0
                self.metrics[key][f'HR@{k}'] = 0
                self.metrics[key][f'NDCG@{k}'] = 0

    def update(self, sorted_list,actions,events):
        for j in range(0, len(actions)):
            event = self.event_types[events[j]]
            next_item = actions[j]
            self.metrics[event]['count'] += 1
            for i in range(len(self.topk)):
                k = self.topk[i]
                rec_list = sorted_list[j, -self.topk[i]:]
                if actions[j] in rec_list:
                    rank = int(np.squeeze(self.topk[i] - np.argwhere(rec_list == next_item)))
                    self.metrics[event][f'HR@{k}'] += 1.0
                    self.metrics[event][f'NDCG@{k}'] += 1.0 / np.log2(rank + 1)
                    self.metrics[event][f'MRR@{k}'] += 1.0 / rank

    def get_metrics(self, event, metric, k):
        return self.metrics[event][f'{metric}@{k}'] / self.metrics[event]['count']
    
    def get_total(self, k=20):
        hr, mrr, ng = 0.0, 0.0, 0.0
        count = 0.0
        for event in self.event_types:
            hr += self.metrics[event][f'HR@{k}']
            mrr += self.metrics[event][f'MRR@{k}']
            ng += self.metrics[event][f'NDCG@{k}']
            count+= self.metrics[event]['count']
        return hr/count, mrr/count, ng/count, count

    def get_total_metric(self, metric):
        m = 0.0
        count = 0.0
        for event in self.event_types:
            m += self.metrics[event][metric]
            count+= self.metrics[event]['count']
        return m/count
    
    def print_summary(self):

        for event in self.event_types:
            count = self.metrics[event]['count']
            if count > 0:
                print('='*50)
                print(f"Event: {event} --- Count: {count}")
                print('-'*50)
                for k in self.topk:
                    print('Hitrate@{}: {}, MRR@{}: {}, NDCG@{}: {}'.format(
                        k, np.round(self.metrics[event][f'HR@{k}']/count, 4),
                        k, np.round(self.metrics[event][f'MRR@{k}']/count, 4),
                        k, np.round(self.metrics[event][f'NDCG@{k}']/count, 4)
                    ))
        print('='*50)
        _, _, _, count = self.get_total(self.topk[-1])
        print(f"Total Events --- Count: {count}")
        for k in self.topk:
            hr, mrr, ng, count = self.get_total(k)
            print('Hitrate@{}: {}, MRR@{}: {}, NDCG@{}: {}'.format(
                       k, hr, k, mrr, k, ng
                    ))
        print('='*50)


                




                    
