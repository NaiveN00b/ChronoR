from pathlib import Path
import pkg_resources
import pickle
from collections import defaultdict
from typing import Dict, Tuple, List

from sklearn.metrics import average_precision_score

import numpy as np
import torch
from models import TKBCModel


DATA_PATH = 'data/'

class TemporalDataset(object):
    def __init__(self, name: str, device='cpu'):
        self.device = device
        self.root = Path(DATA_PATH) / name

        self.data = {}
        for f in ['train', 'test', 'valid']:
            in_file = open(str(self.root / (f + '.pickle')), 'rb')
            self.data[f] = pickle.load(in_file)
        # array([7127,  229, 7127,  364], dtype=uint64)
        maxis = np.max(self.data['train'], axis=0)
        self.n_entities = int(max(maxis[0], maxis[2]) + 1)
        self.n_predicates = int(maxis[1] + 1)
        # 因为有r^-1存在,本数据集中r^-1通过rel_id + n_rel实现
        # 上述操作仅在filtering数据做了，没在train.pickle上做
        self.n_predicates *= 2

        if self.data['valid'].shape[1]>4:# 即时间为时间段
            self.interval = True 
            f = open(str(self.root / 'ts_id.pickle'), 'rb')
            self.time_dict = pickle.load(f)
        else:
            self.interval = False
            
        if maxis.shape[0] > 4:
            self.n_timestamps = max(int(maxis[3] + 1), int(maxis[4] + 1))
        else:
            self.n_timestamps = int(maxis[3] + 1)
        # 对于时间段类型的数据集以及事件类型处理，暂时不用看，来自于TNTComplEx
        try:
            inp_f = open(str(self.root / f'ts_diffs.pickle'), 'rb')
            self.time_diffs = torch.from_numpy(pickle.load(inp_f)).to(self.device).float()
         
            inp_f.close()
        except OSError:
            print("Assume all timestamps are regularly spaced")
            self.time_diffs = None

        try:
            e = open(str(self.root / f'event_list_all.pickle'), 'rb')
            self.events = pickle.load(e)
            e.close()

            f = open(str(self.root / f'ts_id'), 'rb')
            dictionary = pickle.load(f)
            f.close()
            self.timestamps = sorted(dictionary.keys())
        except OSError:
            print("Not using time intervals and events eval")
            self.events = None
        # ---------------------------------------------------------------------------------
        # self.to_skip类似于head-batch和tail-batch
        if self.events is None:
            inp_f = open(str(self.root / f'to_skip.pickle'), 'rb')
            self.to_skip: Dict[str, Dict[Tuple[int, int, int], List[int]]] = pickle.load(inp_f)
            inp_f.close()



    def get_examples(self, split):
        return self.data[split]

    def get_train(self):
        # 将(s,r,o,t)和(o,r^-1,s,t)一起得到
        copy = np.copy(self.data['train'])
        tmp = np.copy(copy[:, 0])
        copy[:, 0] = copy[:, 2]
        copy[:, 2] = tmp
        copy[:, 1] += self.n_predicates // 2  
        return np.vstack((self.data['train'], copy))

    def eval(
            self, model: TKBCModel, split: str, n_queries: int = -1, missing_eval: str = 'both',
            at: Tuple[int] = (1, 3, 10)
    ):
        if self.events is not None:
            return self.time_eval(model, split, n_queries, 'rhs', at)
        test = self.get_examples(split)

        missing = [missing_eval]
        if missing_eval == 'both':
            missing = ['rhs', 'lhs']

        mean_reciprocal_rank = {}
        hits_at = {}
        
        
        if self.interval: 
            examples = test
            for m in missing:
                q = np.copy(examples)
                if m == 'lhs':
                    tmp = np.copy(q[:, 0])
                    q[:, 0] = q[:, 2]
                    q[:, 2] = tmp
                    q[:, 1] = q[:, 1].astype('uint64')+self.n_predicates // 2
                ranks = model.get_ranking(q, self.to_skip[m], batch_size=500,
                                          year2id=self.time_dict, device=self.device)
                mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
                hits_at[m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((ranks <= x).float()).item(),
                    at
                ))))
        else:
            examples = torch.from_numpy(test.astype('int64')).to(self.device)
            for m in missing:
                q = examples.clone()
                if n_queries > 0:
                    permutation = torch.randperm(len(examples))[:n_queries]
                    q = examples[permutation]
                if m == 'lhs':
                    tmp = torch.clone(q[:, 0])
                    q[:, 0] = q[:, 2]
                    q[:, 2] = tmp
                    q[:, 1] += self.n_predicates // 2
                ranks = model.get_ranking(q, self.to_skip[m], batch_size=500,device=self.device)
                mean_reciprocal_rank[m] = torch.mean(1. / ranks).item()
                hits_at[m] = torch.FloatTensor((list(map(
                    lambda x: torch.mean((ranks <= x).float()).item(),
                    at
                ))))

        return mean_reciprocal_rank, hits_at



    def get_shape(self):
        return self.n_entities, self.n_predicates, self.n_entities, self.n_timestamps
