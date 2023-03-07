import pandas as pd
import numpy as np
import torch

from .datasets import LabelDataset

class CathLabelDataset(LabelDataset):
    def __init__(self, path=None, df=None, no_disconnected_per_connected=10):
        if df is None:
            df = pd.read_csv(path, sep=' ')

        # TODO: sparse matrix
        adjecency_matrix = np.zeros((len(df), len(df)), dtype=bool)

        dic = {row['number'] : idx for idx, row in df.iterrows()}
        
        assignable_labels = []
        real_to_assignable = torch.full((len(df),), -1, dtype=int)

        for idx, row in df.iterrows():
            splitted_num = row['number'].split('.')
            
            parent = '.'.join(splitted_num[:-1])
            
            if len(splitted_num) == 4:
                real_to_assignable[idx] = len(assignable_labels)
                assignable_labels.append(idx)

            if parent != '':
                adjecency_matrix[idx, dic[parent]] = True
                adjecency_matrix[dic[parent], idx] = True
                
        assignable_labels = torch.tensor(assignable_labels)

        super().__init__(adjecency_matrix, assignable_labels, real_to_assignable, no_disconnected_per_connected)