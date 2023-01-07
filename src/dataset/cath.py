import torch
import pandas as pd

from .labels import Labels

class CathLabels(Labels):
    def __init__(self, path):
        df = pd.read_csv(path)

        adjecency_matrix = torch.zeros(len(df), dtype=torch.bool)

        dic = {idx: row['number'] for idx, row in df.iterrows()}

        for idx, row in df.iterrows():
            parent = row['number'].split('.')[:-1].join('.')

            if parent != '':
                adjecency_matrix[idx, dic[parent]] = True
                adjecency_matrix[dic[parent], idx] = True

        super().__init__(adjecency_matrix)