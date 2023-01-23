import pandas as pd
import numpy as np

from .datasets import LabelDataset

class CathLabelDataset(LabelDataset):
    def __init__(self, path, no_disconnected_per_connected):
        df = pd.read_csv(path, sep=' ')

        # TODO: sparse matrix
        adjecency_matrix = np.zeros((len(df), len(df)), dtype=bool)

        dic = {row['number'] : idx for idx, row in df.iterrows()}

        for idx, row in df.iterrows():
            parent = '.'.join(row['number'].split('.')[:-1])

            if parent != '':
                adjecency_matrix[idx, dic[parent]] = True
                adjecency_matrix[dic[parent], idx] = True

        super().__init__(adjecency_matrix, no_disconnected_per_connected)