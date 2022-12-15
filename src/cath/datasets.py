import pandas as pd
import numpy as np
import torch
import geoopt
from torch.utils.data import Dataset

class LabelDataset():
    def __init__(self, ball, dim, path_to_labels, smart_init=False):
        self.dim = int(dim)
        self.ball = ball
        
        self.init_labels(path_to_labels)
        self.init_embeddings(smart_init)


    def __len__(self):
        return len(self.labels) - 1


    def get_connected(self, idx):
        return self.labels.connected.iloc[idx]

    
    def get_disconnected(self, idx):
        return self.labels.disconnected.iloc[idx]


    def init_labels(self, path_to_labels):
        self.labels = pd.read_pickle(path_to_labels)
        self.labels['number_len'] = self.labels.number.str.split('.').apply(len)

        root_connected = set(self.labels.index[self.labels.number_len == 1])

        self.labels = pd.concat([pd.DataFrame([['r', '', 'root', 0, root_connected, 0]], columns=self.labels.columns), self.labels])

        self.labels['disconnected'] = self.labels.connected.apply(lambda x: list(set(list(range(len(self.labels)))) - x))

        self.labels.connected = self.labels.connected.apply(list)


    def init_embeddings(self, smart_init):
        '''
        Initializes the embeddings randomly on the ball, though following the hierarchical structure. Assumes that the labels are stored in a breadth-first search manner.
        '''
        if smart_init:
            self.embeddings = geoopt.ManifoldTensor(torch.empty((len(self.labels), self.dim)), manifold=self.ball)

            self.embeddings[0] = self.ball.origin(self.dim)
            
            for idx, row in self.labels.iterrows():
                parent_embedding = self.embeddings[row.parent_idx]

                direction = torch.randn_like(parent_embedding)
                if direction @ parent_embedding < 0:
                    direction = -direction

                self.embeddings[idx] = self.ball.geodesic_unit(torch.tensor(0.5) * row.number_len, parent_embedding, direction)
        else:
            self.embeddings = self.ball.random((len(self.labels), self.dim))
            # self.embeddings[0] = self.ball.origin()

        self.embeddings = geoopt.ManifoldParameter(self.embeddings)