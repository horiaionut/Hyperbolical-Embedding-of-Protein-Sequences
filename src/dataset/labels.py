import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class Labels(Dataset):
    def __init__(self, adjecency_matrix, no_positive, no_negative):
        self.adj = adjecency_matrix
        self.no_positive = no_positive
        self.no_negative = no_negative

        self.preprocess_connected_labels()

    def preprocess_connected_labels(self):
        no_full_items = len(self.adj) // self.no_positive
        last_idx = no_full_items * self.no_positive

        connected_labels = torch.nonzero(self.adj).shuffle()
        
        self.items = connected_labels[: last_idx].reshape(-1, self.no_positive)
        
        if last_idx < len(connected_labels):
            no_needed = len(connected_labels) - last_idx

            self.items = torch.concat((
                self.items,
                torch.concat((
                    connected_labels[last_idx:],
                    connected_labels[: no_needed]
                ))
            ))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx], self._choose_disconnected(idx)

    def _choose_disconnected(self, idx):


    def _get_connected(self, idx):
        return torch.arange(self.adj.shape[0])[self.adj[idx]]

    def _get_disconnected(self, idx):
        return torch.arange(self.adj.shape[0]).repeat()[self.adj[idx] == False]