import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, label_dataset, hot_labels=True, path=None, df=None):
        if df is None:
            df = pd.read_csv(path, sep=' ')
            
        self.seqs = df.seq.apply(' '.join).to_list()
        
        self.labels = label_dataset.real_to_assignable[df.label.to_list()]
        
        self.labels_one_hot = torch.nn.functional.one_hot(self.labels, len(label_dataset.assignable_labels)).to(float)
    

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return \
            self.seqs[idx], \
            self.labels[idx], \
            self.labels_one_hot[idx]


class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, adj, assignable_labels, real_to_assignable, no_disconnected_per_connected):
        self.unconnected = []
        self.assignable_labels = assignable_labels
        self.real_to_assignable = real_to_assignable
        self.no_disconnected_per_connected = no_disconnected_per_connected
        
        all_items = np.arange(len(adj))

        for row in adj:
            self.unconnected.append(all_items[row == False])

        self.unconnected = np.array(self.unconnected, dtype=object)

        non_zero = np.nonzero(adj)
        self.items = np.concatenate((
                non_zero[0][:, np.newaxis], 
                non_zero[1][:, np.newaxis]
            ), 
            axis=1
        )

    def no_labels(self):
        return len(self.unconnected)
        
    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return np.concatenate((
                self.items[idx],
                np.random.choice(
                    self.unconnected[self.items[idx]][0],
                    self.no_disconnected_per_connected
                )
            )
        )


class SequenceLabelDataset(torch.utils.data.Dataset):
    def __init__(self, sequence_dataset, label_dataset):
        super(SequenceLabelDataset, self).__init__()      

        self.sequence_dataset = sequence_dataset
        self.label_dataset = label_dataset
        self.len = max(len(sequence_dataset), len(label_dataset))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        return *self.sequence_dataset[idx % len(self.sequence_dataset)], \
                self.label_dataset[idx % len(self.label_dataset)]

