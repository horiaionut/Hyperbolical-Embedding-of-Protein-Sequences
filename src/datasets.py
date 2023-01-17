import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

class SequenceDataset(Dataset):
    def __init__(self, path, no_labels):
        df = pd.read_csv(path, sep=' ')
        df.seq = df.seq.apply(' '.join)

        self.seqs = df.seq.to_list()
        self.labels_one_hot = torch.nn.functional.one_hot(torch.tensor(df.label.to_list()), no_labels)

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return \
            self.seqs[idx], \
            self.labels_one_hot[idx]


class LabelDataset(torch.utils.data.Dataset):
    def __init__(self, adj, no_disconnected_per_connected):
        self.unconnected = []
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
        return len(self.items)

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

