
from torch import nn
import logging
import torch

from .utils.poincare import PoincareDistance

class Sequence01Loss(nn.Module):
    def __init__(self):
        super(Sequence01Loss, self).__init__()
    
    def forward(self, loggits, targets):
        return (loggits.max(1)[1] != targets).sum() / len(loggits)

class SequenceLoss(nn.Module):
    def __init__(self):
        super(SequenceLoss, self).__init__()
        self.cel = nn.CrossEntropyLoss(reduction='mean')

    def forward(self, loggits, targets):
        loss = self.cel(loggits, targets)

        if loss < 0:
            logging.error(loggits, targets)
            raise AssertionError

        return loss


class LabelLoss(nn.Module):
    def __init__(self, dist=PoincareDistance):
        super(LabelLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.dist = dist

    def forward(self, e):
        '''
        Parameters
        ----------
        @e: torch.tensor
            Tensor of size (batch_size, 2 + self.no_disconnected_per_connected, embed_dim). 
            The first component of dimension 1 is the embedding of a label l. 
            The second one is the embedding of a label l' connected to l in the adjecency matrix.
            The last self.no_disconnected_per_connected components are embeddings of labels disconnected from label l in the adjecency matrix.
       '''

        # Project to Poincare Ball
        e = e/(1+torch.sqrt(1+e.norm(dim=-1, keepdim=True)**2))
        # Within a batch take the embeddings of all but the first component
        o = e.narrow(1, 1, e.size(1) - 1)
        # Embedding of the first component
        s = e.narrow(1, 0, 1).expand_as(o)
        dists = self.dist.apply(s, o).squeeze(-1)
        # Distance between the first component and all the remaining component (embeddings of)
        outputs = -dists
        targets = torch.zeros(outputs.shape[0]).long().to(outputs.device)
        return self.loss(outputs, targets)


class SequenceLabelLoss(nn.Module):
    def __init__(self, _lambda):
        super(SequenceLabelLoss, self).__init__()
        self.seq_loss = SequenceLoss()
        self.label_loss = LabelLoss()
        self._lambda = _lambda

    def forward(self, seq_logits, seq_targets, label_embs):
        return self.seq_loss(seq_logits, seq_targets) + self._lambda * self.label_loss(label_embs)