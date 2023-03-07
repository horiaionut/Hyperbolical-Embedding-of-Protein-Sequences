import torch
from torch import nn

class LabelEmbedModel(nn.Module):
    def __init__(self, n_labels, emb_dim, dropout_p=0, eye=False):
        super(LabelEmbedModel, self).__init__()
        self.eye = eye
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout_p)
        self.e = nn.Embedding(
                    n_labels, emb_dim,
                    max_norm=1.0,
                    # sparse=True,
                    scale_grad_by_freq=False
                )
        self.init_weights()

    def init_weights(self, scale=1e-4):
        if self.eye:
            nn.init.eye_(self.e.weight)
        else:
            self.e.state_dict()['weight'].uniform_(-scale, scale)

    def forward(self, idx):
        return self.dropout(self.e.weight.clone()[idx])


class Classifier(nn.Module):
    def __init__(self, seq_encoder, tokenizer, label_embeddings, assignable_labels, device):
        super(Classifier, self).__init__()
        self.seq_encoder = seq_encoder
        self.label_embeddings = label_embeddings
        self.tokenizer = tokenizer
        self.device = device
        
        self.assignable_labels = assignable_labels
        
    def forward(self, seqs):
        # TODO: can i tokenize on gpu?
        ids = self.tokenizer.batch_encode_plus(seqs, add_special_tokens=True, padding="longest")

        with torch.no_grad():
            input_ids      = torch.tensor(ids['input_ids']).to(self.device)
            attention_mask = torch.tensor(ids['attention_mask']).to(self.device)

        seq_embd_repr = self.seq_encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq_embds     = seq_embd_repr.last_hidden_state

        with torch.no_grad():
            # TODO: which one to use?
            seq_embds = seq_embd_means_no_inplace_operation(seqs, seq_embds, self.label_embeddings.module.emb_dim, self.device)
            # seq_embds =  seq_embd_means_inplace_operation(seqs, seq_embds)

        return seq_embds @ self.label_embeddings(self.assignable_labels).T # TODO: move the .T to constructor
        
    def classify(self, seqs):
        return self(seqs).max(1)[1]
    
    
def seq_embd_means_no_inplace_operation(seqs, seq_embds, emb_dim, device):
    mask = torch.empty((seq_embds.shape[0], seq_embds.shape[1]))

    for i in range(len(seq_embds)):
        mask[i] = torch.where(torch.arange(seq_embds.shape[1]) <= len(seqs[i]) // 2 + 1, 1, 0)

    mask = mask.unsqueeze(2).repeat(1, 1, emb_dim).to(device)

    return (seq_embds * mask.to(device)).mean(axis=1)


def seq_embd_means_inplace_operation(seqs, seq_embds, device):
    seq_embd_means = torch.empty((len(seq_embds), seq_embds.shape[-1]))

    for i in range(len(seqs)):
        seq_embd_means[i] = seq_embds[i, :len(seqs[i]) // 2 + 1].mean(axis=0)
        
    return seq_embd_means.to(device)