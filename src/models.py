from torch import nn

class LabelEmbedModel(nn.Module):
    def __init__(self, n_labels, emb_dim=1024, dropout_p=0, eye=False):
        super(LabelEmbedModel, self).__init__()
        self.eye = eye
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

