import torch
torch.manual_seed(1)
torch.cuda.manual_seed(1)

import logging
logging.basicConfig(level=logging.DEBUG,filename='../../logs/cath-lr=0.01.log')   

from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from transformers import T5Tokenizer, T5EncoderModel

from ..datasets import SequenceLabelDataset, SequenceDataset
from ..cath import CathLabelDataset
from ..losses import SequenceLabelLoss, SequenceLoss, Sequence01Loss
from ..models import LabelEmbedModel, Classifier

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using {}".format(device))



################ DATA

label_dataset = CathLabelDataset(path='../../data/cath/labels.csv', no_disconnected_per_connected=10)
seq_train_dataset = SequenceDataset(label_dataset, path='../../data/cath/train.csv', )

trainloader = torch.utils.data.DataLoader(
    seq_train_dataset,
    # SequenceLabelDataset(seq_train_dataset, label_dataset),
    batch_size=2,
    shuffle=True,
    num_workers=2,
    pin_memory=True
)

seq_val_dataset = SequenceDataset(label_dataset, path='../../data/cath/val.csv')

valloader = torch.utils.data.DataLoader(
    seq_val_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=1,
    pin_memory=True
)



################ MODELS

seq_encoder = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_half_uniref50-enc") #.to(device)
seq_encoder = torch.nn.DataParallel(seq_encoder).to(device) # for training on multiple GPUs

tokenizer = T5Tokenizer.from_pretrained('Rostlab/prot_t5_xl_half_uniref50-enc', do_lower_case=False)

label_embeddings = LabelEmbedModel(label_dataset.no_labels(), eye=True) #.to(device)
label_embeddings = torch.nn.DataParallel(label_embeddings).to(device) # for training on multiple GPUs

classifier = Classifier(seq_encoder, tokenizer, label_embeddings, label_dataset.assignable_labels, device)


################ TRAIN

train_loss = SequenceLoss()
eval_bce_loss = SequenceLoss()
eval_01_loss = Sequence01Loss()

optimizer = torch.optim.Adam([
        {'params': classifier.parameters(), 'lr': 0.01}
        # {'params': seq_encoder.parameters(), 'lr': 0.001}
    ])


def evaluate(classifier, loader, loss_fn, process_batch):
    running_loss = 0
    
    classifier.train(False)
    
    for idx, batch in enumerate(loader):
        seqs, labels = process_batch(batch)
        
        labels = labels.to(device)
        
        logits = classifier(seqs)
        loss = loss_fn(logits, labels)
        
        running_loss += loss.item()
    
    return running_loss / len(loader)


def train(classifier, trainloader, valloader, validate_after=1000, batch_loss_after=50):
    running_loss = 0
    last_loss = 0
    
    pbar = tqdm(enumerate(trainloader))
    
    classifier.train(True)
        
    for idx, batch in pbar:
        seqs, _, labels_hot = batch #, edges = batch
        labels_hot = labels_hot.to(device)
        # edges = edges.to(device)
        
        optimizer.zero_grad()
        
        logits = classifier(seqs)
        # loss =  train_loss(logits, labels_hot, label_model(edges))
        loss =  train_loss(logits, labels_hot)
        
        running_loss += loss.item()
        loss.backward()
        optimizer.step()
        
        # REPORT
        
        if idx % validate_after == validate_after - 1:
            bce = evaluate(classifier, valloader, eval_bce_loss, lambda batch: (batch[0], batch[2]))
            l01 = evaluate(classifier, valloader, eval_01_loss, lambda batch: (batch[0], batch[1]))
            
            print(f'val bce loss: {bce}')
            print(f'val 0-1 loss: {l01}')
            
            logging.debug(f'val bce loss: {bce}')
            logging.debug(f'val 0-1 loss: {l01}')
            
            classifier.train(True)

        if idx % batch_loss_after == batch_loss_after - 1:
            last_loss = running_loss / batch_loss_after
            logging.debug(f"Batch {idx} / {len(trainloader)}, loss {last_loss}")
            pbar.set_description(f"Batch {idx} / {len(trainloader)}, loss {last_loss}")
            running_loss = 0.

    return last_loss


EPOCHS = 10
for epoch in range(EPOCHS):
    loss = train(classifier, trainloader, valloader)