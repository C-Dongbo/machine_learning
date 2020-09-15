
import argparse
import os

import numpy as np
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import SentenceClassificationDataset, preprocess, collate_fn
from models import BiLSTM_ATTN

def save(dirname, model, model_name, *args):
    torch.save(model.state_dict(), os.path.join(dirname, 'model_{}.pt'.format(model_name)))

def train():
    print('unique labels = {}'.format(dataset.get_unique_labels_num()))
    print('vocab size = {}'.format(dataset.get_vocab_size()))

    if config.model_name == 'BILSTM_ATTN':
        model = BiLSTM_ATTN(config.embedding, config.strmaxlen, dataset.get_vocab_size(), config.output_size)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
    
        


    if config.mode == 'train':

        train_loader = DataLoader(dataset=dataset,
                                  batch_size=config.batch,
                                  shuffle=True,
                                  collate_fn=collate_fn,
                                  num_workers=0)

        total_batch = len(train_loader)

        pre_loss = 100000
        for epoch in range(config.max_epoch):
            avg_loss = []
            predictions = []
            label_vars = []
            for i, (data, labels) in enumerate(train_loader):
                
                labels = labels.resize_((len(labels),1))
                predictions, attention = model(data)
                
                loss = criterion(predictions, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # if i%100==0:
                #     print('Batch : ', i + 1, '/', total_batch,
                #         ', Loss in this minibatch: ', loss.item())
                avg_loss.append(loss.item())
            avg_loss_val = np.array(avg_loss).mean()
            print('epoch:', epoch, ' train_loss:', avg_loss_val)
            if avg_loss_val < pre_loss:
                save("./model/", model, config.model_name)
                pre_loss = avg_loss_val
            else:
                pre_loss = avg_loss_val


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--output_size', type=int, default=1) #output label size
    args.add_argument('--max_epoch', type=int, default=20)
    args.add_argument('--batch', type=int, default=32)
    args.add_argument('--strmaxlen', type=int, default=100)
    args.add_argument('--embedding', type=int, default=300)
    args.add_argument('--model_name', type=str, default='BILSTM_ATTN')

    config = args.parse_args()
    dataset = SentenceClassificationDataset('./data', config.strmaxlen)
    
    train()
