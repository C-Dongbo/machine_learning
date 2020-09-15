
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SVM
from dataset import SentenceClassificationDataset, collate_fn


from sklearn.model_selection import train_test_split



def train():   

    train_loader = DataLoader(dataset=dataset,
                                batch_size=config.batch,
                                shuffle=True,
                                collate_fn=collate_fn,
                                num_workers=0)

    model = SVM(config.embedding, config.strmaxlen, dataset.get_vocab_size(), config.output_size)
    optimizer = optim.SGD(model.parameters(), lr=config.lr)
    model.train()
    for epoch in range(config.epoch):

        sum_loss = 0
        for i, (data, labels) in enumerate(train_loader):


            optimizer.zero_grad()
            output = model(data).squeeze()
            weight = model.weight.squeeze()
            weight = weight.reshape((weight.shape[0],1))
            
            loss = model.loss(output, labels)
            tmp = weight.t() @ weight
            loss += config.c * tmp[0][0] / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss /len(dataset)))


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--c", type=float, default=0.01)
    args.add_argument("--lr", type=float, default=0.1)
    args.add_argument('--strmaxlen', type=int, default=50)
    args.add_argument('--embedding', type=int, default=300)    
    args.add_argument("--batch", type=int, default=32)
    args.add_argument("--epoch", type=int, default=10)
    args.add_argument('--output_size', type=int, default=1)
    args.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    config = args.parse_args()

    dataset = SentenceClassificationDataset('./data', config.strmaxlen)

    train()