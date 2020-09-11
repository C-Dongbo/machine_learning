
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


from sklearn.model_selection import train_test_split

def fit(X, Y, model, args):

    

    X = torch.FloatTensor(X)
    Y = torch.FloatTensor(y)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=48)

    N = len(Y)

    optimizer = optim.SGD(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epoch):
        perm = torch.randperm(N)
        sum_loss = 0

        for i in range(0, N, args.batchsize):
            x = X[perm[i : i + args.batchsize]].to(args.device)
            y = Y[perm[i : i + args.batchsize]].to(args.device)

            optimizer.zero_grad()
            output = model(x).squeeze()
            weight = model.weight.squeeze()

            loss = torch.mean(torch.clamp(1 - y * output, min=0))
            loss += args.c * (weight.t() @ weight) / 2.0

            loss.backward()
            optimizer.step()

            sum_loss += float(loss)

        print("Epoch: {:4d}\tloss: {}".format(epoch, sum_loss / N))


if __name__ =='__main__':

