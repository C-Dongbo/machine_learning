
import argparse
import os

import numpy as np
import torch

from torch.autograd import Variable
from torch import nn, optim
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from dataset import SentenceClassificationDataset, preprocess, collate_fn
from models import BiLSTM_ATTN

def save(dirname, model, model_name, *args):
    torch.save(model.state_dict(), os.path.join(dirname, 'model_{}.pt'.format(model_name)))

def train():
    print('unique labels = {}'.format(dataset.get_unique_labels_num()))
    print('vocab size = {}'.format(dataset.get_vocab_size()))



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


def eval(raw_data):

    preprocessed_data = preprocess('../data/processing_data',raw_data, config.strmaxlen)
    model.load_state_dict(torch.load("./model/model.pt"))
    model.eval()

    output_prediction = model(preprocessed_data)
    point = output_prediction.data.squeeze(dim=1).tolist()
    point = [np.argmax(p) for p in point]

    return list(zip(np.zeros(len(point)), point))

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--mode', type=str, default='train')
    args.add_argument('--pause', type=int, default=0)
    args.add_argument('--iteration', type=str, default='0')
    args.add_argument('--output_size', type=int, default=1) #output label size
    args.add_argument('--max_epoch', type=int, default=20)
    args.add_argument('--batch', type=int, default=64)
    args.add_argument('--strmaxlen', type=int, default=100)
    args.add_argument('--embedding', type=int, default=300)
    args.add_argument('--model_name', type=str, default='BILSTM_ATTN')

    config = args.parse_args()
    dataset = SentenceClassificationDataset('./data', config.strmaxlen)

    if config.model_name == 'BILSTM_ATTN':
        model = BiLSTM_ATTN(config.embedding, config.strmaxlen, dataset.get_vocab_size(), config.output_size)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)

    train()



    '''
    evaluate test data
    '''
    test_sent, test_label = dataset.get_test_data()
    res = eval(test_sent)

    cnt = 0
    y_pred = [] 
    y_true = []
    for idx ,result in enumerate(res):
        # print('predict = {}, true = {}'.format(result[1], label[idx]))
        y_pred.append(result[1])
        y_true.append(int(test_label[idx].replace("\n","")))
        if result[1] == int(test_label[idx].replace("\n","")):
            cnt += 1

    print(classification_report(y_true, y_pred, target_names=['class 0', 'class 1']))
    print('Accuracy : ' + str(cnt/len(test_label)))