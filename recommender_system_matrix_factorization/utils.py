import numpy as np
import os


def rmse(pred_list, true_list):
    cost = 0
    for i in range(len(pred_list)):
        cost += pow(float(true_list[i]) - float(pred_list[i]), 2)
    return np.sqrt(cost/len(pred_list))


def load_each_matrix(model_path):
    Q = np.load(os.path.join(model_path,'Q.npy'))
    P = np.load(os.path.join(model_path,'P.npy'))
    Q_b = np.load(os.path.join(model_path,'Q_bias.npy'))
    P_b = np.load(os.path.join(model_path,'P_bias.npy'))
    b = np.load(os.path.join(model_path,'global_bias.npy'))
    return Q, P, Q_b, P_b, b 
    

def load_id2idx(model_path):
    userId2idx, movieId2idx = {}, {}
    with open(os.path.join(model_path,'userId2idx'),'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            id, idx = line.split('\t')
            userId2idx[id] = idx.replace('\n','')
        
    with open(os.path.join(model_path,'movieId2idx'),'r',encoding='utf8') as f:
        lines = f.readlines()
        for line in lines:
            id, idx = line.split('\t')
            movieId2idx[id] = idx.replace('\n','')

    return userId2idx, movieId2idx


    