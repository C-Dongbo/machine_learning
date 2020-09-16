# -*- coding: utf-8 -*-


import os
import re
import numpy as np
from torch.utils.data import Dataset
import torch
import random
from utils import get_data, make_vocab, read_vocab


class SentenceClassificationDataset(Dataset):
    def __init__(self, dataset_path: str, max_length: int):
        """
        :param dataset_path: 데이터셋 root path
        :param max_length: 문자열의 최대 길이
        """
        self.dataset_path = dataset_path

        with open(os.path.join(dataset_path,'sample_data'),'r',encoding='utf8') as f:
          self.train_sentences , self.train_labels = get_data(f)
          
        with open(os.path.join(dataset_path,'test_data'),'r',encoding='utf8') as f:
          self.test_sentences , self.test_labels = get_data(f)        
        print('data loading complete!')

        if os.path.isfile('./data/vocab.txt'):
            self.vocab = read_vocab()
        else:
            self.vocab = make_vocab(self.train_sentences)

        print('make vocab complete! vocab size = {}'.format(len(self.vocab)))
        
        self.sentences = preprocess(self.vocab, self.train_sentences, max_length)
        self.labels = [np.float32(x) for x in self.train_labels]
        print('training sentences :', len(self.sentences))

    def __len__(self):
        """
        :return: 전체 데이터의 수를 리턴합니다
        """
        return len(self.sentences)

    def __getitem__(self, idx):
        """
        :param idx: 필요한 데이터의 인덱스
        :return: 인덱스에 맞는 데이터, 레이블 pair를 리턴합니다
        """
        return self.sentences[idx], self.labels[idx]

    def get_unique_labels_num(self):
        return len(set(self.labels))

    def get_vocab_size(self):
        return len(self.vocab)

    def get_test_data(self):
        return self.test_sentences, self.test_labels

def preprocess(vocabs: set, data: list, max_length: int):
    """
    :param data: 문자열 리스트 ([문자열1, 문자열2, ...])
    :param max_length: 문자열의 최대 길이
    :return: 벡터 리스트 ([[0, 1, 5, 6], [5, 4, 10, 200], ...]) max_length가 4일 때
    """

    unk = '[unk]'
    pad = '[pad]'

    vocab = {}
    vocab[pad] = 0
    vocab[unk] = 1




    vocab_id =2 
    for vo in vocabs:
        vocab[vo.replace('\n', '')] = vocab_id
        vocab_id += 1

    vectorized_data = []
    for datum in data:
        sent = datum.replace('\n', '').replace('\r', '')
        vec = []
        for token in sent.split(' '):
            if token in vocab.keys():
                vec.append(vocab[token])
            else:
                vec.append(vocab[unk])
        vectorized_data.append(vec)

    zero_padding = np.zeros((len(data), max_length), dtype=np.int32)
    for idx, seq in enumerate(vectorized_data):
        length = len(seq)
        if length >= max_length:
            length = max_length
            zero_padding[idx, :length] = np.array(seq)[:length]
        else:
            zero_padding[idx, :length] = np.array(seq)
    return zero_padding



def collate_fn(data: list):
    """
    :param data: 데이터 리스트
    :return:
    """
    sentences = []
    label = []
    for datum in data:
        sentences.append(datum[0])
        label.append(datum[1])

    sent2tensor = torch.tensor(sentences, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.long)
    label2tensor = torch.tensor(label, device='cuda' if torch.cuda.is_available() else 'cpu', dtype=torch.float)
    return sent2tensor, label2tensor



# if __name__ == '__main__':
