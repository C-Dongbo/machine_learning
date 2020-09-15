import numpy as np

import torch
from torch import nn
from torch.autograd import Variable

class SVM(nn.Module):
  def __init__(self, embedding_dim, max_length, vocab_size, num_classes):
    super(SVM, self).__init__()

    self.embedding_dim = embedding_dim
    self.character_size = vocab_size + 2
    self.max_seq_len = max_length
    self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

    self.out = nn.Linear(self.max_seq_len * self.embedding_dim, num_classes)
    self.weight = self.out.weight

  def forward(self, data):
    data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
    embeds = self.embeddings(data_in_torch)
    input_1d = torch.flatten(embeds, start_dim=1, end_dim = 2)
    return self.out(input_1d)

  def loss(self, predictions, target):
    return torch.mean(torch.clamp(1 - target * predictions, min=0))
