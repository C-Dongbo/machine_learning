import numpy as np
import torch
from copy import deepcopy
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F

class BiLSTM_ATTN(nn.Module):
  def __init__(self, embedding_dim, max_length, vocab_size, num_classes):
    super(BiLSTM_ATTN, self).__init__()
    self.embedding_dim = embedding_dim
    self.max_length = max_length
    self.character_size = vocab_size + 2
    self.num_classes = num_classes
    self.n_hidden = 150
    
    # Embedding Layer
    self.embeddings = nn.Embedding(self.character_size, self.embedding_dim)

    # Bi-directional LSTM for RCNN
    self.lstm = nn.LSTM(input_size=self.embedding_dim,
                        hidden_size=self.n_hidden,
                        num_layers=1,
                        dropout=0.7,
                        bidirectional=True)

    self.out = nn.Linear(self.n_hidden * 2, self.num_classes)
    
  def attention(self, lstm_output, final_state):
    hidden = final_state.view(-1, self.n_hidden * 2, 1)   # hidden : [batch_size, n_hidden * num_directions(=2), 1(=n_layer)]
    attn_weights = torch.bmm(lstm_output, hidden).squeeze(2) # attn_weights : [batch_size, n_step]
    soft_attn_weights = F.softmax(attn_weights, 1)
    # [batch_size, n_hidden * num_directions(=2), n_step] * [batch_size, n_step, 1] = [batch_size, n_hidden * num_directions(=2), 1]
    context = torch.bmm(lstm_output.transpose(1, 2), soft_attn_weights.unsqueeze(2)).squeeze(2)
    return context, soft_attn_weights.data.numpy() # context : [batch_size, n_hidden * num_directions(=2)]    

  def forward(self, data):
    data_in_torch = Variable(torch.from_numpy(np.array(data)).long())
    embeds = self.embeddings(data_in_torch)
    embeds = embeds.permute(1, 0, 2)
    output, (h_n, c_n) = self.lstm(embeds)
    output = output.permute(1, 0, 2)
    attn_output, attention = self.attention(output, h_n)
    return self.out(attn_output) , attention