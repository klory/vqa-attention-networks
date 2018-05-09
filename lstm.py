import torch
import torch.nn as nn
from torch import optim

import pdb
class LSTM(nn.Module):
    def __init__(self, hidden_dim=512, embed_dim=512, vocab_size=6000, batch_size=16):
        super(LSTM, self).__init__()
        # parameters of lstm
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_layers = 1 # number of stacked layer of lstm

        # network graph
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTM(self.embed_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),\
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        """
        inputs: N, T, V
        """
        max_len = inputs.shape[1]
        embedded_inputs = self.embedding(inputs).view(max_len, self.batch_size, -1) # T, N, E
        h = self.init_hidden()
        hidden_states = list()
        for t in range(max_len):
            o, h = self.lstm(embedded_inputs[t:t+1], h)
            hidden_states.append(h[0])

        # hi
        hidden_states = torch.cat(hidden_states, 0).view(self.batch_size, max_len, -1)
        return hidden_states
