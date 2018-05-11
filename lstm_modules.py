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

        # network graph
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim),\
                torch.zeros(self.batch_size, self.hidden_dim))

    def forward(self, inputs):
        """
        inputs: N, T, V
        """
        max_len = inputs.shape[1]
        embedded_inputs = self.embedding(inputs).view(max_len, self.batch_size, -1) # T, N, E
        h = self.init_hidden()
        hidden_states = list()
        for t in range(max_len):
            h = self.lstm(embedded_inputs[t:t+1], h)
            hidden_states.append(h[0])

        # hi
        hidden_states = torch.cat(hidden_states, 0).view(self.batch_size, max_len, -1)
        return hidden_states

class LSTM_Attention(nn.Module):
    def __init__(self, hidden_dim=512, embed_dim=512, vocab_size=6000, batch_size=16, dropout_rate=0.5):
        super(LSTM_Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate

        # network graph
        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim)
        self.lstm1 = nn.LSTMCell(self.embed_dim, self.hidden_dim)
        self.lstm2 = nn.LSTMCell(2*self.hidden_dim, self.hidden_dim)
        self.output_layer = nn.Linear(self.hidden_dim, self.vocab_size)

    def init_hidden(self):
        return (torch.zeros(self.batch_size, self.hidden_dim),\
                torch.zeros(self.batch_size, self.hidden_dim))

    def attention(self, h, img_features):
        """
        inputs:
            h: 1, N, H
            img_features: N, L, D
        """
        #TODO: dropout layer???
        #TODO: weights initializing
        L, D = img_features.shape[1], img_features.shape[2]
        assert(D == self.hidden_dim), 'dimension not match for img_features and hidden state'
        alpha = torch.zeros((self.batch_size, L))
        for i in range(L):
            img_vec = img_features[:, i, :]
            alpha[:, i] = torch.sum(h * img_vec, 1)

        alpha = torch.unsqueeze(alpha, 1)
        v_hat = torch.matmul(alpha, img_features)
        return v_hat.view(self.batch_size, -1)

    def forward(self, inputs, img_features):
        """
        inputs: N, T, V
        img_faetures: N, L, D
        """
        # convert input into torch tensor
        inputs = torch.tensor(inputs, dtype=torch.long)
        img_features = torch.tensor(img_features, dtype=torch.float)

        # get the sequence length
        max_len = inputs.shape[1]
        embedded_inputs = self.embedding(inputs).view(max_len, self.batch_size, -1) # T, N, E

        # initialize two lstms for the stacked lstm
        h1, c1 = self.init_hidden()
        h2, c2 = self.init_hidden()
        hidden_states = list()
        for t in range(max_len):
            h1, c1 = self.lstm1(embedded_inputs[t], (h1, c1))
            v_att = self.attention(h1, img_features)

            # concatenate h_att and h1
            input_lstm2 = torch.cat([v_att, h1], dim=1)
            h2, c2 = self.lstm2(input_lstm2, (h2, c2))
            hidden_states.append(h2)

        # re-orgainize view of hidden_states for later use
        hidden_states = torch.cat(hidden_states, 0).view(self.batch_size, max_len, -1)
        return hidden_states
