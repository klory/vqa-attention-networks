import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
class Attention_layer(nn.Module):
    def __init__(self):
        super(Attention_layer, self).__init__()
        self.linear = Linear_layer()

    def forward(self, img_features, qes_features):
        """
        inputs:
            img_featrues: N, L, D
            q_features: N, T, V
        """
        L, D = img_features.shape[1], img_features.shape[2]
        T, V = qes_features.shape[1], qes_features.shape[2]
        assert (D == V), "dimension of img_features and q_features not match"

        img_features = torch.tensor(img_features, dtype=torch.float)
        qes_features = torch.tensor(qes_features, dtype=torch.float)


        q_features = torch.unsqueeze(qes_features, 2) # N, T, 1, V,
        q_features = q_features.repeat(1, 1, L, 1) # N, T, L, V

        i_features = torch.unsqueeze(img_features, 0).view(-1, 1, L, D) # N, 1, L, D
        i_features = i_features.repeat(1, T, 1, 1) # N, T, L, D

        h_temp = i_features + q_features

        # compute attention
        att = self.linear(h_temp, D, 1) # N, T, L
        att = F.softmax(att, dim=2)

        I_hat = torch.matmul(att, img_features) # N, T, D
        return qes_features + I_hat

class Linear_layer(nn.Module):
    def __init__(self):
        super(Linear_layer, self).__init__()
        
    def forward(self, inputs, in_dim, out_dim):
        weights = torch.empty(out_dim, in_dim)
        nn.init.xavier_uniform_(weights)
        o = torch.squeeze(F.linear(inputs, weights))
        return o
