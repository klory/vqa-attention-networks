import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
import sys

class Attention_layer(nn.Module):
    def __init__(self, feature_size, att_type=1):
        super(Attention_layer, self).__init__()
        if att_type == 1:
            self.att_layer = Attention_1(feature_size)
        elif att_type == 2:
            self.att_layer = Attention_2(feature_size)
        else:
            sys.exit(0)
            
        self.nonlinear_1 = Nonlinear_layer(feature_size)
        self.nonlinear_2 = Nonlinear_layer(feature_size)

    def forward(self, feature_1, feature_2):
        feature_1_embbed = self.nonlinear_1(feature_1)
        feature_2_embbed = self.nonlinear_2(feature_2)

        f_hat, att = self.att_layer(feature_1, feature_2)

        return (feature_2_embbed + f_hat, feature_1_embbed, att)

class Attention_1(nn.Module):
    def __init__(self, feature_size):
        super(Attention_1, self).__init__()
        self.fc = nn.Linear(feature_size, 1)

    def forward(self, feature_1, feature_2):
        """
        inputs:
            feature_1: N, L, D
            featrue_2: N, T, V
        """
        L, D = feature_1.shape[1], feature_1.shape[2]
        T, V = feature_2.shape[1], feature_2.shape[2]
        assert (D == V), "dimension of feature_1 and feature_2 not match"

        feature2 = torch.unsqueeze(feature_2, 2) # N, T, 1, V,
        feature2 = feature2.repeat((1, 1, L, 1)) # N, T, L, V

        feature1 = torch.unsqueeze(feature_1, 0).permute(1, 0, 2, 3) # N, 1, L, D
        feature1 = feature1.repeat((1, T, 1, 1)) # N, T, L, D

        h_temp = feature1 + feature2

        # compute attention
        att = torch.squeeze(self.fc(h_temp)) # N, T, L
        if len(att.shape) == 2:
            att = torch.unsqueeze(att, 0)
        att = F.softmax(att, dim=2) # weights for feature_1

        f_hat = torch.matmul(att, feature_1) # N, T, D
        return f_hat, att

class Attention_2(nn.Module):
    def __init__(self, feature_size):
        super(Attention_2, self).__init__()
        self.fc1 = nn.Linear(feature_size, feature_size, bias=False)
        self.fc2 = nn.Linear(feature_size, 1)

    def forward(self, feature_1, feature_2):
        L, D = feature_1.shape[1], feature_1.shape[2]
        T, V = feature_2.shape[1], feature_2.shape[2]
        assert (D == V), "dimension of img_feature and q_feature not match"

        feature1 = self.fc1(feature_1) # N, L, D
        att = torch.matmul(feature_2, feature1.permute(0, 2, 1)) # N, T, L
        att = F.softmax(att, dim=2) # N, T, L

        f_hat = torch.matmul(att, feature_1) # N, T, D
        return f_hat, att

class Nonlinear_layer(nn.Module):
    def __init__(self, f_size):
        super(Nonlinear_layer, self).__init__()
        self.fc1 = nn.Linear(f_size, f_size)
        self.fc2 = nn.Linear(f_size, f_size)

    def forward(self, inputs):
        o_1 = self.fc1(inputs)
        o_2 = self.fc2(inputs)
        o_tanh = torch.tanh(o_1)
        o_sig = torch.sigmoid(o_2)
        return o_tanh * o_sig
