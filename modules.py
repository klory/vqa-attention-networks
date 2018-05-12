import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb
class Attention_layer(nn.Module):
    def __init__(self, att_type=1):
        super(Attention_layer, self).__init__()
        if att_type == 1:
            self.att_layer = Attention_1()
        elif att_type == 2:
            self.att_layer = Attention_2()
        else:
            print("Attention type %d not supported." % att_type)
            break

        self.fc1 = Linear_layer()
        self.fc2 = Linear_layer()
        self.nonlinear_1 = Nonlinear_layer()
        self.nonlinear_2 = Nonlinear_layer()

    def forward(self, feature_1, feature_2):
        """
        inputs:
            feature_1: N, T, V
            featrue_2: N, L, D
        """
        feature_1_embbed = self.nonlinear_1((self.fc1(feature_1, V, V), V, V)
        feature_2_embbed = self.nonlinear_2(self.fc2(feature_2, D, D), D, D)

        return (feature_1_embbed + self.att_layer(feature_1, feature_2), feature_2_embbed)

class Attention_1(nn.Module):
    def __init__(self):
        super(Attention_1, self).__init__()
        self.fc = Linear_layer()

    def forward(self, feature_1, feature_2):
        T, V = feature_1.shape[1], qes_features.shape[2]
        L, D = feature_2.shape[1], img_features.shape[2]
        assert (D == V), "dimension of img_features and q_features not match"

        # conver to pytorch tensor
        feature_1 = torch.tensor(feature_1, dtype=torch.float)
        feature_2 = torch.tensor(feature_2, dtype=torch.float)

        feature1 = torch.unsqueeze(feature_1, 2) # N, T, 1, V,
        feature1 = features1.repeat(1, 1, L, 1) # N, T, L, V

        feature2 = torch.unsqueeze(feature_2, 0).view(-1, 1, L, D) # N, 1, L, D
        feature2 = feature2.repeat(1, T, 1, 1) # N, T, L, D

        h_temp = feature1 + feature2

        # compute attention
        att = self.fc(h_temp, D, 1) # N, T, L
        att = F.softmax(att, dim=2)

        I_hat = torch.matmul(att, features_2) # N, T, D
        return I_hat

class Attention_2(nn.Module):
    def __init__(self):
        super(Attention_2, self).__init__()
        pass

    def forward(self):
        pass
class Nonlinear_layer(nn.Module):
    def __init__(self):
        super(Nonlinear_layer, self).__init__()
        self.fc1 = Linear_layer()
        self.fc2 = Linear_layer()

    def forward(self, inputs, in_dim, out_dim):
        o_1 = self.fc1(inputs, in_dim, out_dim)
        o_2 = self.fc2(inputs, in_dim, out_dim)
        o_tanh = torch.tanh(o_1)
        o_sig = torch.sigmoid(o_2)
        return o = o_tanh * o_sig

class Linear_layer(nn.Module):
    def __init__(self):
        super(Linear_layer, self).__init__()
        
    def forward(self, inputs, in_dim, out_dim):
        weights = torch.empty(out_dim, in_dim)
        nn.init.xavier_uniform_(weights)
        o = torch.squeeze(F.linear(inputs, weights))
        return o
