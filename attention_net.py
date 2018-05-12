import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Attention_layer, Nonlinear_layer

import pdb
class Attention_net(nn.Module):
    def __init__(self, img_size=1024, vocab_size=15881, embed_size=512, att_num=6, output_size=3000):
        super(Attention_net, self).__init__()
        self.img_emb = Nonlinear_layer(img_size, embed_size)
        self.que_emb = nn.Embedding(self.vocab_size, self.embed_dim)
        for i in range(att_num):
            if i%2 == 0:
                f1_size, f2_size, att_type = embed_size, embed_size, 1
                module = Attention_layer(f1_size, f2_size, att_type)
            else:
                f1_size, f2_size, att_type = embed_size, embed_size, 1
                module = Attention_layer(f1_size, f2_size, att_type)
            self.add_module("att{}".format(i), module)
        self.fc = nn.Linear(2*embed_size, output_size)
        self.att_num = att_num

    def forward(self, img_features, que_features):
        """
        inputs:
            img_featrues: N, 49, 1024
            q_features: N, 22, 15881
        """
        img = self.img_emb(img_features) # [N, 49, 1024] --> [N, 49, 512]
        que = self.que_emb(que_features) # [N, 22, 15881] --> [N, 22, 512]
        que = F.dropout(que)
        for i in range(self.att_num):
            if i%2 == 0:
                img, que, que_att = self._modules['att{}'.format(i)](img, que) # que_att: [N, 49, 22]
            else:
                que, img, img_att = self._modules['att{}'.format(i)](que, img) # img_att: [N, 22, 49]
        
        
        return qes_features + I_hat
