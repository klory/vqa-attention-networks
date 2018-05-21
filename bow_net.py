import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Attention_layer
import pdb

class Bow_net(nn.Module):
    def __init__(self, img_size=4096, vocab_size=15881, embed_size=512, output_size=3000):
        super(Bow_net, self).__init__()
        self.img_emb = nn.Linear(img_size, embed_size, bias=True)
        self.img_bn = nn.BatchNorm1d(embed_size)
        self.que_emb = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(2*embed_size, output_size)

    def forward(self, img_features, que_features):
        """
        inputs:
            img_featrues: N, 49, 1024
            q_features: N, 22
        """
        N = img_features.size(0)
        img = self.img_bn(self.img_emb(img_features)) # [N, 4096] --> [N, 512]
        img = F.dropout(F.relu(img))
        que = self.que_emb(que_features) # [N, 22] --> [N, 22, 512]
        que = F.dropout(que)
        que = torch.sum(que, 1)
        x = torch.cat((img, que), 1) # [N, 1024]
        x = self.fc(x) # [N, 1024] -> [N, #answers]
        return x