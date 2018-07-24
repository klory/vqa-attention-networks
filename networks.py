import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Attention_layer
import pdb

class iBOWIMG(nn.Module):
    def __init__(self, img_size, vocab_size, embed_size, output_size):
        super(iBOWIMG, self).__init__()
        self.img_emb = nn.Linear(img_size, embed_size, bias=True)
        self.img_bn = nn.BatchNorm1d(embed_size)
        self.que_emb = nn.Embedding(vocab_size, embed_size)
        self.fc = nn.Linear(2*embed_size, output_size)

    def forward(self, img_features, que_features):
        """
        inputs:
            img_featrues: N, 196, 1024
            q_features: N, 22
        """
        img = self.img_bn(self.img_emb(img_features)) # [N, 4096] --> [N, 512]
        img = F.dropout(F.relu(img))
        que = self.que_emb(que_features) # [N, 22] --> [N, 22, 512]
        que = F.dropout(que)
        que = torch.sum(que, 1)
        x = torch.cat((img, que), 1) # [N, 1024]
        x = self.fc(x) # [N, 1024] -> [N, #answers]
        return x

class AttentionNet(nn.Module):
    def __init__(self, block_num=196, word_num=22, img_size=1024, vocab_size=15881, embed_size=512, att_num=6, output_size=3000):
        super(AttentionNet, self).__init__()
        self.img_emb = nn.Linear(img_size, embed_size, bias=True)
        self.que_emb = nn.Embedding(vocab_size, embed_size)
        for i in range(att_num):
            if i%2 == 0:
                att_type = 1
                module = Attention_layer(embed_size, att_type)
            else:
                att_type = 1
                module = Attention_layer(embed_size, att_type)
            self.add_module("att{}".format(i), module)
        self.fc = nn.Linear(2*block_num*word_num, output_size)
        self.batchnorm = nn.BatchNorm1d(output_size)
        self.att_num = att_num

    def forward(self, img_features, que_features):
        """
        inputs:
            img_featrues: N, 196, 2048
            q_features: N, q_max_len, q_vocab_len
        """
        batch_size = img_features.size(0)
        img = self.img_emb(img_features) # [N, 196, 1024] --> [N, 196, 512]
        img = F.dropout(F.relu(img))
        que = self.que_emb(que_features) # [N, 22] --> [N, 22, 512]
        que = F.dropout(que)
        for i in range(self.att_num):
            if i%2 == 0:
                img, que, que_att = self._modules['att{}'.format(i)](img, que) # que_att: [N, 196, 22], image-guided question attentions
            else:
                que, img, img_att = self._modules['att{}'.format(i)](que, img) # img_att: [N, 22, 196], question-guided image attentions
        
        x = torch.cat((que_att, img_att.transpose(1,2)), 0)
        x = x.view(batch_size, -1)
        x = self.fc(x) # [N, 2*22*196] -> [N, #answers]
        # return F.softmax(x, dim=1), que_att, img_att
        x = self.batchnorm(x)
        return x, que_att, img_att


