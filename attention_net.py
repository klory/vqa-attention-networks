import torch
import torch.nn as nn
import torch.nn.functional as F
from modules import Attention_layer
import pdb

class Attention_net(nn.Module):
    def __init__(self, block_num=49, word_num=22, img_size=1024, vocab_size=15881, embed_size=512, att_num=6, output_size=3000):
        super(Attention_net, self).__init__()
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
        self.att_num = att_num

    def forward(self, img_features, que_features):
        """
        inputs:
            img_featrues: N, 49, 1024
            q_features: N, 22, 15881
        """
        batch_size = img_features.size(0)
        img = self.img_emb(img_features) # [N, 49, 1024] --> [N, 49, 512]
        img = F.dropout(F.relu(img))
        que = self.que_emb(que_features) # [N, 22] --> [N, 22, 512]
        que = F.dropout(que)
        for i in range(self.att_num):
            if i%2 == 0:
                img, que, que_att = self._modules['att{}'.format(i)](img, que) # que_att: [N, 49, 22]
            else:
                que, img, img_att = self._modules['att{}'.format(i)](que, img) # img_att: [N, 22, 49]
        
        x = torch.cat((que_att, img_att), 0)
        x = x.view(batch_size, -1)
        # pdb.set_trace()
        x = self.fc(x) # [N, 2*22*49] -> [N, #answers]
        return F.softmax(x, dim=1), que_att, img_att
