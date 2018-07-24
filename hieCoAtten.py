import torch
import torch.nn as nn
import torch.nn.functional as F

class HieCoAtten(nn.Module):
    def __init__(self, block_num=196, word_num=22, img_size=1024, vocab_size=15881, embed_size=512, att_num=6, output_size=3000):
        super(HieCoAtten, self).__init__()
        self.img_emb = nn.Linear(img_size, embed_size, bias=True)
        self.que_emb = nn.Embedding(vocab_size, embed_size)
        self.fc_Wbv = nn.Linear(embed_size, embed_size)
        self.fc_Wbq = nn.Linear(embed_size, embed_size)
        self.fc_Wv = nn.Linear(embed_size, embed_size)
        self.fc_Wq = nn.Linear(embed_size, embed_size)
        self.fc_Whv = nn.Linear(embed_size, 1)
        self.fc_Whq = nn.Linear(embed_size, 1)
        self.fc = nn.Linear(2*embed_size, output_size)

    def forward(self, img_features, que_features):
        """
        inputs:
            img_featrues: N, 196, 1024
            q_features: N, 22, 15881
        """
        batch_size = img_features.size(0)
        img = self.img_emb(img_features) # [N, 196, 1024] --> [N, 196, 512]
        img = F.dropout(F.relu(img))
        que = self.que_emb(que_features) # [N, 22] --> [N, 22, 512]
        que = F.dropout(que)

        Cv = self.fc_Wbv(img) # N, 196, 512
        Cq = self.fc_Wbv(que) # N, 22, 512
        C = F.tanh(torch.matmul(Cq, Cv.transpose(1,2))) # [N, 22, 196]
        C = F.dropout(C)

        img_ = self.fc_Wv(img) # N, 196, 512
        que_ = self.fc_Wq(que) # N, 22, 512

        Hv = F.tanh(img_ + (torch.matmul(que_.transpose(1, 2), C)).transpose(1,2)) # N, 196, 512
        Hv = F.dropout(Hv)
        av = F.softmax(self.fc_Whv(Hv), dim=1) # N, 196, 1
        av_ = av.permute(0, 2, 1) # N, 1, 196
        v = torch.squeeze(torch.bmm(av_, img)) # N, 512
        av = torch.squeeze(av)

        Hq = F.tanh(que_ + (torch.matmul(img_.transpose(1, 2), C.transpose(1,2))).transpose(1,2)) # N, 22, 512
        Hq = F.dropout(Hq)
        aq = F.softmax(self.fc_Whq(Hq), dim=1) # N, 22, 1
        aq_ = aq.permute(0, 2, 1) # N, 1, 22
        q = torch.squeeze(torch.bmm(aq_, que)) # N, 512
        aq = torch.squeeze(aq)
        
        x = torch.cat((v, q), 0) # N, 1024
        x = x.view(batch_size, -1) # N, 1024
        x = self.fc(x) # [N, 1024] -> [N, #answers]
        return x, av, aq