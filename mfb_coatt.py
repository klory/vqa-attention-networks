import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MFB(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.word_embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.lstm = nn.LSTM(input_size=cfg.emb_dim,
                hidden_size=cfg.hidden_dim,
                num_layers=cfg.num_layers,
                batch_first=True)

        # question attention conv layers
        self.ques_att_conv1 = nn.Conv2d(cfg.hidden_dim, 512, [1,1])
        # number of question for glimpse: 2
        self.ques_att_conv2 = nn.Conv2d(512, 2, [1,1])

        # question attentive feature fuse with image feature, according to paper: k * o = 5000, k = 5
        self.ques_proj1 = nn.Linear(2*cfg.hidden_dim, 5000)
        self.img_conv1d = nn.Conv2d(cfg.img_feature_channel, 5000, [1, 1])

        # co-attention conv layers
        self.co_att_conv1 = nn.Conv2d(1000, 512, [1,1])
        # number of question for glimpse: 2
        self.co_att_conv2 = nn.Conv2d(512, 2, [1,1])

        # co_attentive feature fuse with question attentive feature
        self.ques_proj2 = nn.Linear(2*cfg.hidden_dim, 5000)
        self.img_proj2 = nn.Linear(2*cfg.img_feature_channel, 5000)

        # prediction fully connected layer
        self.linear_pred = nn.Linear(1000, 1000)

    def forward(self, img_features, questions, is_training=True):
        """
        inputs:
            img_feature: from VGG16 conv5-3: N, 196, 512
            questions: question token, N, 22
        """
        que_embedded = F.tanh(self.word_embedding(questions))

        lstm_o, _ = self.lstm(que_embedded)
        """
        ques_feature = torch.zeros((self.cfg.batch_size, self.cfg.hidden_dim), dtype=torch.float)
        # get output for every question
        for i in range(self.cfg.batch_size):
            ques_feature[i] = lstm_o[i][question_len[i] ]
        """

        ques_feature = F.dropout(lstm_o, training=is_training) # N, T, H

        # Question Attention
        ques_feature = ques_feature.permute((0, 2, 1)) # N, H, T
        ques_feature = torch.unsqueeze(ques_feature, 3) # N, H, T, 1

        ques_att = self.ques_att_conv1(ques_feature) # N, 512, T, 1
        ques_att = F.relu(ques_att)
        # 2: generate multiple attention weights
        ques_att = self.ques_att_conv2(ques_att) # N, 2, T, 1
        ques_att_list =list()
        for i in range(2):
            att = F.softmax(ques_att[:, i:i+1, :, :], dim=3)
            ques_att_list.append(torch.sum(att * ques_feature, 2, keepdim=True))

        ques_att_feature = torch.cat(ques_att_list, 1) # N, 2*H
        N, H = ques_att_feature.shape[:2]
        ques_att_feature = ques_att_feature.view(N, H)

        # Image feature and question attentive feature fusion
        # TODO debug here, N, 196, 5000, not as expected
        img_features = torch.unsqueeze(img_features.permute((0, 2, 1)), 3) # N, D, L, 1: N 512 196 1
        img_projed = self.img_conv1d(img_features) # N, D, L, 1: N 512 196 1 --> N, 5000, 196, 1
        ques_projed = self.ques_proj1(ques_att_feature) # N 5000
        ques_projed = ques_projed.view(list(ques_projed.shape[:2]) + [1, 1]) # N 5000 1 1

        # fusion: element-wise product
        fusion_feature = img_projed * ques_projed # N, 5000, 196, 1
        fusion_feature = F.dropout(fusion_feature)
        fusion_feature = fusion_feature.permute(0, 2, 1, 3).view(self.cfg.batch_size, self.cfg.img_feature_dim, 1000, 5)
        fusion_feature = torch.sum(fusion_feature, 3, keepdim=True) # N, 196, 1000, 1
        # power normalization and normalization
        fusion_feature = fusion_feature.permute(0, 2, 1, 3) # N, 1000, 196, 1
        fusion_pow_normed = torch.sqrt(F.relu(fusion_feature)) - torch.sqrt(F.relu(-fusion_feature))
        fusion_normed = F.normalize(fusion_pow_normed.view(N, -1)) # N, 1000, 196, 1
        fusion_normed = fusion_normed.view(self.cfg.batch_size, 1000, self.cfg.img_feature_dim, -1)


        # Co-att
        co_att = self.co_att_conv1(fusion_normed) # N, 512, 100, 1
        co_att = F.relu(co_att)
        co_att = self.co_att_conv2(co_att) # N, 2, 100, 1
        #co_att = co_att.view(self.cfg.batch_size, -1)
        co_att_weights = F.softmax(co_att, dim=3)
        co_att_feature_list = list()
        for i in range(2):
            co_att_feature_list.append(torch.sum(co_att_weights[:, i:i+1, :, :] * img_features, 2, keepdim=True))

        co_att_feature = torch.cat(co_att_feature_list, 1) # N, 1024
        N, H = co_att_feature.shape[:2]
        co_att_feature = co_att_feature.view(N, H)

        # question att feature and co-att feature fusion
        ques_att_proj = self.ques_proj2(ques_att_feature) # N, 5000
        co_att_proj = self.img_proj2(co_att_feature) # N, 5000
        att_feature = ques_att_proj * co_att_proj # N, 5000
        # power normalization and normalization
        att_feature = F.dropout(att_feature)
        att_feature = att_feature.view(self.cfg.batch_size, 1, 1000, -1) # N, 1, 1000, 5
        att_feature = torch.sum(att_feature, 3) # N, 1, 1000, 1
        att_pow_normed = torch.squeeze(torch.sqrt(F.relu(att_feature)) - torch.sqrt(F.relu(-att_feature)))
        att_pow_normed = att_pow_normed.view(N, -1)
        att_normed = F.normalize(att_pow_normed) # N, 1000, 100, 1

        logits = self.linear_pred(att_normed)
        prob = F.softmax(logits, dim=1)

        return logits
