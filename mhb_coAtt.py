import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MHBCoAtt(nn.Module):
  def __init__(self, cfg):
    """
    Implementation of MFB co-attention model.
    Input: 
     img: (before pool5 of 152 ResNet) N, 14*14, 2048
     ques: (sequence of tokens) N, T

    Model:
      1. Embed question into an embedding space
      2. Feed questiong embedded feautre into LSTM
      3. Feed otput from LSTM into question attention model: 1*1 convolution layer -> ReLU -> 1*1 convolution layer -> softmax
      4. Question attentive feautre fuse with img feature by MFB
      5. Fused features are fed into co-attention model
      6. Fused attentive feature fuse with question attentive feature from step 4 by MFB
    """
    super(MHBCoAtt, self).__init__()
    self.cfg = cfg
    # word embedding: q_vocab_size, 1024
    self.word_embedding = nn.Embedding(cfg.q_vocab_size, cfg.emb_dim)
    # LSTM
    if cfg.glove:
        self.lstm = nn.LSTM(input_size=cfg.emb_dim*2,
        hidden_size=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        batch_first=True)
    else:
        self.lstm = nn.LSTM(input_size=cfg.emb_dim,
        hidden_size=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        batch_first=True)

    self.dropout_l = nn.Dropout(p = 0.3)
    # question attention
    self.ques_att_conv1 = nn.Conv2d(cfg.hidden_dim, 512, [1,1])
    self.ques_att_conv2 = nn.Conv2d(512, 2, [1,1])

    # question attentive feature fuse with image feature, according to paper: k * o = 5000, k = 5
    self.ques_proj1 = nn.Linear(2*cfg.hidden_dim, 5000)
    self.img_conv1d = nn.Conv2d(cfg.img_feature_channel, 5000, [1, 1])
    self.dropout_m = nn.Dropout(p = 0.1)

    # co-attention conv layers
    self.co_att_conv1 = nn.Conv2d(1000, 512, [1,1])
    self.co_att_conv2 = nn.Conv2d(512, 2, [1,1])

    # co_attentive feature fuse with question attentive feature
    self.ques_proj2 = nn.Linear(2*cfg.hidden_dim, 5000)
    self.ques_proj3 = nn.Linear(2*cfg.hidden_dim, 5000)
    self.img_proj2 = nn.Linear(2*cfg.img_feature_channel, 5000)
    self.img_proj3 = nn.Linear(2*cfg.img_feature_channel, 5000)

    # prediction fully connected layer
    self.linear_pred = nn.Linear(2000, cfg.a_vocab_size)

  def forward(self, img_features, questions, glove_matrix=None, is_training=True):
    """
    inputs:
      img_feature: from VGG16 conv5-3: N, 196, 1024
      questions: question token, N, 22
    """
    N, L, D = img_features.shape

    que_embedded = F.tanh(self.word_embedding(questions)) # N, T, H
    if self.cfg.glove:
      assert glove_matrix is not None, 'glove should not be NoneType.'
      lstm_o, _ = self.lstm(torch.cat((que_embedded, glove_matrix), dim=2).permute(1,0,2))# T, N, H
    else:
      lstm_o, _ = self.lstm(que_embedded.permute(1, 0, 2)) # T, N, H
    ques_feature = self.dropout_l(lstm_o) # T, N, H

    # Question Attention
    ques_feature = ques_feature.permute((1, 2, 0)) # N, H, T
    ques_feature = torch.unsqueeze(ques_feature, 3) # N, H, T, 1

    ques_att = self.ques_att_conv1(ques_feature) # N, 1024, T, 1 -> N, 512, T, 1
    ques_att = F.relu(ques_att)
    ques_att = self.ques_att_conv2(ques_att) # N, 512, T, 1 -> N, 2, T, 1, generate 2 set of attention weights
    ques_att_list =list()
    for i in range(2):
      att = F.softmax(ques_att[:, i:i+1, :, :], dim=2)
      ques_att_list.append(torch.sum(att * ques_feature, 2, keepdim=True))

    ques_att_feature = torch.cat(ques_att_list, 1) # N, 2*H, 1, 1
    N, H = ques_att_feature.shape[:2]
    ques_att_feature = ques_att_feature.view(N, H)

    # Question attentive feature fused with image feature and fusion
    ques_projed = self.ques_proj1(ques_att_feature) # N 5000
    ques_projed = ques_projed.view(list(ques_projed.shape[:2]) + [1, 1]) # N 5000 1 1

    img_features = torch.unsqueeze(img_features.permute((0, 2, 1)), 3) # N, D, L, 1: N 2048 196 1
    img_projed = self.img_conv1d(img_features) # N, D, L, 1: N 2048 196 1 --> N, 5000, 196, 1
    # fusion: element-wise product
    fusion_feature = img_projed * ques_projed # N, 5000, 196, 1
    fusion_feature = self.dropout_m(fusion_feature)
    fusion_feature = fusion_feature.permute(0, 2, 1, 3).view(N, self.cfg.img_feature_dim, 1000, 5)
    fusion_feature = torch.sum(fusion_feature, 3, keepdim=True) # N, 196, 1000, 1
    # power normalization and normalization
    fusion_feature = fusion_feature.permute(0, 2, 1, 3) # N, 1000, 196, 1
    fusion_pow_normed = torch.sqrt(F.relu(fusion_feature)) - torch.sqrt(F.relu(-fusion_feature))
    fusion_normed = F.normalize(fusion_pow_normed.view(N, -1)) 
    fusion_normed = fusion_normed.view(N, 1000, self.cfg.img_feature_dim, -1) # N, 1000, 196, 1

    # Co-att
    co_att = self.co_att_conv1(fusion_normed) # N, 1000, 196, 1 -> N, 512, 196, 1
    co_att = F.relu(co_att)
    co_att = self.co_att_conv2(co_att) # N, 512, 196, 1 -> N, 2, 196, 1
    co_att_feature_list = list()
    for i in range(2):
      co_att_weights = F.softmax(co_att[:, i:i+1, :, :], dim=2)
      co_att_feature_list.append(torch.sum(co_att_weights * img_features, 2, keepdim=True))

    co_att_feature = torch.cat(co_att_feature_list, 1) # N, 4096
    N, H = co_att_feature.shape[:2]
    co_att_feature = co_att_feature.view(N, H)

    # question att feature and co-att feature fusion
    ques_att_proj = self.ques_proj2(ques_att_feature) # N, 5000
    co_att_proj = self.img_proj2(co_att_feature) # N, 5000
    att_feature = ques_att_proj * co_att_proj # N, 5000
    # power normalization and normalization
    att_feature = self.dropout_m(att_feature)
    att_feature = att_feature.view(N, 1, 1000, -1) # N, 1, 1000, 5
    att_feature = torch.sum(att_feature, 3, keepdim=True) # N, 1, 1000, 1
    att_pow_normed = torch.squeeze(torch.sqrt(F.relu(att_feature)) - torch.sqrt(F.relu(-att_feature)))
    att_pow_normed = att_pow_normed.view(N, -1)
    att_normed2 = F.normalize(att_pow_normed) # N, 1000, 100, 1

    # question att feature and co-att feature fusion
    ques_att_proj = self.ques_proj3(ques_att_feature) # N, 5000
    co_att_proj = self.img_proj3(co_att_feature) # N, 5000
    att_feature = ques_att_proj * co_att_proj # N, 5000
    # power normalization and normalization
    att_feature = self.dropout_m(att_feature)
    att_feature = att_feature.view(N, 1, 1000, -1) # N, 1, 1000, 5
    att_feature = torch.sum(att_feature, 3, keepdim=True) # N, 1, 1000, 1
    att_pow_normed = torch.squeeze(torch.sqrt(F.relu(att_feature)) - torch.sqrt(F.relu(-att_feature)))
    att_pow_normed = att_pow_normed.view(N, -1)
    att_normed3 = F.normalize(att_pow_normed) # N, 1000

    att_normed_23 = torch.cat([att_normed2, att_normed3], 1) # N, 2000
    logits = self.linear_pred(att_normed_23)
    prob = F.log_softmax(logits)

    return prob

class MHB(nn.Module):
  def __init__(self, cfg):
    super(MHB, self).__init__()
    self.model_name = cfg.model_name
    self.cfg = cfg
    # mean pool to make image features of dimension N, 2048
    self.mean_pool = nn.AvgPool2d((14, 14))
    self.Embedding = nn.Embedding(cfg.q_vocab_size, cfg.emb_dim)
    self.LSTM = nn.LSTM(input_size=cfg.emb_dim, hidden_size=cfg.hidden_dim, num_layers=1, batch_first=False)
    
    self.linear_q_1 = nn.Linear(cfg.hidden_dim, 5000)
    self.linear_q_2 = nn.Linear(cfg.hidden_dim, 5000)

    self.linear_i_1 = nn.Linear(cfg.img_feature_channel, 5000)
    self.linear_i_2 = nn.Linear(cfg.img_feature_channel, 5000)

    self.lstm_dropout = nn.Dropout(0.3)
    self.mfb_dropout = nn.Dropout(0.1)

    self.linear_out = nn.Linear(2000, cfg.a_vocab_size)

  def forward(self, img_feature, questions, q_length):
    batch_size, max_len = questions.size()
    lstm_out = torch.zeros((batch_size, self.cfg.hidden_dim), dtype=torch.float).cuda()

    img_feature = img_feature.view(batch_size, 14, 14, self.cfg.img_feature_channel)
    img_feature = img_feature.permute(0, 3, 1, 2)
    i_mean_pooled = self.mean_pool(img_feature) # N, 1, 1, 2048
    q_embedded = self.Embedding(questions) # N, T, V
    q_embedded = q_embedded.permute(1, 0, 2) # T, N, V
    lstm_outs, _ = self.LSTM(q_embedded)

    for i in range(batch_size):
      lstm_out[i] = lstm_outs[q_length[i]-1][i]

    lstm_out = self.lstm_dropout(lstm_out)

    q_proj_1 = self.linear_q_1(lstm_out)
    i_proj_1 = self.linear_i_1(i_mean_pooled.view(batch_size, -1))
    # mhb bilinear fusion with k = 5, o = 1000
    mhb_1 = torch.mul(q_proj_1, i_proj_1)
    mhb_1_dropout = self.mfb_dropout(mhb_1)
    mhb_1 = mhb_1_dropout.view(batch_size, 1, 1000, 5)
    mhb_1 = torch.squeeze(torch.sum(mhb_1, 3)) # N, 1000
    # squared-root relu and l2 normalization
    mhb_1 = torch.sqrt(F.relu(mhb_1)) + (- torch.sqrt(F.relu(-mhb_1)))
    mhb_1 = F.normalize(mhb_1)

    q_proj_2 = self.linear_q_2(lstm_out)
    i_proj_2 = self.linear_i_2(i_mean_pooled.view(batch_size, -1))
    # mhb bilinear fusion with k = 5, o = 1000
    mhb_2 = torch.mul(q_proj_2, i_proj_2)
    mhb_2 = torch.mul(mhb_2, mhb_1_dropout)
    mhb_2 = self.mfb_dropout(mhb_2)
    mhb_2 = mhb_2.view(batch_size, 1, 1000, 5)
    mhb_2 = torch.squeeze(torch.sum(mhb_2, 3)) # N, 1000
    # squared-root relu and l2 normalization
    mhb_2 = torch.sqrt(F.relu(mhb_2)) + (- torch.sqrt(F.relu(-mhb_2)))
    mhb_2 = F.normalize(mhb_2)

    mhb_12 = torch.cat((mhb_1, mhb_2), 1) # N, 2000
    logits = self.linear_out(mhb_22)
    pred = F.log_softmax(logits)

    return pred
