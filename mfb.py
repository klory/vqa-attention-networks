import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class MFB(nn.Module):
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
    super(MFB, self).__init__()
    self.cfg = cfg
    # word embedding: q_vocab_size, 1024
    self.word_embedding = nn.Embedding(cfg.q_vocab_size, cfg.emb_dim)
    # LSTM
    self.lstm = nn.LSTM(input_size=cfg.emb_dim,
        hidden_size=cfg.hidden_dim,
        num_layers=cfg.num_layers,
        batch_first=True)

    self.dropout_l = nn.Dropout(p = 0.3)
    # question attention
    self.ques_att_conv1 = nn.Conv2d(cfg.hidden_dim, 1024, [1,1])
    if cfg.model_name == 'mfb-multilayer':
      self.ques_att_multiconv = nn.Conv2d(1024, 512, [1,1])
      self.ques_att_conv2 = nn.Conv2d(512, 2, [1,1])
    else:
      self.ques_att_conv2 = nn.Conv2d(1024, 2, [1,1])

    # question attentive feature fuse with image feature, according to paper: k * o = 5000, k = 5
    self.ques_proj1 = nn.Linear(2*cfg.hidden_dim, 5000)
    self.img_conv1d = nn.Conv2d(cfg.img_feature_channel, 5000, [1, 1])
    self.dropout_m = nn.Dropout(p = 0.1)

    # co-attention conv layers
    self.co_att_conv1 = nn.Conv2d(1000, 1024, [1,1])
    if cfg.model_name == 'mfb-multilayer':
      self.co_att_multiconv = nn.Conv2d(1024, 512, [1,1])
      self.co_att_conv2 = nn.Conv2d(512, 2, [1,1])
    else:
      self.co_att_conv2 = nn.Conv2d(1024, 2, [1,1])

    # co_attentive feature fuse with question attentive feature
    self.ques_proj2 = nn.Linear(2*cfg.hidden_dim, 5000)
    self.img_proj2 = nn.Linear(2*cfg.img_feature_channel, 5000)

    # prediction fully connected layer
    self.linear_pred = nn.Linear(1000, cfg.a_vocab_size)

  def forward(self, img_features, questions, is_training=True):
    """
    inputs:
      img_feature: from VGG16 conv5-3: N, 196, 1024
      questions: question token, N, 22
    """
    # TODO: concat glove embedding
    que_embedded = F.tanh(self.word_embedding(questions))
    lstm_o, _ = self.lstm(que_embedded)
    ques_feature = self.dropout_l(lstm_o) # N, T, H

    # Question Attention
    ques_feature = ques_feature.permute((0, 2, 1)) # N, H, T
    ques_feature = torch.unsqueeze(ques_feature, 3) # N, H, T, 1

    ques_att = self.ques_att_conv1(ques_feature) # N, 1024, T, 1
    ques_att = F.relu(ques_att)
    if self.cfg.model_name == 'mfb-multilayer':
      ques_att = self.ques_att_multiconv(ques_att) # N, 512, T, 1
      ques_att = F.relu(ques_att)
    ques_att = self.ques_att_conv2(ques_att) # N, 2, T, 1, generate 2 set of attention weights
    ques_att_list =list()
    for i in range(2):
      att = F.softmax(ques_att[:, i:i+1, :, :], dim=3)
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
    co_att = self.co_att_conv1(fusion_normed) # N, 1024, 196, 1
    co_att = F.relu(co_att)
    if self.cfg.model_name == 'mfb-multilayer':
      co_att = self.co_att_multiconv(co_att) # N, 512, T, 1
      co_att = F.relu(co_att)
    co_att = self.co_att_conv2(co_att) # N, 2, 196, 1
    #co_att = co_att.view(self.cfg.batch_size, -1)
    co_att_feature_list = list()
    for i in range(2):
      co_att_weights = F.softmax(co_att[:, i:i+1, :, :], dim=3)
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
    att_normed = F.normalize(att_pow_normed) # N, 1000, 100, 1

    logits = self.linear_pred(att_normed)
    prob = F.softmax(logits, dim=1)

    return logits
