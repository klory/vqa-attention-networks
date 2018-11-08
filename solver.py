import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import copy
import pdb
import sys
import os

from data_loader import VqaDataset
from utils import clean_state_dict

class Solver(object):
  def __init__(self, model, cfg, qa_data):
    self.model_name = cfg.model_name
    print("Model: %s" % self.model_name)
    print(model)
    self.model = model
    self.cfg = cfg
    self.lr = cfg.lr
    self.glove = cfg.glove

    if self.model_name == 'mhb_coAtt' or self.model_name == 'mhb':
      self.criterion = nn.KLDivLoss()
    else:
      self.criterion = nn.CrossEntropyLoss()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.writer = SummaryWriter()

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)
    self.model.to(self.device)

    data_split = ['train', 'val']
    self.datasets = {x: VqaDataset(qa_data, x, cfg, feature_type=cfg.feature_type) for x in data_split}
    self.data_loader = {x: DataLoader(self.datasets[x], batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers) for x in data_split}
    if self.cfg.early_stopping:
      self.min_val_loss = 1e10
      self.i_patience = 0
      self.patience = 10

  def adjust_learning_rate(self):
    self.lr *= self.cfg.decay_rate
    for param_group in self.optimizer.param_groups:
      param_group['lr'] = self.lr

  def train(self):
    num_train_data = len(self.datasets['train'])
    batch_size = self.cfg.batch_size
    if num_train_data % batch_size == 0:
      trainiter_per_epoch = num_train_data // batch_size
    else:
      trainiter_per_epoch = num_train_data // batch_size + 1
    total_train_iter = self.cfg.num_epoch * trainiter_per_epoch
    print('total training iterations', total_train_iter)

    self.best_model = copy.deepcopy(self.model.state_dict())
    
    with tqdm(total=total_train_iter) as pbar:
      for epoch in range(self.cfg.num_epoch):
        # training
        self.model.train()
        for j, data in enumerate(self.data_loader['train']):
          if self.model_name in ['mhb_coAtt', 'mhb']:
            if self.glove:
              i, q, a, _, g = data
              q, i, a, g = q.to(self.device), i.to(self.device), a.to(self.device), g.to(self.device)
              logits = self.model.forward(i, q, glove_matrix=g)
            else:
              i, q, a, _ = data
              q, i, a, = q.to(self.device), i.to(self.device), a.to(self.device)
              logits = self.model.forward(i, q)
          else:
            assert self.cfg.soft_answer == 0, 'soft_answer is not supported'
            if self.glove:
              i, q, a, q_l, g = data
              a = torch.tensor(a, dtype=torch.long)
              q, i, a, q_l, g = q.to(self.device), i.to(self.device), a.to(self.device), q_l.to(self.device), g.to(self.device)
              logits = self.model.forward(i, q, q_l, glove_matrix=g)
            else:
              i, q, a, q_l = data
              a = torch.tensor(a, dtype=torch.long)
              q, i, a, q_l = q.to(self.device), i.to(self.device), a.to(self.device), q_l.to(self.device)
              logits = self.model.forward(i, q, q_l)

          loss = self.criterion(logits, a)
          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()

          pred = F.softmax(logits, dim=1)
          pred = pred.max(1)[1]

          if self.model_name == 'mhb' or self.model_name == 'mhb_coAtt':
            a = a.max(1)[1]
          acc = (pred == a).float().mean()

          step = epoch*trainiter_per_epoch+j+1
          if self.cfg.lr_decay and (step % self.cfg.decay_step == 0):
            self.adjust_learning_rate()
          
          pbar.update(1)

        # validation
        val_loss, val_acc = self.val()
        print(">>> epoch: %d, iter: %d\n"
        ">>> training loss: %f\t training acc: %f\n"
        ">>> val loss: %f\t val acc: %f\n" 
        ">>> learning rate: %f" % (epoch, j+1, loss, acc, val_loss, val_acc, self.lr))
        # training loss and val loss summary
        self.writer.add_scalars(self.cfg.model_name + '/loss', {'train loss': loss, 'val loss': val_loss}, step)
        self.writer.add_scalars(self.cfg.model_name + '/acc', {'train acc': acc, 'val acc': val_acc}, step)

  def val(self):
    total_loss = 0.0
    total_acc = 0.0
    self.model.eval()
    for j, data in enumerate(self.data_loader['val']):
      if self.model_name in ['mhb', 'mhb_coAtt']:
        if self.glove:
          i, q, a, _, g = data
          q, i, a, g = q.to(self.device), i.to(self.device), a.to(self.device), g.to(self.device)
          logits = self.model.forward(i, q, glove_matrix=g)
        else:
          i, q, a, _ = data
          q, i, a = q.to(self.device), i.to(self.device), a.to(self.device)
          logits = self.model.forward(i, q)
      else:
        assert self.cfg.soft_answer == 0, 'soft_answer is not supported'
        if self.glove:
          i, q, a, q_l, g = data
          a = torch.tensor(a, dtype=torch.long)
          q, i, a, q_l, g = q.to(self.device), i.to(self.device), a.to(self.device), q_l.to(self.device), g.to(self.device)
          logits = self.model.forward(i, q, q_l, glove_matrix=g)
        else:
          i, q, a, q_l = data
          a = torch.tensor(a, dtype=torch.long)
          q, i, a, q_l = q.to(self.device), i.to(self.device), a.to(self.device), q_l.to(self.device)
          logits = self.model.forward(i, q, q_l)

      loss = self.criterion(logits, a)

      pred = F.softmax(logits, dim=1)
      pred = pred.max(1)[1]
      if self.model_name == 'mhb' or self.model_name == 'mhb_coAtt':
        a = a.max(1)[1]
      acc = (pred == a).float().mean()
      total_acc += (pred == a).sum(dtype=torch.float32)
      # just random pick one batch for validation
      if self.cfg.mode == 'training':
        break
      if j % int(len(self.data_loader['val']) / 100) == 0:
        print('Processed: %d / %d' % (j, len(self.data_loader['val'])))

    if self.cfg.early_stopping:
      # validation loss
      if loss < self.min_val_loss:
        self.min_val_loss = loss
        self.i_patience = 0
        self.best_model = copy.deepcopy(self.model.state_dict())
      else:
        self.i_patience += 1

      if self.i_patience == self.patience:
        self.save()
        print("validation loss does not decrease in {} epochs, training stops".format(self.patience))
        sys.exit(0)

    if self.cfg.mode == 'training':
      return loss, acc
    else:
      total_acc = total_acc / (len(self.data_loader['val']) * self.cfg.batch_size)
      print('Evaluation accuracy: %f' % total_acc)
      if not os.path.exists(self.cfg.results):
        os.makedirs(self.cfg.results)
      with open(os.path.join(self.cfg.results, self.model_name+'.txt'), 'w') as f:
        f.write('Evaluation accuracy: %.6f' % total_acc)

  def save(self):
    if not os.path.exists(self.cfg.out_dir):
      os.makedirs(self.cfg.out_dir)
    output_filename = os.path.join(self.cfg.out_dir, self.model_name+'.pth')
    if self.cfg.early_stopping:
      self.model.load_state_dict(self.best_model)
    torch.save(clean_state_dict(self.model.state_dict()), output_filename)
