import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
import time
import copy
import pdb
import sys
import os

from timer import Timer
from data_loader import VqaDataset
from utils import clean_state_dict

class Solver(object):
  def __init__(self, model, cfg, qa_data):
    self.model = model
    self.cfg = cfg
    self.lr = cfg.lr

    self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    self.writer = SummaryWriter()

    data_split = ['train', 'val']
    self.datasets = {x: VqaDataset(qa_data, x, feature_type=cfg.feature_type) for x in data_split}
    self.data_loader = {x: DataLoader(self.datasets[x], batch_size=cfg.batch_size, shuffle=cfg.shuffle, num_workers=cfg.num_workers) for x in data_split}
    if self.cfg.early_stopping:
      self.min_val_loss = 200.0
      self.val_count = 0

  def adjust_learning_rate(self):
    for param_group in self.optimizer.param_groups:
      self.lr *= self.cfg.decay_rate
      param_group['lr'] = self.lr

  def train(self):
    # self.model.apply(self.weights_init)
    self.criterion = nn.CrossEntropyLoss()

    num_train_data = len(self.datasets['train'])
    batch_size = self.cfg.batch_size
    if num_train_data % batch_size == 0:
      trainiter_per_epoch = num_train_data // batch_size
    else:
      trainiter_per_epoch = num_train_data // batch_size + 1
    total_train_iter = self.cfg.num_epoch * trainiter_per_epoch

    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
      print("Let's use", torch.cuda.device_count(), "GPUs!")
      self.model = nn.DataParallel(self.model)
    self.model.to(self.device)
    self.best_model = copy.deepcopy(self.model.state_dict())

    # training
    timer = Timer()
    last_checktime = 0.0
    last_step = 0
    with tqdm(total=total_train_iter) as pbar:
      for e in range(self.cfg.num_epoch):
        acc = 0.0
        self.model.train()
        for j, data in enumerate(self.data_loader['train']):
          now = time.time()
          i, q, a = data

          timer.tic()
          q, i, a = q.to(self.device), i.to(self.device), a.to(self.device)

          logits = self.model.forward(i, q)
          loss = self.criterion(logits, a)

          self.optimizer.zero_grad()
          loss.backward()
          self.optimizer.step()
          timer.toc()

          pred = F.softmax(logits, dim=1)
          pred = pred.data.max(1)[1]
          acc = (pred == a.data).float().mean()

          step = e*trainiter_per_epoch+j+1
          if self.cfg.lr_decay and (step % self.cfg.decay_step == 0):
            self.adjust_learning_rate()

          if (now - last_checktime > self.cfg.smm_interval) or ((e+j)==0):
            # print training loss and validation loss(random pick one batch from validation)
            val_loss, val_acc = self.val()
            print(">>> epoch: %d, iter: %d\n"
            ">>> training loss: %f\t training acc: %f\n"
            ">>> val loss: %f\t val acc: %f\n" 
            ">>> speed: %f per iter\n"
            ">>> learning rate: %f" % (e, j+1, loss.data, acc.data, val_loss.data, val_acc.data, timer.average_time, self.lr))
            # training loss and val loss summary
            self.writer.add_scalars(self.cfg.model_name + '/loss', {'train loss': loss.data, 'val loss': val_loss.data}, step)
            self.writer.add_scalars(self.cfg.model_name + '/acc', {'train acc': acc.data, 'val acc': val_acc.data}, step)
            last_checktime = now
            pbar.update(step - last_step)
            last_step = step

  def val(self):
    num_val_data = len(self.datasets['val'])
    idx = np.random.choice(num_val_data, self.cfg.batch_size)
    valiter_per_epoch = num_val_data // self.cfg.batch_size
    l_value = 0.0
    self.model.eval()
    for j, data in enumerate(self.data_loader['val']):
      i, q, a = data
      q, i, a = q.to(self.device), i.to(self.device), a.to(self.device)

      logits = self.model.forward(i, q)
      pred = F.softmax(logits, dim=1)
      pred = pred.data.max(1)[1]
      acc = (pred == a.data).float().mean()

      loss = self.criterion(logits, a)
      # just random pick one batch for validation
      break

    if self.cfg.early_stopping:
      # validation loss
      l_value = loss.data
      if l_value < self.min_val_loss:
        self.min_val_loss = l_value
        self.val_count = 0
        self.best_model = copy.deepcopy(self.model.state_dict())
      else:
        self.val_count += 1

      if self.val_count == 10:
        self.save()
        print("Early stopping!")
        sys.exit(0)
    self.model.train()
    return loss, acc

  def save(self):
    output_filename = os.path.join(self.cfg.out_dir, self.cfg.model_name+'.pth')
    if self.cfg.early_stopping:
      self.model.load_state_dict(self.best_model)
    torch.save(clean_state_dict(self.model.state_dict()), output_filename)
