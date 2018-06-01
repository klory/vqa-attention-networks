import torch
import torch.nn as nn
import numpy as np
from tensorboardX import SummaryWriter
from mfb_coatt import MFB
from data_loader import *
from tqdm import tqdm
import copy
import pdb
from utils import *

class Solver(object):
    def __init__(self, model, cfg, training_data, val_data):
    self.model = model
    self.cfg = cfg
    self.writer = SummaryWriter()

    self.train_ques, self.train_image_features, self.train_image_id_list = train_data
    self.val_ques, self.val_image_features, self.val_image_id_list = val_data

def train(self):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.lr)
    if self.cfg.lr_decay:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(10,50), gamma=0.90)
    
    num_train_data = len(self.train_ques)
    batch_size = self.cfg.batch_size
    trainiter_per_epoch = num_train_data // batch_size
    total_train_iter = self.cfg.num_epoch * trainiter_per_epoch

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(self.model)
    self.model.to(device)
    best_model = copy.deepcopy(model.state_dict())

    # training
    with tqdm(total=total_train_iter) as pbar:
        for e in range(self.cfg.num_epoch):
            idx = np.arange(num_train_data)
            if self.cfg.shuffle:
                np.random.shuffle(idx)
                self.train_ques = np.array(self.train_ques)[idx]
            acc = 0.0
            self.model.train()
            for j in range(1, trainiter_per_epoch+1):
                i, q, a = sample_batch_hard(j,
                        batch_size,
                        self.train_img_features,
                        self.train_img_id_list,
                        self.train_ques,
                        # TODO qa_data
                        qa_data)

                q, i, a = q.to(device), i.to(device), a.to(device)

                logits = self.model.forward(i, q)
                loss = criterion(logits, a)
                l_value = loss.data

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pred = torch.argmax(logits.data, dim=1)
                acc += torch.sum(pred.data == a)

                if j%self.cfg.print_freq == 0:
                    #print("epoch: %d, iter: %d, loss: %f" % (e, j, l_value))
                    writer.add_scalar('mfb/train_loss', l_value, e*trainiter_per_epoch+j)
                    pbar.update(self.cfg.print_freq)
            acc = acc.float() / float(num_train_data)
            #print("Accuracy for epoch %d is %f." % (e, acc))
            writer.add_scalar('mfb/train_acc', acc.data, e*trainiter_per_epoch+j)
            if lr_decay and e>9:
                scheduler.step()

        self.val()

    def val(self):
        num_val_data = len(self.val_ques)
        valiter_per_epoch = num_val_data // self.cfg.batch_size
        idx = np.arange(num_val_data)
        np.random.shuffle(idx)
        l_value = 0.0
        if self.cfg.early_stopping:
            min_val_loss = 200.0
            val_count = 0
        acc = 0.0
        self.model.eval()
        for k in range(valiter_per_epoch):
            i, q, a = sample_batch_hard(j,
                    cfg.batch_size,
                    self.val_img_features,
                    self.val_img_id_list,
                    self.val_ques,
                    qa_data)


            q, i, a = q.to(device), i.to(device), a.to(device)

            logits = self.model.forward(i, q)
            pred = torch.argmax(logits.data, dim=1)

            acc += torch.sum(pred.data == a)
            loss = criterion(logits, a)
            l_value += loss.data

        l_value /= valiter_per_epoch
        self.writer.add_scalar('mfb/val_loss', l_value, e*trainiter_per_epoch+k)
        acc = acc.float() / float(num_train_data)
        self.writer.add_scalar('mfb/val_acc', acc.data, e*trainiter_per_epoch+k)

        if self.cfg.early_stopping:
            # validation loss
            #print("Validation loss for epoch %d is %f." % (e, l_value))
            if l_value < min_val_loss:
                min_val_loss = l_value
                val_count = 0
                best_model = copy.deepcopy(model.state_dict())
            else:
                val_count += 1

            #print("Validation acc for epoch %d is %f." % (e, acc))
            if val_count == 5:
                print("Early stopping!")
                sys.exit(0)

    def save(self):
        self.model.load_state_dict(self.best_model)
        torch.save(clean_state_dict(self.model.state_dict()))
