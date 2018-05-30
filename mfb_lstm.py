import torch
import torch.nn as nn
import numpy as np
from data_loader import *
from tensorboardX import SummaryWriter
from mfb_coatt import MFB
from tqdm import tqdm
import sys
import copy
import pdb
from utils import clean_state_dict
from easydict import EasyDict as edict


data_dir = './data/'

cfg = edict()
cfg.batch_size = 1

# loading_data
qa_data = load_questions_answers()
question_vocab = qa_data['question_vocab']

cfg.vocab_size = len(question_vocab)
cfg.hidden_dim = 512
cfg.emb_dim = 512
cfg.num_layers = 1 # one layer LSTM
cfg.img_feature_dim = 196
cfg.img_feature_channel = 512

model = MFB(cfg)
img_feature = torch.tensor(np.random.random([1, 196, 512]), dtype=torch.float)
questions = torch.tensor(np.random.randint(0, high=10, size=[1, 22]), dtype=torch.long)
question_len = torch.tensor([21], dtype=torch.long)
pdb.set_trace()
model.forward(img_feature, questions, question_len)
print(model)


"""
# tensorboardX graph
writer = SummaryWriter()
model = VisLSTM(vocab_size=vocab_size)

dummy_ques = torch.randint(100, size=[2, 22], dtype=torch.long)
dummy_img = torch.rand([2, 4096])
writer.add_graph(model, (dummy_ques, dummy_img, ))
"""

writer = SummaryWriter()

print("Loading training image feature...")
train_img_features, train_img_id_list = load_image_features_small(data_dir, 'train')

print("Loading validation image feature...")
val_img_features, val_img_id_list = load_image_features_small(data_dir, 'val')

# process data
print("Loading training question data")
train_ques = qa_data['training']
print("Loading validation question data")
val_ques = qa_data['validation']


def process_data(ques_dict, img_features, img_list):
    ques = list()
    img = list()
    ans = list()
    for q in ques_dict:
        img_id = q['image_id']
        img_idx = np.where(img_list == img_id)[0]
        ques.append(q['question'])
        img.append(img_features[img_idx])
        ans.append(q['answer'])

    ques = torch.tensor(np.vstack(ques), dtype=torch.long)
    img = torch.tensor(np.vstack(img), dtype=torch.float)
    ans = torch.tensor(np.array(ans), dtype=torch.long)

    return ques, img, ans

# process training data
print("Proccessing training question data")
train_ques, train_img, train_ans = process_data(train_ques, train_img_features, train_img_id_list)
# process validation data
print("Proccessing validation question data")
val_ques, val_img, val_ans = process_data(val_ques, val_img_features, val_img_id_list)

print("Model building")
model = MFB(cfg)
print(model)

"""

lr = 1e-3
early_stopping = False
lr_decay = True
decay_rate = lr/40
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=np.arange(10,50), gamma=0.95)
criterion = nn.CrossEntropyLoss()

num_epoch = 50

num_train_data = len(train_ques)
num_val_data = len(val_ques)
trainiter_per_epoch = num_train_data // batch_size
valiter_per_epoch = num_val_data // batch_size

# print loss every 100 iters
print_freq = 600
min_val_loss = 200.0
val_count = 5

total_train_iter = num_epoch * trainiter_per_epoch

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model.to(device)

best_model = copy.deepcopy(model.state_dict())

if torch.cuda.is_available():
    device = torch.device("cuda")
with tqdm(total=total_train_iter) as pbar:
    for e in range(num_epoch):
        idx = np.arange(num_train_data)
        np.random.shuffle(idx)
        acc = 0.0
        model.train()
        for j in range(1, trainiter_per_epoch+1):
            indices = idx[(j-1)*batch_size: j*batch_size]

            q, i, a = train_ques[indices], train_img[indices], train_ans[indices]
            q, i, a = q.to(device), i.to(device), a.to(device)

            logits, _ = model.forward(q, i, first_words=False)
            loss = criterion(logits, a)
            l_value = loss.data

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred = torch.argmax(logits.data, dim=1)
            acc += torch.sum(pred.data == a)

            if j%print_freq == 0:
                #print("epoch: %d, iter: %d, loss: %f" % (e, j, l_value))
                writer.add_scalar('visMulLSTM/train_loss', l_value, e*trainiter_per_epoch+j)
                pbar.update(print_freq)
        acc = acc.float() / float(num_train_data)
        #print("Accuracy for epoch %d is %f." % (e, acc))
        writer.add_scalar('visMulLSTM/train_acc', acc.data, e*trainiter_per_epoch+j)
        if lr_decay and e>9:
            scheduler.step()

        idx = np.arange(num_val_data)
        np.random.shuffle(idx)
        l_value = 0.0
        acc = 0.0
        model.eval()
        for k in range(valiter_per_epoch):
            indices = idx[k*batch_size: (k+1)*batch_size]

            q, i, a = val_ques[indices], val_img[indices], val_ans[indices]
            q, i, a = q.to(device), i.to(device), a.to(device)

            logits, _ = model.forward(q, i)
            pred = torch.argmax(logits.data, dim=1)

            acc += torch.sum(pred.data == a)
            loss = criterion(logits, a)
            l_value += loss.data

        l_value /= valiter_per_epoch
        writer.add_scalar('visMulLSTM/val_loss', l_value, e*trainiter_per_epoch+k)
        acc = acc.float() / float(num_train_data)
        writer.add_scalar('visMulLSTM/val_acc', acc.data, e*trainiter_per_epoch+k)

        if early_stopping:
            # validation loss
            #print("Validation loss for epoch %d is %f." % (e, l_value))
            if l_value< min_val_loss:
                min_val_loss = l_value
                val_count = 0
                best_model = copy.deepcopy(model.state_dict())
            else:
                val_count += 1

            #print("Validation acc for epoch %d is %f." % (e, acc))
            if val_count == 5:
                print("Early stopping!")
                sys.exit(0)

model.load_state_dict(best_model)
torch.save(clean_state_dict(model.state_dict()))
"""
