import torch
import torch.nn as nn
import numpy as np
from data_loader import *
from tensorboardX import SummaryWriter
from lstm_modules import VisLSTM
import pdb


data_dir = './data/'
batch_size = 64

# loading_data
qa_data = load_questions_answers_small(num_que=5000)
question_vocab = qa_data['question_vocab']

vocab_size = len(question_vocab)
hidden_dimension = 512
embed_dimension = 512

"""
# tensorboardX graph
writer = SummaryWriter()
model = VisLSTM(vocab_size=vocab_size)

dummy_ques = torch.randint(100, size=[2, 22], dtype=torch.long)
dummy_img = torch.rand([2, 4096])
writer.add_graph(model, (dummy_ques, dummy_img, ))
"""


writer = SummaryWriter()

train_img_features, train_img_id_list = load_image_features_small(data_dir, 'train', num_que=5000)
val_img_features, val_img_id_list = load_image_features_small(data_dir, 'train', num_que=5000)

# process data
train_ques = qa_data['training']
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
train_ques, train_img, train_ans = process_data(train_ques, train_img_features, train_img_id_list)
# process validation data
val_ques, val_img, val_ans = process_data(val_ques, val_img_features, val_img_id_list)

model = VisLSTM(vocab_size=vocab_size)


lr = 1e-4
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

num_epoch = 40

num_train_data = len(train_ques)
num_val_data = len(val_ques)
trainiter_per_epoch = num_train_data // batch_size
valiter_per_epoch = num_val_data // batch_size

# print loss every 100 iters
print_feq = 100
min_val_loss = 200.0
val_count = 5
for e in range(num_epoch):
    idx = np.arange(num_train_data)
    np.random.shuffle(idx)
    acc = 0.0
    model.train()
    for j in range(1, trainiter_per_epoch+1):
        indices = idx[(j-1)*batch_size: j*batch_size]

        q, i, a = train_ques[indices], train_img[indices], train_ans[indices]

        logits, _ = model.forward(q, i)
        loss = criterion(logits, a)
        l_value = loss.data

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = torch.argmax(logits.data, dim=1)
        acc += torch.sum(pred.data == a)

        if j%100 == 0:
            print("epoch: %d, iter: %d, loss: %f" % (e, j, l_value))
            writer.add_scalar('visLSTM/train_loss', l_value, e*iter_per_epoch+j)
    acc = acc / float(num_train_data)
    print("Accuracy for epoch %d is %f." % (e, acc))
    writer.add_scalar('visLSTM/train_acc', acc.data, e*trainiter_per_epoch+j)

    # validation loss
    idx = np.arange(num_val_data)
    np.random.shuffle(idx)
    l_value = 0.0
    model.eval()
    for k in range(valiter_per_epoch):
        indices = idx[j*batch_size: (j+1)*batch_size]

        q, i, a = val_ques[indices], val_img[indices], val_ans[indices]
        logits, _ = model.forward(q, i)
        loss = criterion(logits, a)
        l_value += loss.data

    l_value /= valiter_per_epoch
    if l_value< min_val_loss:
        min_val_loss = l_value
        val_count = 0
    else:
        val_count += 1
