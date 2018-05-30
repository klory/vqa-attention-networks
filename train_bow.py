import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_questions_answers, load_image_features, load_image_features_small, load_questions_answers_small, load_questions_answers_small, load_image_features_4096
from tensorboardX import SummaryWriter
import progressbar as pb
import numpy as np
from bow_net import Bow_net
import copy
import argparse
import sys
import pdb
from utils import clean_state_dict

parser = argparse.ArgumentParser(description='PyTorch VQA Attention Network')
parser.add_argument('--data_dir', type=str, default='data',
                    help='data directory (default: data)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epoch', type=int, default=60,
                    help='num of training epochs (default: 60)')
parser.add_argument('--use_soft', action='store_true', default=False,
                    help='using soft cross entropy')
args = parser.parse_args()

data_dir = args.data_dir
batch_size = args.batch_size
num_epoch = args.num_epoch
use_soft = args.use_soft

net_name = 'bow_net'

# Load QA Data
print("Reading QA DATA")
qa_data = load_questions_answers(token_type='word', version=2, data_dir=data_dir)
print("train questions", len(qa_data['training']))
print("val questions", len(qa_data['validation']))
print("answer vocab", len(qa_data['answer_vocab']))
print("question vocab", len(qa_data['question_vocab']))
print("max question length", qa_data['max_question_length'])

print("Reading Image DATA")
train_image_features, train_image_id_list = load_image_features_4096(data_dir, 'train')
print("train image features_4096", train_image_features.shape)
print("train image_id_list_4096", train_image_id_list.shape)
val_image_features, val_image_id_list = load_image_features_4096(data_dir, 'val')
print("val image features_4096", val_image_features.shape)
print("val image_id_list_4096", val_image_id_list.shape)

train_image_features = torch.tensor(train_image_features)
val_image_features = torch.tensor(val_image_features)

# Define Data Loader
def sample_batch_hard(batch_no, batch_size, features, image_id_map, qa, split):
  si = (batch_no * batch_size)%len(qa)
  ei = min(len(qa), si + batch_size)
  n = ei - si
  sentence = np.ndarray( (n, qa_data['max_question_length']), dtype=int) # [N, 22]
  answers = np.zeros((n,), dtype=int) # [N,]
  fc7 = torch.empty( (n,4096) ) # [N, 4096]

  count = 0
  for i in range(si, ei):
    sentence[count,:] = qa[i]['question'][:]
    answers[count] = qa[i]['answer']
    fc7_index = image_id_map[ qa[i]['image_id'] ]
    fc7[count,:] = features[fc7_index, :]
    count += 1
  return fc7, torch.tensor(sentence), torch.tensor(answers), torch.tensor(answers)


# Define Data Loader 
def sample_batch_soft(batch_no, batch_size, features, image_id_map, qa, split):
  si = (batch_no * batch_size)%len(qa)
  ei = min(len(qa), si + batch_size)
  n = ei - si
  sentence = np.ndarray( (n, qa_data['max_question_length']), dtype=int) # [N, 22]
  soft_answers = np.zeros((n, len(qa_data['answer_vocab'])), dtype=int) # [N, answer_vocab_size]
  answers = np.zeros((n,), dtype=int) # [N,]
  fc7 = torch.empty( (n,4096) ) # [N, 4096]

  count = 0
  for i in range(si, ei):
    sentence[count,:] = qa[i]['question'][:]
    sparse_soft_answers = qa[i]['answers']
    idx = list(sparse_soft_answers.keys())
    probs = list(sparse_soft_answers.values())
    soft_answers[count,idx] = probs
    answers[count] = qa[i]['answer']
    fc7_index = image_id_map[ qa[i]['image_id'] ]
    fc7[count, :] = features[fc7_index, :]
    count += 1
  return fc7, torch.tensor(sentence), torch.tensor(soft_answers), torch.tensor(answers)

train_image_id_map = {image_id: i for i, image_id in enumerate(train_image_id_list)}
val_image_id_map = {image_id: i for i, image_id in enumerate(val_image_id_list)}

model = Bow_net(output_size=len(qa_data['answer_vocab']))

# model = Attention_net(block_num=train_image_features.size(1), word_num=qa_data['max_question_length'], img_size=train_image_features.size(2), vocab_size=len(qa_data['question_vocab']), embed_size=512, att_num=6, output_size=len(qa_data['answer_vocab']))
def soft_loss(preds, labels):
    s = F.softmax(labels, dim=1)
    res = torch.mean(torch.sum(-1.0*s*torch.log(preds) - (1-s)*torch.log(1-preds), dim=1))
    return res

if use_soft:
    sample_batch = sample_batch_soft
    criterion = soft_loss
else:
    sample_batch = sample_batch_hard
    criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)

writer = SummaryWriter()
best_model_wts = copy.deepcopy(model.state_dict())

train_qa_data = qa_data['training']
val_qa_data = qa_data['validation']

batchs_train = len(train_qa_data) // batch_size + 1
batchs_val = len(val_qa_data) // batch_size + 1

n_iter_train = 0
count = 0
best_val_loss = 200.0
best_val_acc = 0.0
for epoch in range(num_epoch):
    pbar = pb.ProgressBar()

    # Train
    model.train()
    loss_value = 0.0
    correct = 0.0
    for j in pbar(range(batchs_train)):
        img_features, que_features, soft_answers, answers = sample_batch(j, batch_size, train_image_features, train_image_id_map, train_qa_data, 'train')
        
        img_features = img_features.to(device)
        que_features = que_features.to(device)
        soft_answers = soft_answers.to(device)
        answers = answers.to(device)
        
        output = model(img_features, que_features)
        loss = criterion(output, soft_answers)
        loss_value += loss.data * answers.size(0)

        pred = F.softmax(output, dim=1)
        pred = pred.data.max(1)[1] # get the index of the max log-probability
        acc = pred.eq(soft_answers.data).cpu().sum()
        correct += acc

        writer.add_scalar(net_name+'_hard/train_loss_per_iter', loss.data, n_iter_train)
        writer.add_scalar(net_name+'_hard/train_acc_per_iter', acc, n_iter_train)
        n_iter_train += 1

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    train_epoch_loss = loss_value / len(train_qa_data)
    train_epoch_acc = correct.float() / len(train_qa_data)
    print("\nTrain epoch {}, loss {}, acc {}".format(epoch, train_epoch_loss, train_epoch_acc))

    if epoch > 20 and epoch % 10 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] *= 0.5
    
#     for module in model.modules():
#         if module.__class__.__name__.find("BatchNorm") > -1:
#             module.train()
#             # BatchNorm for some reasons is not stable in eval

    pbar = pb.ProgressBar()

    # Evaluate
    model.eval()
    loss_value = 0.0
    correct = 0.0
    
    for j in pbar(range(batchs_val)):
        img_features, que_features, soft_answers, answers = sample_batch(j, batch_size, val_image_features, val_image_id_map, val_qa_data, 'val')
        
        img_features = img_features.to(device)
        que_features = que_features.to(device)
        soft_answers = soft_answers.to(device)
        answers = answers.to(device)
        
        output = model(img_features, que_features)

        loss = criterion(output, soft_answers)
        loss_value += loss.data * answers.size(0)

        pred = F.softmax(output, dim=1)
        pred = output.data.max(1)[1] # get the index of the max log-probability
        acc = pred.eq(answers.data).cpu().sum()
        correct += acc

    val_epoch_loss = loss_value / len(train_qa_data)
    val_epoch_acc = correct.float() / len(val_qa_data)
    print("\nValidation epoch {}, loss {}, acc {}".format(epoch, val_epoch_loss, val_epoch_acc))
    
    writer.add_scalars(net_name+'_hard/loss', {'train_per_epoch': train_epoch_loss, 'val_per_epoch': val_epoch_loss}, epoch)
    writer.add_scalars(net_name+'_hard/accuracy', {'train_per_epoch': train_epoch_acc, 'val_per_epoch': val_epoch_acc}, epoch)
    
    if val_epoch_loss < best_val_loss:
        best_val_loss = val_epoch_loss
        best_val_acc = val_epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        count = 0
    else:
        count += 1
        if count >= 5:
            print('Validation loss does not decrease in 5 epochs, training stops')
            break
    
    
# load best model weights
model.load_state_dict(best_model_wts)
torch.save(clean_state_dict(model.state_dict()), net_name+'_hard.pth')
