import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from data_loader import load_questions_answers
from tensorboardX import SummaryWriter
import progressbar as pb
import numpy as np
from attention_net import Attention_net, Attention_net_parallel
import copy
import argparse
import sys
import pdb
from utils import clean_state_dict, VqaDataset
import time

parser = argparse.ArgumentParser(description='PyTorch VQA Attention Network')
parser.add_argument('--data_dir', type=str, default='data',
                    help='data directory (default: data)')
parser.add_argument('--batch_size', type=int, default=64,
                    help='input batch size for training (default: 64)')
parser.add_argument('--num_epochs', type=int, default=60,
                    help='num of training epochs (default: 60)')
parser.add_argument('--use_soft', action='store_true', default=False,
                    help='using soft cross entropy')
args = parser.parse_args()

data_dir = args.data_dir
batch_size = args.batch_size
num_epochs = args.num_epochs
use_soft = args.use_soft

# Load QA Data
print("Reading QA DATA")
qa_data = load_questions_answers(data_dir=data_dir)
print("train questions", len(qa_data['train']))
print("val questions", len(qa_data['val']))
print("answer vocab", len(qa_data['answer_vocab']))
print("question vocab", len(qa_data['question_vocab']))
print("max question length", qa_data['max_question_length'])

# Define Data Loader
data_splits = ('train', 'val')

datasets = {x: VqaDataset(qa_data, x) for x in data_splits}

dataloaders = {x: DataLoader(datasets[x], batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False) 
                    for x in data_splits}

dataset_sizes = {x: len(datasets[x]) for x in data_splits}

print(dataset_sizes)

# [Discarded]
def sample_batch_soft(batch_no, batch_size, features, image_id_map, qa, split):
  si = (batch_no * batch_size)%len(qa)
  ei = min(len(qa), si + batch_size)
  n = ei - si
  sentence = np.ndarray( (n, qa_data['max_question_length']), dtype=int) # [N, 22]
  soft_answers = np.zeros((n, len(qa_data['answer_vocab'])), dtype=int) # [N, answer_vocab_size]
  answers = np.zeros((n,), dtype=int) # [N,]
  fc7 = torch.empty( (n,49,1024) ) # [N, 49, 1024]

  count = 0
  for i in range(si, ei):
    sentence[count,:] = qa[i]['question'][:]
    sparse_soft_answers = qa[i]['answers']
    idx = list(sparse_soft_answers.keys())
    probs = list(sparse_soft_answers.values())
    soft_answers[count,idx] = probs
    answers[count] = qa[i]['answer']
    fc7_index = image_id_map[ qa[i]['image_id'] ]
    fc7[count,:,:] = features[fc7_index, :, :]
    count += 1
  return fc7, torch.tensor(sentence), torch.tensor(soft_answers), torch.tensor(answers)

# Define model
model_name = 'Attention_net_parallel'
model = Attention_net_parallel(block_num=49, word_num=qa_data['max_question_length'], 
                    img_size=1024, vocab_size=len(qa_data['question_vocab']), 
                    embed_size=512, att_num=6, output_size=len(qa_data['answer_vocab']))

def soft_loss(preds, labels):
    s = F.softmax(labels, dim=1)
    res = torch.mean(torch.sum(-1.0 * s * torch.log(preds) - (1-s) * torch.log(1-preds), dim=1))
    return res

if use_soft:
    sample_batch = sample_batch_soft
    criterion = soft_loss
else:
    criterion = nn.CrossEntropyLoss()

#optimizer = torch.optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

num_params = 0
for param in model.parameters():
    num_params += param.numel()
print("Num parameters {}".format(num_params))

print('Load model to GPUs')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if torch.cuda.device_count() > 1:
    print("Use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)
model = model.to(device)

writer = SummaryWriter()

best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0
n_iter = 0
count = 0
since = time.time()
for epoch in range(num_epochs):
    print('-' * 20)
    print('Epoch {:4d}/{:4d}'.format(epoch, num_epochs - 1))

    # Each epoch has a train and val phase
    for phase in data_splits:
        if phase == dataloaders[phase]:
            model.train()  # Set model to train mode
        else:
            model.eval()   # Set model to val mode

        running_loss = 0.0
        running_corrects = 0

        pbar = pb.ProgressBar()
        for j, (image_features, questions, answers) in zip(pbar(range(len(dataloaders[phase]))), dataloaders[phase]):
            image_features = image_features.to(device)
            questions = questions.to(device)
            answers = answers.to(device)
            optimizer.zero_grad()
            
            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == data_splits[0]):
                logits, que_att, img_att = model(image_features, questions)
                preds = F.softmax(logits, dim=1)
                preds = preds.data.max(1)[1] # get the index of the max log-probability
                loss = criterion(logits, answers)
                # backward + optimize only if in training phase
                if phase == data_splits[0]:
                    writer.add_scalar(model_name+'/batch_loss', loss, n_iter)
                    n_iter += 1
                    loss.backward()
                    optimizer.step()
            
            # statistics
            running_loss += loss.item() * image_features.size(0)
            running_corrects += torch.sum(preds == answers.data)
        
        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]
        
        if phase == data_splits[0]:
            loss_train, acc_train = epoch_loss, epoch_acc
        else:
            loss_test, acc_test = epoch_loss, epoch_acc

        print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

        # deep copy the model
        if phase == data_splits[1] and epoch_acc > best_acc:
            count = 0
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        elif phase == data_splits[1] and epoch_acc <= best_acc:
            count += 1
    
    writer.add_scalars(model_name+'/epoch_loss', {'train': loss_train, 'test': loss_test}, epoch)
    writer.add_scalars(model_name+'/epoch_accuracy', {'train': acc_train, 'test': acc_test}, epoch)

    if count >= 5:
        print("Epoch = {}, test accuracy does not increase in five epochs, training stops".format(epoch))
        break

time_elapsed = time.time() - since
print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
print('Best val Acc: {:4f}'.format(best_acc))
    
# load best model weights
model.load_state_dict(best_model_wts)
torch.save(clean_state_dict(model.state_dict()), 'models/{}.pth'.format(model_name))
