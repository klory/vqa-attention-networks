import numpy as np
import torch
import pdb
# courtesy from https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3

from torch.data import Dataset
import torch

# define datasets
class VqaDataset(Dataset):
  def __init__(self, qa_data, split):
    self.qa = qa_data[split]
    self.split = split

  def __getitem__(self, index):
    image_id = self.qa[index]['image_id']
    filepath = join('data/{}_blocks_vgg19'.format(self.split), 'COCO_{}2014_{:012d}.npy'.format(self.split, image_id))
    image_features = np.load(filepath) # 1024, 7, 7
    image_features = np.transpose(image_features, (1,2,0)) # 7, 7, 1024
    image_features = image_features.reshape(-1, image_features.shape[-1]) # 49, 1024
    question = self.qa[index]['question'] # 
    answer = self.qa[index]['answer'] # scalar
    return torch.tensor(image_features, dtype=torch.float), torch.tensor(question, dtype=torch.long), torch.tensor(answer, dtype=torch.long)

  def __len__(self):
    return len(self.qa)

def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def sample_batch_hard(batch_no, batch_size, features, image_id_map, qa, qa_data):
    pdb.set_trace()
    si = (batch_no * batch_size)%len(qa)
    ei = min(len(qa), si + batch_size)
    n = ei - si
    sentence = np.ndarray( (n, qa_data['max_question_length']), dtype=int) # [N, 22]
    answers = np.zeros((n,), dtype=int) # [N,]
    fc7 = torch.empty( (n,49,1024) ) # [N, 49, 1024]
    count = 0
    for i in range(si, ei):
        sentence[count,:] = qa[i]['question'][:]
        answers[count] = qa[i]['answer']
        fc7_index = image_id_map[ qa[i]['image_id'] ]
        fc7[count,:,:] = features[fc7_index, :, :]
        count += 1
    return fc7, torch.tensor(sentence), torch.tensor(answers)

def sample_batch_soft(batch_no, batch_size, features, image_id_map, qa):
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
