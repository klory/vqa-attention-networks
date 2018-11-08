import torch
from torch.utils.data import Dataset
import torch
from os.path import join
import spacy
import numpy as np
import pdb

# define datasets
class VqaDataset(Dataset):
  def __init__(self, qa_data, split, cfg, feature_type='resnet152'):
    assert feature_type in ('resnet152', 'vgg19Fc'), 'feature_type does not exist'
    self.qa = qa_data[split]
    self.split = split
    self.feature_type = feature_type
    self.glove = cfg.glove
    self.glove_dict = dict()
    if self.glove:
      self.glove_model = spacy.load('en_vectors_web_lg')
      self.word_to_idx = qa_data['question_vocab']
      self.idx_to_word = dict()
      for w, i in self.word_to_idx.items():
        self.idx_to_word[i] = w
    self.soft_answer = cfg.soft_answer
    self.num_answer = cfg.num_answer

  def __getitem__(self, index, soft_answer=False):
    image_id = self.qa[index]['image_id']
    filepath = join('data/{}_{}'.format(self.feature_type, self.split), 'COCO_{}2014_{:012d}.npy'.format(self.split, image_id))
    image_features = np.load(filepath) # [2048, 14, 14]
    image_features = np.transpose(image_features, (1,2,0)) # 14, 14, 2018
    image_features = image_features.reshape(-1, image_features.shape[-1]) # 196, 2048
    question = self.qa[index]['question'] # [22]
    ques_length = self.qa[index]['ques_length']
    # process answer
    if not self.soft_answer:
      answer = self.qa[index]['answer'] # scalar
    else:
      answer = np.zeros((self.num_answer))
      soft_answer = self.qa[index]['answers']
      for a, v in soft_answer.items():
        answer[int(a)] = v

    if not self.glove:
      return torch.tensor(image_features, dtype=torch.float), torch.tensor(question, dtype=torch.long), torch.tensor(answer, dtype=torch.float), torch.tensor(ques_length, dtype=torch.long)
    else:
      max_len = len(question)
      glove_matrix = np.zeros((max_len, 300))
      for i in range(max_len):
        w = int(question[i])
        if w == 0:
          glove_matrix[i] = np.zeros((1, 300))
        else:
          if w not in self.glove_dict:
            self.glove_dict[w] = self.glove_model(self.idx_to_word[w]).vector
          glove_matrix[i] = self.glove_dict[w]
      return torch.tensor(image_features, dtype=torch.float), torch.tensor(question, dtype=torch.long), torch.tensor(answer, dtype=torch.float), torch.tensor(ques_length, dtype=torch.long), torch.tensor(glove_matrix, dtype=torch.float)

  def __len__(self):
    return len(self.qa)
