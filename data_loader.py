import torch
from torch.utils.data import Dataset
import torch
from os.path import join
import numpy as np

# define datasets
class VqaDataset(Dataset):
  def __init__(self, qa_data, split, feature_type='resnet152'):
    assert feature_type in ('resnet152', 'vgg19Fc'), 'feature_type does not exist'
    self.qa = qa_data[split]
    self.split = split
    self.feature_type = feature_type

  def __getitem__(self, index):
    image_id = self.qa[index]['image_id']
    filepath = join('data/{}_{}'.format(self.feature_type, self.split), 'COCO_{}2014_{:012d}.npy'.format(self.split, image_id))
    image_features = np.load(filepath) # [2048, 14, 14]
    image_features = np.transpose(image_features, (1,2,0)) # 14, 14, 2018
    image_features = image_features.reshape(-1, image_features.shape[-1]) # 196, 2048
    question = self.qa[index]['question'] # [22]
    answer = self.qa[index]['answer'] # scalar
    return torch.tensor(image_features, dtype=torch.float), torch.tensor(question, dtype=torch.long), torch.tensor(answer, dtype=torch.long)

  def __len__(self):
    return len(self.qa)
