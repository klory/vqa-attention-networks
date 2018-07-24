import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

import os
from os import listdir
from os.path import isfile, join
import data_loader
import argparse
import numpy as np
import time
from tqdm import tqdm
from PIL import Image
import pdb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class FeaturesExtractor(nn.Module):
    def __init__(self):
        super(FeaturesExtractor, self).__init__()
        resnet = models.resnet152(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])

    def forward(self, images):
        return self.features(images)

def main():
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--split', type=str, default='train',
                       help='train|val')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch Size')
    parser.add_argument('--feature_type', type=str, default='resnet152',
                        help='features to extract: resnet152')
    
    args = parser.parse_args()
    output_dir = args.output_dir
    batch_size = args.batch_size
    feature_type = args.feature_type

    saving_dir = join(output_dir, '{}_{}'.format(feature_type, args.split))
    if not os.path.exists(saving_dir):
        os.makedirs(saving_dir)
    
    image_names = [join('data/vqa/{}2014'.format(args.split), x) for x in sorted(listdir('data/vqa/{}2014'.format(args.split))) if x[0]!='.']
    print('total image count', len(image_names))

    print('define model')
    myextractor = FeaturesExtractor().to(device)
    myextractor.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((448,448)),
        transforms.ToTensor(),
        normalize
    ])
    
    print('start extracting features')
    idx = 0
    idx_saving = 0
    with tqdm(total=len(image_names)) as pbar:
        while idx < len(image_names):
            image_batch = torch.zeros((batch_size, 3, 448, 448))
            count = 0
            for i in range(0, batch_size):
                if idx >= len(image_names):
                    break
                img = Image.open(image_names[idx])
                image_batch[i,:,:,:] = preprocess(img)
                idx += 1
                count += 1

            image_batch = image_batch.to(device)
            features_batch = myextractor(image_batch)
            features = features_batch[0:count].data.cpu().numpy()

            for i in range(count):
                filepath = join(saving_dir, image_names[idx_saving].split('/')[-1].split('.')[0])
                np.save(filepath, features[i])
                idx_saving += 1
                pbar.update(1)

if __name__ == '__main__':
    main()
