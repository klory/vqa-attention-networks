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
import progressbar as pb
from PIL import Image
import pdb

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor,self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
        self.maxpool = torch.nn.MaxPool2d(2, stride=2)

    def forward(self, x):
        outputs = []
        for i, module in enumerate(self.submodule):
            x = module(x)
            if i in self.extracted_layers:
                outputs += [x]
        
        x0 = self.maxpool(outputs[0])
        x1 = outputs[1]
        output = torch.cat([x0, x1], 1)
        return output

def main():
    parser = argparse.ArgumentParser(description='extract image features')
    parser.add_argument('--split', type=str, default='train',
                       help='train|val')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch Size')
    
    args = parser.parse_args()
    data_dir = args.data_dir
    output_dir = args.output_dir
    batch_size = args.batch_size

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    all_qa_data = data_loader.load_questions_answers(data_dir=data_dir)
    qa_data = all_qa_data[args.split]
    
    print("Total Questions", len(qa_data))
    
    image_ids = {}
    for qa in qa_data:
        image_ids[qa['image_id']] = 1
    
    image_id_list = [img_id for img_id in image_ids]
    print("Total Images", len(image_id_list))

    print("Saving image id list")
    features_filename = join(output_dir, args.split+'_image_id_list.txt')
    temp = [join(output_dir, 'train_blocks_vgg19', 'COCO_{}2014_{:012d}'.format(args.split, img_id)) for img_id in image_id_list]
    temp = '\n'.join(temp)
    with open(features_filename, 'w') as f:
        f.write(temp)

    print('define model')
    mymodel = models.vgg19(pretrained=True)
    extract_layer_ids = [27, 36]
    myexactor = FeatureExtractor(mymodel.features,extract_layer_ids)
    myexactor.eval()

    print("load model to GPU")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        myexactor = nn.DataParallel(myexactor)
    myexactor.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    print('start extracting features')
    idx = 0
    idx_saving = 0
    with pb.ProgressBar(max_value=len(image_id_list)) as bar:
        while idx < len(image_id_list):
            image_batch = torch.zeros( (batch_size, 3, 224, 224) )
            count = 0
            for i in range(0, batch_size):
                if idx >= len(image_id_list):
                    break
                image_file = join(args.data_dir, 'vqa/%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
                img = Image.open(image_file)
                image_batch[i,:,:,:] = preprocess(img)
                idx += 1
                count += 1

            image_batch.to(device)
            features_batch = myexactor(image_batch)
            features = features_batch[0:count,:,:,:].data.cpu().numpy()
            for i in range(count):
                filepath = join(output_dir, 'train_blocks_vgg19', 'COCO_{}2014_{:012d}'.format(args.split, image_id_list[idx_saving]))
                np.save(filepath, features[i, :, :, :])
                idx_saving += 1
                bar.update(idx_saving)

if __name__ == '__main__':
    main()
