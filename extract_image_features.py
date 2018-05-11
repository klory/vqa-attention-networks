import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms

from scipy import misc
from os import listdir
from os.path import isfile, join
import data_loader
import argparse
import numpy as np
import pickle
import h5py
import time
import progressbar as pb
from PIL import Image

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', type=str, default='train',
                       help='train/val')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch Size')
    
    args = parser.parse_args()
    all_data = data_loader.load_questions_answers('word')
    if args.split == "train":
        qa_data = all_data['training']
    else:
        qa_data = all_data['validation']
    
    print("Total Questions", len(qa_data))
    image_ids = {}
    
    for qa in qa_data:
        image_ids[qa['image_id']] = 1
    
    image_id_list = [img_id for img_id in image_ids]
    print("Total Images", len(image_id_list))
    
    mymodel = models.vgg19(pretrained=True)
    extract_list = [27, 36]
    myexactor = FeatureExtractor(mymodel.features,extract_list)
    myexactor.eval()

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

    fc7 = np.ndarray( (len(image_id_list), 1024, 7, 7 ) )
    idx = 0

    while idx < len(image_id_list):
        start = time.clock()
        image_batch = torch.zeros( (args.batch_size, 3, 224, 224) )

        count = 0
        for i in range(0, args.batch_size):
            if idx >= len(image_id_list):
                break
            image_file = join(args.data_dir, 'vqa/%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
            img = Image.open(image_file)
            image_batch[i,:,:,:] = preprocess(img)
            idx += 1
            count += 1
        
        # pdb.set_trace()
        image_batch.to(device)
        fc7_batch = myexactor(image_batch)
        fc7[(idx - count):idx, :,:,:] = fc7_batch[0:count,:,:,:].data.cpu().numpy()
        end = time.clock()
        print("Time for batch 32 photos", end - start)
        print("Hours For Whole Dataset" , (len(image_id_list) * 1.0)*(end - start)/60.0/60.0/32.0)
        print("Images Processed", idx)
        
    print("Saving features")
    h5f_fc7 = h5py.File( join(args.data_dir, args.split + '.h5'), 'w')
    h5f_fc7.create_dataset('features', data=fc7)
    h5f_fc7.close()

    print("Saving image id list")
    h5f_image_id_list = h5py.File( join(args.data_dir, args.split + '_image_id_list.h5'), 'w')
    h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
    h5f_image_id_list.close()
    print("Done!")

if __name__ == '__main__':
    main()
