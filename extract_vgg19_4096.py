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
import h5py

class MyExtracter(nn.Module):
    def __init__(self):
        super(MyExtracter, self).__init__()
        self.vgg = models.vgg19(pretrained=True)
        self.vgg.classifier = nn.Sequential(
            *(self.vgg.classifier[i] for i in range(6)))

    def forward(self, images):
        return self.vgg(images)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_small', action='store_true', default=False,
                        help='use small dataset')
    parser.add_argument('--split', type=str, default='train',
                       help='train/val')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Data directory')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch Size')

    args = parser.parse_args()
    output_dir = args.output_dir
    use_small = args.use_small
    batch_size = args.batch_size
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_que = 20000
    if use_small:
        all_data = data_loader.load_questions_answers_small('word', num_que=num_que)
    else:
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
    
    mymodel = MyExtracter()
    mymodel.eval()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        mymodel = nn.DataParallel(mymodel)
    mymodel.to(device)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    N = len(image_id_list)
    fc = np.zeros((N, 4096))
    idx = 0
    with pb.ProgressBar(max_value=N//batch_size+1) as bar:
        n_batch = 0
        while idx < N:
            image_batch = torch.zeros( (batch_size, 3, 224, 224) )

            count = 0
            for i in range(0, batch_size):
                if idx >= N:
                    break
                image_file = join(args.data_dir, 'vqa/%s2014/COCO_%s2014_%.12d.jpg'%(args.split, args.split, image_id_list[idx]) )
                img = Image.open(image_file)
                image_batch[i,:,:,:] = preprocess(img)
                idx += 1
                count += 1

            image_batch = image_batch.to(device)
            fc_batch = mymodel(image_batch)
            # pdb.set_trace()
            fc[(idx - count):idx, :] = fc_batch[0:count,:].data.cpu().numpy()
            bar.update(n_batch)
            n_batch += 1
            
        print("Saving features")
        features_filename = join(args.output_dir, args.split+'_'+str(num_que)+'.h5') if use_small \
                            else join(args.output_dir, args.split + '_4096.h5')
        h5f_fc7 = h5py.File(features_filename, 'w')
        h5f_fc7.create_dataset('features', data=fc)
        h5f_fc7.close()

        print("Saving image id list")
        features_filename = join(args.output_dir, args.split+'_image_id_list_'+str(num_que)+'.h5') if use_small \
                            else join(args.output_dir, args.split + '_image_id_list_4096.h5')
        h5f_image_id_list = h5py.File(features_filename, 'w')
        h5f_image_id_list.create_dataset('image_id_list', data=image_id_list)
        h5f_image_id_list.close()
        print("Done!")

if __name__ == '__main__':
    main()
