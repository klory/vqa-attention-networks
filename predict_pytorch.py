import torch
from torchvision import models
from extract_image_features import FeatureExtractor
from os.path import isfile, join
import re
from PIL import Image
from torchvision import transforms
import argparse
import data_loader
import numpy as np
from attention_net import *
import pdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default = 'data/cat.jpg',
                help='Image Path')
    parser.add_argument('--model_path', type=str, default = 'att1_hard.pth',
                help='Model Path')
    parser.add_argument('--data_dir', type=str, default='data',
                help='Data directory')
    parser.add_argument('--question', type=str, default='Which animal is this?',
                help='Question')
    
    

    args = parser.parse_args()

    print("Image:", args.image_path)
    print("Question:", args.question)

    # build up vgg image feature extractor
    Vgg19 = models.vgg19(pretrained=True)
    extract_list = [27, 36]
    extractor = FeatureExtractor(Vgg19.features, extract_list)
    extractor.eval()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    preprocess = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        normalize
    ])

    img = preprocess(Image.open(args.image_path))
    img = torch.unsqueeze(img, 0)
    fc7 = extractor(img)
    fc7 = fc7.permute(0, 2, 3, 1)
    fc7 = fc7.view(1, -1, fc7.shape[3])

    # get vocabulary to encode questions
    vocab_data = data_loader.get_question_answer_vocab(version=2, data_dir=args.data_dir)
    qvocab = vocab_data['question_vocab']
    q_map = { vocab_data['question_vocab'][qw] : qw for qw in vocab_data['question_vocab']}
    
    question_vocab = vocab_data['question_vocab']
    word_regex = re.compile(r'\w+')
    question_ids = np.zeros((1, vocab_data['max_question_length']), dtype = 'int32')
    question_words = re.findall(word_regex, args.question)
    base = vocab_data['max_question_length'] - len(question_words)
    for i in range(0, len(question_words)):
        if question_words[i] in question_vocab:
            question_ids[0][base + i] = question_vocab[ question_words[i] ]
        else:
            question_ids[0][base + i] = question_vocab['UNK']

    ans_map = { vocab_data['answer_vocab'][ans] : ans for ans in vocab_data['answer_vocab']}
    
    model = Attention_net()
    state_dict = torch.load(args.model_path)
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)
    model.eval()

    q_ids = torch.tensor(question_ids, dtype=torch.long)
    pred, _, _ = model(fc7, q_ids)

    print("Ans:", ans_map[pred.data.max(1)[1].numpy()[0]])

    answer_probab_tuples = [(pred[0][idx], idx) for idx in range(len(pred[0]))]
    answer_probab_tuples.sort()
    print("Top Answers")
    for i in range(5):
        print(ans_map[ answer_probab_tuples[i][1] ])

if __name__ == '__main__':
        main()
