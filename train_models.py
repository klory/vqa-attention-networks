import torch
import torch.nn.init as init
import pickle
import os
from cfg import cfg
from solver import Solver
from utils import load_questions_answers
from hieCoAtten import HieCoAtten
from mfb import MFB
import argparse
import sys

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='mfb', help='mfb|hieCoAtten|mfb-multilayer (default=mfb)')
parser.add_argument('--version', type=int, default=2, help='vqa dataset version (1|2, default=2)')
parser.add_argument('--image_first', type=bool, default=0, help='whether to save image (default=0)')
parser.add_argument('--num_answer', type=int, default=1000, help='number of answer (default=1000)')
args = parser.parse_args()

cfg.model_name = args.model_name
cfg.image_first = args.image_first
cfg.version = args.version
cfg.num_answer = args.num_answer

qa_data = load_questions_answers(image_first=cfg.image_first, version=cfg.version, num_ans=cfg.num_answer)

cfg.q_vocab_size = len(qa_data['question_vocab'])
cfg.a_vocab_size = len(qa_data['answer_vocab'])
print("q_vocab_size", cfg.q_vocab_size)
print("a_vocab_size", cfg.a_vocab_size)

num_train_data = len(qa_data['train'])

if cfg.model_name == 'mfb' or 'mfb-multilayer':
  model = MFB(cfg)
elif cfg.model_name == 'hieCoAtten':
  model = HieCoAtten(cfg)
else:
  print("model %s not supported." % cfg.model_name)
  sys.exit(-1)

for name, param in model.named_parameters():
  if name.find('bias') == -1:
    init.xavier_uniform_(param)
solver = Solver(model, cfg, qa_data)
solver.train()
solver.save()
print("Training done")

