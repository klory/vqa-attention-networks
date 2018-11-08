import torch
import torch.nn.init as init
import pickle
import os
from cfg import cfg
from solver import Solver
from utils import load_questions_answers
from hieCoAtten import HieCoAtten
from mhb_coAtt import MHBCoAtt, MHB
import argparse
import sys

import pdb

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', type=str, default='mhb', help='mhb|mhb_coAtt|hieCoAtten| (default=mhb)')
parser.add_argument('--version', type=int, default=2, help='vqa dataset version (1|2, default=2)')
parser.add_argument('--image_first', type=bool, default=0, help='whether to save image (default=0)')
parser.add_argument('--num_answer', type=int, default=1000, help='number of answer (default=1000)')
parser.add_argument('--mode', type=str, default='training', help='training | testing')
parser.add_argument('--glove', type=bool, default=0, help='whether or not to use glove embedding')
args = parser.parse_args()

cfg.model_name = args.model_name
cfg.image_first = args.image_first
cfg.version = args.version
cfg.num_answer = args.num_answer
cfg.mode = args.mode
cfg.glove = args.glove
if cfg.model_name == 'mhb' or cfg.model_name == 'mhb_coAtt':
  cfg.soft_answer = 1
else:
  cfg.soft_answer = 0

qa_data = load_questions_answers(image_first=cfg.image_first, version=cfg.version, num_ans=cfg.num_answer)

cfg.q_vocab_size = len(qa_data['question_vocab'])
cfg.a_vocab_size = len(qa_data['answer_vocab'])
print("q_vocab_size", cfg.q_vocab_size)
print("a_vocab_size", cfg.a_vocab_size)

num_train_data = len(qa_data['train'])

if cfg.model_name == 'mhb_coAtt':
  model = MHBCoAtt(cfg)
elif cfg.model_name == 'mhb':
  model = MHB(cfg)
elif cfg.model_name == 'hieCoAtten':
  model = HieCoAtten(cfg)
else:
  print("model %s not supported." % cfg.model_name)
  sys.exit(-1)

for name, param in model.named_parameters():
  if name.find('bias') == -1:
    init.xavier_uniform_(param)

if cfg.mode == 'testing':
  pre_trained = os.path.join(cfg.out_dir, cfg.model_name+'.pth')
  model.load_state_dict(torch.load(pre_trained))
  
solver = Solver(model, cfg, qa_data)

if cfg.mode == 'training':
  solver.train()
  solver.save()
  print("Training done")
else:
  print("Start to evaluate model: %s" % cfg.model_name)
  solver.val()
  print("Testing done")

