import torch
import pickle
import os
from cfg import cfg
from solver import Solver
from data_loader import *
from mfb_coatt import MFB
from lstm_modules import *

qa_data = load_questions_answers()
cfg.vocab_size = len(qa_data['answer_vocab'])
train_ques_data = qa_data['training']['question']
val_ques_data = qa_data['validation']['question']

train_image_featrues, train_image_id_list = load_image_features()
val_image_featrues, val_image_id_list = load_image_features()

train_data = (train_ques_data, train_image_fatures, train_image_id_list)
val_data = (val_ques_data, val_image_features, val_image_id_list)

if cfg.model == 'mfb':
    model = MFB(cfg)
elif cfg.model == 'vislstm':
    model = VisLSTM(cfg)
else:
    print("model %s not supported." % cfg.model)

solver = MFBSolver(model,
        cfg,
        train_ques_data,
        train_image_features,
        train_image_id_list,
        val_ques_data,
        val_image_features,
        val_image_id_list)

solver.train()
solver.save()
print("Training done")

