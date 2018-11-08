from easydict import EasyDict as edict

cfg = edict()

cfg.shuffle = True # whether to shuffle the data
cfg.num_workers = 4 # dataloader processes

# network configuration
cfg.hidden_dim = 1024
cfg.emb_dim = 300
cfg.num_layers = 1

cfg.glove = True
cfg.soft_answer = False

# training configureation
cfg.batch_size = 64
cfg.lr = 7e-4
cfg.num_epoch = 18
cfg.early_stopping = False
cfg.lr_decay = True
cfg.decay_rate = 0.5 # for learning rate
cfg.decay_step = 40000
cfg.img_feature_channel = 2048

cfg.image_first = True
cfg.feature_type = 'resnet152'
cfg.img_feature_dim = 196

#summary interval(seconds, not used)
cfg.smm_interval = 180

cfg.out_dir = './models'
cfg.results = './results'
