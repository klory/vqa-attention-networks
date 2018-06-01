from easydict import EasyDict as edict

cfg = edict()

cfg.shuffle = True # whether shuffle the training data or not durint training
cfg.image_first = True
# network configuration
cfg.hidden_dim = 512
cfg.emb_dim = 512
cfg.num_layers = 1
cfg.img_feature_dim = 196
cfg.img_feature_channel = 512

# training configureation
cfg.lr = 3e-4
cfg.num_epoch = 50
cfg.early_stopping = False
cfg.lr_decay = True
cfg.decay_rate = lr/40.0

# printing
cfg.print_freq = 600

cfg.model = 'mfb'
