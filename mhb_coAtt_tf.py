import tensorflow as tf

class MHBCoAtt(object):
  def __init__(self, cfg):
    self.cfg = cfg

    self.weight_initializer = tf.contrib.layers.xavier_initializer()
    self.emb_initializer = tf.random_uniform_initializer(minval=-0.08, maxval=0.08)
    self.score_smm = list()
    self.event_smm = list()

  def _emb_layer(self, ques):
    with tf.variable_scope('ques_embedding') as scope:
      embedding = tf.get_variable('embedding_weight', [self.cfg.vocab_size, self.cfg.emb_dim], initializer=self.emb_initializer, trainable=self.is_training)
      embedded = tf.nn.embedding_lookup(embedding, ques, name='embedding')
      self.score_smm.append(embedded)
      return embedded

  def _dense_layer(self):
    pass

  def _conv_layer(self, inputs, dim_out, name):
    dim_in = inputs.get_shape().as_list()[1]
    with tf.variable_scope(name) as scope:
      w = tf.get_variable(name+'_weight', [1, 1, dim_in, dim_out])
      b = tf.get_variable(name+'_bias', [dim_out])
      output = tf.nn.relu(tf.nn.conv2d(inputs, w, 1, 'VALID', data_format='NCHW') + b, name=name+'_output')
      self.score_smm.append(output)
      return output

  def build_graph(self, mode='training'):
    self.is_training = (mode == 'trainging')
    img = tf.placeholder(tf.float32, [None, 196, 2048], name='image')
    ques = tf.placeholder(tf.int32, [None, cfg.max_len], name='question')
    if self.cfg.glove:
      glove_matrix = tf.placeholder(tf.float32, [None, 300], name='glove_matrix')

    ques_embedded = tf.tanh(self._emb_layer(ques), name='ques_emb_tanh')
    if self.cfg.glove:
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cfg.hidden_dim*2)
    else:
      lstm_cell = tf.contrib.rnn.BasicLSTMCell(self.cfg.hidden_dim)

    with tf.variable_scope('lstm') as scope:
      outputs = []
      c, h = lstm_cell.zero_state(self.cfg.batch_size, dtype=tf.float32)
      for t in range(self.cfg.max_len):
        if self.cfg.glove:
          o, (c, h) = lstm_cell(tf.concat([ques_embedded, glove_matrix], 1), [c, h])
        else:
          o, (c, h) = lstm_cell(ques_embedded, [c, h])
        outputs.append(tf.expand_dims(o, 2)) # N, H, 1
        self.score_smm.append(o)

    ques_feature = tf.concat(outputs, 2) # N, H, T
    ques_feature = tf.contrib.layer.dropout(
        ques_feature,
        keep_prob = 0.5,
        is_training = self.is_training,
        name='ques_feature') # N, T, H
    ques_expand = tf.expand_dims(ques_feature, axis=3, name='ques_expand') # N, H, T, 1

    ques_att_1 = self._conv_layer(ques_expand, 512, 'ques_att_1') # N, 512, T, 1
    self.score_smm.append(ques_att_1)
    ques_att_2 = self._conv_layer(ques_att_1, 2, 'ques_att_2') # N, 2, T, 1
    self.score_smm.append(ques_att_2)

    ques_att_softmax = tf.nn.softmax(ques_att_2, dim=2, name='ques_att_softmax')

    ques_att_list = []
    for i in range(2):
      att = tf.slice(ques_att_softmax, [0,i,0,0], [-1, 1, -1, -1]) # N, 1, T, 1
      ques_att_list.append(tf.reduce_sum(tf.multiply(att, ques_expand), axis=2, keepdims=True)) # N, H, 1, 1

    ques_att_feature = tf.concat(ques_att_list, 1, name='ques_att_feature') # N, 2*H, 1, 1

    N, H = ques_att_feature.get_shape().as_list()[:2]
    

