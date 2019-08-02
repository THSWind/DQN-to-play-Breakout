from agent_dir.agent import Agent
from colors import *
from tqdm import *
from collections import namedtuple

import tensorflow as tf
import numpy as np
import os, sys
import random

SEED = 11037
random.seed(SEED)
np.random.seed(SEED)
tf.set_random_seed(SEED)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
print(config)
stages = ["[OBSERVE]", "[EXPLORE]", "[TRAIN]"]

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class Agent_DQN(Agent):
  def __init__(self, env, args):
    """
    Initialize every things you need here.
    For example: building your model
    """
    super(Agent_DQN,self).__init__(env)
    self.args = args
    self.batch_size = args.batch_size
    self.lr = args.learning_rate
    self.dueling = args.dueling_dqn
    self.double = args.double_dqn
    self.gamma = args.gamma_reward_decay
    self.n_actions = env.action_space.n # = 4
    self.state_dim = env.observation_space.shape[0] # 84
    self.output_logs = args.output_logs # 'loss.csv'
    self.global_step = tf.Variable(0, trainable=False)
    self.add_global = self.global_step.assign_add(1)
    self.step = 0
    self.stage = ""
    self.memory = self.ReplayMemory(self.args.replay_memory_size)
    self.TRAINING = tf.Variable(initial_value=True, dtype=tf.bool, trainable=False)

    self.s = tf.placeholder(tf.float32, [None, 84, 84, 4], 
      name='s')
    self.s_ = tf.placeholder(tf.float32, [None, 84, 84, 4], 
      name='s_')
    self.y_input = tf.placeholder(tf.float32, [None]) 
    self.action_input = tf.placeholder(tf.float32, [None, self.n_actions])

    self.q_eval = self.build_net(self.s, 'eval_net') # online Q
    self.q_target = self.build_net(self.s_, 'target_net') # target Q

    # self.q_eval = self.build_net_resnet(self.s, 'eval_net') # online Q
    # self.q_target = self.build_net_resnet(self.s_, 'target_net') # target Q

    # self.q_eval = self.build_net_alex(self.s, 'eval_net') # online Q
    # self.q_target = self.build_net_alex(self.s_, 'target_net') # target Q
    
    self.train_summary = []
    self.buildOptimizer()
    self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
    self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

    self.replace_target_op = [tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)] 

    self.epsilon = self.args.epsilon_start

    self.ckpts_path = self.args.save_dir + "dqn.ckpt"
    self.saver = tf.train.Saver(max_to_keep = 3)
    self.sess = tf.Session(config=config)
    
    self.summary_writer = tf.summary.FileWriter(self.args.log_dir, graph=self.sess.graph)

    if args.test_dqn:
      #you can load your model here
      print('loading trained model')
      ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
      print(ckpt)
      if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        self.step = self.sess.run(self.global_step)
        print('load step: ', self.step)
      else:
        print('load model failed! exit...')
        exit(0)
    else:
      self.init_model()

  def init_model(self):
    ckpt = tf.train.get_checkpoint_state(self.args.save_dir)
    print(ckpt)
    if self.args.load_saver and ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
        print('Reloading model parameters..')
        self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        print(ckpt.model_checkpoint_path)
        self.step = self.sess.run(self.global_step)
        print('load step: ', self.step)
    else:
        print('Created new model parameters..')
        self.sess.run(tf.global_variables_initializer())

  def init_game_setting(self):
    """
    Testing function will call this function at the begining of new game
    Put anything you want to initialize if necessary
    """
    pass
  
  
  class ReplayMemory(object):

    def __init__(self, capacity):
      self.capacity = capacity
      self.memory = []
      self.position = 0

    def push(self, *args):
      """Saves a transition."""
      if len(self.memory) < self.capacity:
          self.memory.append(None)
      self.memory[self.position] = Transition(*args)
      self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
      return random.sample(self.memory, batch_size)

    def __len__(self):
      return len(self.memory)

  def build_net(self, s, var_scope):

    with tf.variable_scope(var_scope):
      with tf.variable_scope('conv1'):
        W1 = self.init_W(shape=[8, 8, 4, 32])
        b1 = self.init_b(shape=[32])
        conv1 = self.conv2d(s, W1, strides=4)
        h_conv1 = tf.nn.relu(tf.nn.bias_add(conv1, b1))


      with tf.variable_scope('conv2'):
        W2 = self.init_W(shape=[4, 4, 32, 64])
        b2 = self.init_b(shape=[64])
        conv2 = self.conv2d(h_conv1, W2, strides=2)
        h_conv2 = tf.nn.relu(tf.nn.bias_add(conv2, b2))

      with tf.variable_scope('conv3'):
        W3 = self.init_W(shape=[3, 3, 64, 64])
        b3 = self.init_b(shape=[64])
        conv3 = self.conv2d(h_conv2, W3, strides=1)
        h_conv3 = tf.nn.relu(tf.nn.bias_add(conv3, b3))

      h_flatten = tf.reshape(h_conv3, [-1, 3136])

      with tf.variable_scope('fc1'):
        W_fc1 = self.init_W(shape=[3136, 512])
        b_fc1 = self.init_b(shape=[512])
        fc1 = tf.nn.bias_add(tf.matmul(h_flatten, W_fc1), b_fc1)

      if not self.dueling:
        with tf.variable_scope('fc2'):
          h_fc1 = tf.nn.relu(fc1)
          W_fc2 = self.init_W(shape=[512, 4])
          b_fc2 = self.init_b(shape=[4])
          out = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2, name='Q')
      else:
        with tf.variable_scope('Value'):
          h_fc1_v = tf.nn.relu(fc1)
          W_v = self.init_W(shape=[512, 1])
          b_v = self.init_b(shape=[1])
          self.V = tf.nn.bias_add(tf.matmul(h_fc1_v, W_v), b_v, name='V')

        with tf.variable_scope('Advantage'):
          h_fc1_a = tf.nn.relu(fc1)
          W_a = self.init_W(shape=[512, 4])
          b_a = self.init_b(shape=[4])
          self.A = tf.nn.bias_add(tf.matmul(h_fc1_a, W_a), b_a, name='A')

        with tf.variable_scope('Q'):
          out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

    return out

  def build_net_alex(self, s, var_scope):

    with tf.variable_scope(var_scope):
      with tf.name_scope('conv1') as scope:
        kernel = tf.Variable(tf.truncated_normal([11, 11, 4, 64], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(s, kernel, [1, 4, 4, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[64], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(bias, name=scope)


      lrn1 = tf.nn.lrn(conv1, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn1')
      pool1 = tf.nn.max_pool(lrn1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool1')


      # 第二个卷积层
      with tf.name_scope('conv2') as scope:
        kernel = tf.Variable(tf.truncated_normal([5, 5, 64, 128], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool1, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[128], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(bias, name=scope)


      lrn2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001 / 9, beta=0.75, name='lrn2')
      pool2 = tf.nn.max_pool(lrn2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='pool2')

      # 第三个卷积层
      with tf.name_scope('conv3') as scope:
        kernel = tf.Variable(tf.random_normal([3, 3, 128, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(pool2, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv3 = tf.nn.relu(bias, name=scope)

      # 第四个卷积层
      with tf.name_scope('conv4') as scope:
        kernel = tf.Variable(tf.truncated_normal([3, 3, 256, 256], dtype=tf.float32, stddev=1e-1),
                             name='weights')
        conv = tf.nn.conv2d(conv3, kernel, [1, 1, 1, 1], padding='SAME')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32),
                             trainable=True, name='biases')
        bias = tf.nn.bias_add(conv, biases)
        conv5 = tf.nn.relu(bias, name=scope)

      # 第一个全连接层
      with tf.name_scope('fcl1') as scope:
        weight = tf.Variable(tf.truncated_normal([5 * 5 * 256, 1024], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[1024], dtype=tf.float32), trainable=True, name='biases')
        h_pool5_flat = tf.reshape(conv5, [-1, 5 * 5 * 256])
        fcl1 = tf.nn.relu(tf.matmul(h_pool5_flat, weight) + biases, name=scope)
        drop1 = tf.nn.dropout(fcl1, 0.7)

      # 第三个全连接层
      with tf.name_scope('fcl3') as scope:
        weight = tf.Variable(tf.truncated_normal([1024, 256], stddev=0.1), name='weights')
        biases = tf.Variable(tf.constant(0.0, shape=[256], dtype=tf.float32), trainable=True, name='biases')
        fcl3 = tf.nn.relu(tf.matmul(drop1, weight) + biases, name=scope)

      if not self.dueling:
        with tf.variable_scope('fc2'):
          h_fc1 = tf.nn.relu(fcl3)
          W_fc2 = self.init_W(shape=[256, 4])
          b_fc2 = self.init_b(shape=[4])
          out = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2, name='Q')
      else:
        with tf.variable_scope('Value'):
          h_fc1_v = tf.nn.relu(fcl3)
          W_v = self.init_W(shape=[256, 1])
          b_v = self.init_b(shape=[1])
          self.V = tf.nn.bias_add(tf.matmul(h_fc1_v, W_v), b_v, name='V')

        with tf.variable_scope('Advantage'):
          h_fc1_a = tf.nn.relu(fcl3)
          W_a = self.init_W(shape=[256, 4])
          b_a = self.init_b(shape=[4])
          self.A = tf.nn.bias_add(tf.matmul(h_fc1_a, W_a), b_a, name='A')

        with tf.variable_scope('Q'):
          out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

    return out


  def identity_block(self, X_input, kernel_size, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 3
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("id_block_stage" + str(stage)):
      filter1, filter2, filter3 = filters
      X_shortcut = X_input

      # First component of main path
      x = tf.layers.conv2d(X_input, filter1,
                           kernel_size=(1, 1), strides=(1, 1), name=conv_name_base + '2a')
      x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2a', training=self.TRAINING)
      x = tf.nn.relu(x)

      # Second component of main path
      x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size),
                           padding='same', name=conv_name_base + '2b')
      # batch_norm2 = tf.layers.batch_normalization(conv2, axis=3, name=bn_name_base+'2b', training=self.TRAINING)
      x = tf.nn.relu(x)

      # Third component of main path
      x = tf.layers.conv2d(x, filter3, kernel_size=(1, 1), name=conv_name_base + '2c')
      x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=self.TRAINING)

      # Final step: Add shortcut value to main path, and pass it through a RELU activation
      X_add_shortcut = tf.add(x, X_shortcut)
      add_result = tf.nn.relu(X_add_shortcut)

    return add_result

  def convolutional_block(self, X_input, kernel_size, filters, stage, block, stride=2):
    """
    Implementation of the convolutional block as defined in Figure 4
    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    kernel_size -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    stride -- Integer, specifying the stride to be used
    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    with tf.name_scope("conv_block_stage" + str(stage)):
      # Retrieve Filters
      filter1, filter2, filter3 = filters

      # Save the input value
      X_shortcut = X_input

      # First component of main path
      x = tf.layers.conv2d(X_input, filter1,
                           kernel_size=(1, 1),
                           strides=(stride, stride),
                           name=conv_name_base + '2a')
      x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2a', training=self.TRAINING)
      x = tf.nn.relu(x)

      # Second component of main path
      x = tf.layers.conv2d(x, filter2, (kernel_size, kernel_size), name=conv_name_base + '2b', padding='same')
      x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2b', training=self.TRAINING)
      x = tf.nn.relu(x)

      # Third component of main path
      x = tf.layers.conv2d(x, filter3, (1, 1), name=conv_name_base + '2c')
      x = tf.layers.batch_normalization(x, axis=3, name=bn_name_base + '2c', training=self.TRAINING)

      # SHORTCUT PATH
      X_shortcut = tf.layers.conv2d(X_shortcut, filter3, (1, 1),
                                    strides=(stride, stride), name=conv_name_base + '1')
      X_shortcut = tf.layers.batch_normalization(X_shortcut, axis=3, name=bn_name_base + '1', training=self.TRAINING)

      # Final step: Add shortcut value to main path, and pass it through a RELU activation
      X_add_shortcut = tf.add(X_shortcut, x)
      add_result = tf.nn.relu(X_add_shortcut)

    return add_result

  def build_net_resnet(self, x, var_scope):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    Returns:
    """


    assert (x.shape == (x.shape[0], 84, 84, 4))
    with tf.variable_scope(var_scope):
      # stage 1
      x = tf.layers.conv2d(x, filters=64, kernel_size=(11, 11), strides=(4, 4),padding='same', name='conv1')
      x = tf.layers.batch_normalization(x, axis=3, name='bn_conv1')
      x = tf.nn.relu(x)
      x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2))

      # stage 2
      x = self.convolutional_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
      x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
      x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')

      #
      x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))
      x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv2')
      x = tf.layers.batch_normalization(x, axis=3, name='bn_conv2')
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x, filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv3')

      flatten = tf.layers.flatten(x, name='flatten')
      fc1 = tf.layers.dense(flatten, units=128, activation=tf.nn.relu)

      if not self.dueling:
        with tf.variable_scope('fc2'):
          h_fc1 = tf.nn.relu(fc1)
          W_fc2 = self.init_W(shape=[128, 4])
          b_fc2 = self.init_b(shape=[4])
          out = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2, name='Q')
      else:
        with tf.variable_scope('Value'):
          h_fc1_v = tf.nn.relu(fc1)
          W_v = self.init_W(shape=[128, 1])
          b_v = self.init_b(shape=[1])
          self.V = tf.nn.bias_add(tf.matmul(h_fc1_v, W_v), b_v, name='V')

        with tf.variable_scope('Advantage'):
          h_fc1_a = tf.nn.relu(fc1)
          W_a = self.init_W(shape=[128, 4])
          b_a = self.init_b(shape=[4])
          self.A = tf.nn.bias_add(tf.matmul(h_fc1_a, W_a), b_a, name='A')

        with tf.variable_scope('Q'):
          out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

    return out

  def build_net_resnet_complex(self, x, var_scope):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER
    Arguments:
    Returns:
    """


    assert (x.shape == (x.shape[0], 84, 84, 4))
    with tf.variable_scope(var_scope):
      # stage 1
      x = tf.layers.conv2d(x, filters=64, kernel_size=(11, 11), strides=(4, 4),padding='same', name='conv1')
      x = tf.layers.batch_normalization(x, axis=3, name='bn_conv1')
      x = tf.nn.relu(x)
      x = tf.layers.max_pooling2d(x, pool_size=(3, 3), strides=(2, 2))

      # stage 2
      x = self.convolutional_block(x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', stride=1)
      x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='b')
      x = self.identity_block(x, 3, [64, 64, 256], stage=2, block='c')
      #stage 3
      x = self.convolutional_block(x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a', stride=1)
      x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='e')
      x = self.identity_block(x, 3, [128, 128, 512], stage=3, block='f')

      #
      x = tf.layers.max_pooling2d(x, pool_size=(2, 2), strides=(2, 2))
      x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv2')
      x = tf.layers.batch_normalization(x, axis=3, name='bn_conv2')
      x = tf.nn.relu(x)
      x = tf.layers.conv2d(x, filters=512, kernel_size=(3, 3), strides=(1, 1), padding='valid', name='conv3')

      flatten = tf.layers.flatten(x, name='flatten')
      fc1 = tf.layers.dense(flatten, units=512, activation=tf.nn.relu)

      if not self.dueling:
        with tf.variable_scope('fc2'):
          h_fc1 = tf.nn.relu(fc1)
          W_fc2 = self.init_W(shape=[512, 4])
          b_fc2 = self.init_b(shape=[4])
          out = tf.nn.bias_add(tf.matmul(h_fc1, W_fc2), b_fc2, name='Q')
      else:
        with tf.variable_scope('Value'):
          h_fc1_v = tf.nn.relu(fc1)
          W_v = self.init_W(shape=[512, 1])
          b_v = self.init_b(shape=[1])
          self.V = tf.nn.bias_add(tf.matmul(h_fc1_v, W_v), b_v, name='V')

        with tf.variable_scope('Advantage'):
          h_fc1_a = tf.nn.relu(fc1)
          W_a = self.init_W(shape=[512, 4])
          b_a = self.init_b(shape=[4])
          self.A = tf.nn.bias_add(tf.matmul(h_fc1_a, W_a), b_a, name='A')

        with tf.variable_scope('Q'):
          out = self.V + (self.A - tf.reduce_mean(self.A, axis=1, keep_dims=True))

    return out

  def init_W(self, shape, name='weights', 
    w_initializer=tf.truncated_normal_initializer(0, 1e-2)):

    return tf.get_variable(
      name=name,
      shape=shape, 
      initializer=w_initializer)

  def init_b(self, shape, name='biases', 
    b_initializer = tf.constant_initializer(1e-2)):

    return tf.get_variable(
      name=name,
      shape=shape,
      initializer=b_initializer)

  def conv2d(self, x, kernel, strides=4):

    return tf.nn.conv2d(
      input=x, 
      filter=kernel, 
      strides=[1, strides, strides, 1], 
      padding="VALID")

  def max_pool(self, x, ksize=2, strides=2):
    return tf.nn.max_pool(x, 
      ksize=[1, ksize, ksize, 1], 
      strides=[1, strides, strides, 1], 
      padding="SAME")

  def buildOptimizer(self):
    with tf.variable_scope('loss'):
      # (32, 4) -> (32, 1)
      self.train_summary.append(tf.summary.scalar('avg_q', 
        tf.reduce_mean(self.q_eval)))

      self.q_action = tf.reduce_sum(tf.multiply(self.q_eval, self.action_input), axis=1)
      self.train_summary.append(tf.summary.scalar('avg_action_q', 
        tf.reduce_mean(self.q_action)))  

      self.loss = tf.reduce_mean(tf.square(self.y_input - self.q_action))
      # self.loss = tf.reduce_mean(tf.losses.huber_loss(labels=self.y_input, predictions=self.q_action, delta=0.3))
      self.train_summary.append(tf.summary.scalar('loss', self.loss))

      self.train_summary = tf.summary.merge(self.train_summary)
    with tf.variable_scope('train'):
      #self.logits = self.build_rmsprop_optimizer(self.lr, 0.99, 1e-6, 1, 'graves_rmsprop')
      self.logits = self.build_rmsprop_optimizer(self.lr, 0.99, 1e-6, 1, 'rmsprop')

  def huber_loss(self, predictions, labels, delta):
    residual = tf.abs(predictions, labels)
    large_loss = 0.5 * tf.square(residual)
    small_loss = delta * residual - 0.5 * tf.square(delta)
    cond = tf.less(residual, delta)

    return tf.where(cond, large_loss, small_loss)


  # https://github.com/Jabberwockyll/deep_rl_ale
  def build_rmsprop_optimizer(self, learning_rate, rmsprop_decay, rmsprop_constant, gradient_clip, version):

    with tf.name_scope('rmsprop'):
      optimizer = None
      if version == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=rmsprop_decay, momentum=0.0, epsilon=rmsprop_constant)
      elif version == 'graves_rmsprop':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

      grads_and_vars = optimizer.compute_gradients(self.loss)
      grads = [gv[0] for gv in grads_and_vars]
      params = [gv[1] for gv in grads_and_vars]
      # print(grads)
      if gradient_clip > 0:
        grads = tf.clip_by_global_norm(grads, gradient_clip)[0]

      grads = [grad for grad in grads if grad != None]

      if version == 'rmsprop':
        return optimizer.apply_gradients(zip(grads, params))
      elif version == 'graves_rmsprop':
        square_grads = [tf.square(grad) for grad in grads if grad != None]

        avg_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]
        avg_square_grads = [tf.Variable(tf.zeros(var.get_shape())) for var in params]

        update_avg_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * grad_pair[1])) 
          for grad_pair in zip(avg_grads, grads)]

        update_avg_square_grads = [grad_pair[0].assign((rmsprop_decay * grad_pair[0]) + ((1 - rmsprop_decay) * tf.square(grad_pair[1]))) 
          for grad_pair in zip(avg_square_grads, grads)]
        avg_grad_updates = update_avg_grads + update_avg_square_grads

        rms = [tf.sqrt(avg_grad_pair[1] - tf.square(avg_grad_pair[0]) + rmsprop_constant)
          for avg_grad_pair in zip(avg_grads, avg_square_grads)]

        rms_updates = [grad_rms_pair[0] / grad_rms_pair[1] for grad_rms_pair in zip(grads, rms)]
        train = optimizer.apply_gradients(zip(rms_updates, params))

        return tf.group(train, tf.group(*avg_grad_updates))


  def storeTransition(self, s, action, reward, s_, done):
    """
    Store transition in this step
    Input:
        s: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
        s_: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
        action: int (0, 1, 2, 3)
            the predicted action from trained model
        reward: float64 (0, +1, -1)
            the reward from selected action
    Return:
        None
    """
    
    if self.step == 0:
      self.stage = stages[0]
    elif self.step == self.args.observe_steps:
      self.stage = stages[1]
    elif self.step == self.args.observe_steps + self.args.explore_steps:
      self.stage = stages[2]

    self.step = self.sess.run(self.add_global)
    
    #np.set_printoptions(threshold=np.nan)
    assert np.amin(s) >= 0.0
    assert np.amax(s) <= 1.0
    
    s  = (s * 255).round().astype(np.uint8)
    s_ = (s_ * 255).round().astype(np.uint8)

    # reward = np.sign(reward)

    #print(sys.getsizeof(image)) # 28352, uint8
    #print(sys.getsizeof(s)) # 113024, float32
    
    self.memory.push(s, int(action), int(reward), s_, done)
  
  def learn(self):
    transitions = self.memory.sample(self.batch_size)
    minibatch = Transition(*zip(*transitions))

    state_batch = [(s).astype(np.float32) / 255.0 for s in list(minibatch.state)]
    next_state_batch = [(s_).astype(np.float32) / 255.0 for s_ in list(minibatch.next_state)]
    action_batch = []
    for act in list(minibatch.action):
      one_hot_action = np.zeros(self.n_actions)
      one_hot_action[act] = 1
      action_batch.append(one_hot_action)
    reward_batch = list(minibatch.reward)
    reward_batch = [float(data) for data in reward_batch]
    done_batch = list(minibatch.done)

    y_batch = []
    if not self.double: # not doubleDQN
      q_batch = self.sess.run(self.q_target, 
        feed_dict={self.s_: next_state_batch})
      for i in range(self.batch_size):
        done = done_batch[i]
        if done:
          y_batch.append(reward_batch[i])
        else:
          y = reward_batch[i] + self.gamma * np.max(q_batch[i])
          y_batch.append(y)
    else:
      q_batch_now = self.sess.run(self.q_eval,
        feed_dict={self.s: next_state_batch})
      q_batch = self.sess.run(self.q_target, 
        feed_dict={self.s_: next_state_batch})
      for i in range(self.batch_size):
        done = done_batch[i]
        if done:
          y_batch.append(reward_batch[i])
        else:
          double_q = q_batch[i][np.argmax(q_batch_now[i])]
          y = reward_batch[i] + self.gamma * double_q
          y_batch.append(y)

    run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    _, summary, loss = self.sess.run([self.logits, self.train_summary, self.loss], feed_dict={
      self.s: state_batch,
      self.y_input: y_batch,
      self.action_input: action_batch
      }, options=run_options)

    if self.step % self.args.summary_time == 0:
      self.summary_writer.add_summary(summary, global_step=self.step)

    if self.step % self.args.update_target == 0 and self.step > self.args.observe_steps:
      self.sess.run(self.replace_target_op)
      print("\n[target params replaced]")

    return loss

  def train(self):

    """
    Implement your training algorithm here
    """
    pbar = tqdm(range(self.args.episode_start, self.args.num_episodes))
    current_loss = 0
    train_rewards = []
    train_rewards_noclip = []
    train_episode_len = 0.0
    file_loss = open(self.output_logs, "a")
    file_loss.write("episode,step,epsilon,reward,loss,length\n")
    for episode in pbar:
      # print('episode: ', episode)
      # "state" is also known as "observation"
      obs = self.env.reset() #(84, 84, 4)
      self.init_game_setting()
      train_loss = 0
      
      episode_reward = 0.0
      episode_reward_noclip = 0.0
      for s in range(self.args.max_num_steps):
        # self.env.env.render()
        action = self.make_action(obs, test=False)
        obs_, reward, done, info = self.env.step(action)
        reward_noclip = reward
        reward = np.sign(reward)
        # if reward ==1 :
        #   print('reward no clip', reward_noclip)
        #   print('reward clip', reward)
        episode_reward += reward
        episode_reward_noclip += reward_noclip
        self.storeTransition(obs, action, reward, obs_, done)
        
        # if len(self.memory) > self.args.replay_memory_size:
        # NOT REQUIRED TO POPLEFT(), IT WILL BE REPLACED 
        # self.replay_memory.popleft()

        # once the storage stored > batch_size, start training
        if len(self.memory) > self.batch_size:
          if self.step % self.args.update_current == 0:
            loss = self.learn()
            train_loss += loss

        if self.step % self.args.saver_steps == 0 and episode != 0:
          ckpt_path = self.saver.save(self.sess, self.ckpts_path, global_step = self.step)
          print("\nStep: " + str(self.step) + ", Saver saved: " + ckpt_path)

        obs = obs_
        if done:
          # print('.....................total reward no clip..................', episode_reward_noclip)
          # print('.....................total reward clip..................', episode_reward)
          break

      train_rewards.append(episode_reward)
      train_rewards_noclip.append(episode_reward_noclip)
      train_episode_len += s

      if episode % self.args.num_eval == 0 and episode != 0:
        current_loss = train_loss
        avg_reward_train = np.mean(train_rewards)
        avg_reward_train_noclip = np.mean(train_rewards_noclip)
        train_rewards_noclip = []
        train_rewards = []
        avg_episode_len_train = train_episode_len / float(self.args.num_eval)
        train_episode_len = 0.0
        
        file_loss.write(str(episode) + "," + str(self.step) + "," + "{:.4f}".format(self.epsilon) + "," + "{:.2f}".format(avg_reward_train) + "," + "{:.4f}".format(current_loss) + "," + "{:.2f}".format(avg_episode_len_train) + "," + "{:.2f}".format(avg_reward_train_noclip) + "\n")
        file_loss.flush()
        
        print("\n[Train] Avg Reward: " + "{:.2f}".format(avg_reward_train) + ", Avg Episode Length: " + "{:.2f}".format(avg_episode_len_train) + ", Avg reward no clip: " + "{:.2f}".format(avg_reward_train_noclip))

      pbar.set_description(self.stage + " G: " + "{:.2f}".format(self.gamma) + ', E: ' + "{:.2f}".format(self.epsilon) + ", L: " + "{:.4f}".format(current_loss) + ", D: " + str(len(self.memory)) + ", S: " + str(self.step))

    print('game over')
    # env.destroy()

  def make_action(self, observation, test=True):
    """
    Return predicted action of your agent
    Input:
        observation: np.array
            stack 4 last preprocessed frames, shape: (84, 84, 4)
    Return:
        action: int
            the predicted action from trained model
            """
    state = observation.reshape((1, 84, 84, 4))
    q_value = self.sess.run(self.q_eval, feed_dict={self.s: state})[0]
    
    if test:
      if random.random() <= 0.025:
        return random.randrange(self.n_actions)
      return np.argmax(q_value)

    if random.random() <= self.epsilon:
      action = random.randrange(self.n_actions)
    else:
      action = np.argmax(q_value)

    if self.epsilon > self.args.epsilon_end \
        and self.step > self.args.observe_steps:
      old_e = self.epsilon
      interval = self.args.epsilon_start - self.args.epsilon_end
      self.epsilon -= interval / float(self.args.explore_steps)
      # print('epsilon: ', old_e, ' -> ', self.epsilon)

    return action

    # return self.env.get_random_action()

