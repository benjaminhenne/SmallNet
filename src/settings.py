import cifar10_loader as loader
from activations import *
import tensorflow as tf


class Settings:
    def __init__(self):
        # hyperparameters
        self.epochs = 50
        self.batch_size = 256
        self.logit_dim = 10
        self.l_rate = 5e-4
        self.l1_regularize = False
        self.l1_scale = 0.8
        self.l2_regularize = False
        self.l2_lambda = 0.01
        self.dropout_rate = 0.5
        self.verbose_summaries = False

        self.dataset_path = '/net/store/ni/projects/l1activations/datasets/cifar-10/'
        #self.dataset_path = 'C:/Code/CIFAR-10'
        self.summary_path = './summaries/'
        self.storage_path = './stored_weights'
        self.loader = loader.CIFAR(self.dataset_path)
        self.activations = [tf.nn.relu, tf.nn.elu, tf.nn.tanh, swish, identity]
        self.act_inits = [lambda fan_in: 2.0/fan_in[2] if len(fan_in) == 4 else 2.0/fan_in[1], # relu
                          lambda fan_in: 0.1, # elu
                          lambda fan_in: fan_in[2]**(-1/2) if len(fan_in) == 4 else fan_in[1]**(-1/2), # tanh
                          lambda fan_in: 0.1, # swish
                          lambda fan_in: 0.1 # identity
                          ]
        self.network_layout = 'default'
        #self.optimiser = tf.train.AdamOptimizer()
        self.optimiser = 'Adam'
