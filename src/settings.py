import cifar10_loader as loader
from activations import *
import tensorflow as tf


class Settings:
    """Contains several model-global settings and hyperparameters for model tuning."""
    def __init__(self):
        # hyperparameters
        self.logit_dim = 10
        self.l1_regularize = True
        self.l1_scale = 0.1
        self.l2_regularize = False
        self.l2_lambda = 0.001
        self.dropout_rate = 0.5
        self.verbose_summaries = False

        self.activations = [tf.nn.relu, tf.nn.elu, tf.nn.tanh, swish, identity]
        self.act_inits = [lambda fan_in: 2.0/fan_in[2] if len(fan_in) == 4 else 2.0/fan_in[1], # relu
                          lambda fan_in: 0.1, # elu
                          lambda fan_in: fan_in[2]**(-1/2) if len(fan_in) == 4 else fan_in[1]**(-1/2), # tanh
                          lambda fan_in: 0.1, # swish
                          lambda fan_in: 0.1 # identity
        ]
        self.network_layout = 'default'
        self.optimiser = 'Adam'

    def print_settings(self):
        print('L1 yes/no: \t\t{}'.format(self.l1_regularize))
        print('L1 scale: \t\t{}'.format(self.l1_scale))
        print('L2 yes/no: \t\t{}'.format(self.l2_regularize))
        print('L2 lambda: \t\t{}'.format(self.l2_lambda))
        print('Dropout rate: \t\t{}'.format(self.dropout_rate))
        print('Verbosity: \t\t{}'.format(self.verbose_summaries))
        print('Activations: \t\t{}'.format([a for a in self.activations]))
        print('Network layout: \t\t{}'.format(self.network_layout))
        print('Optimiser: \t\t{}'.format(self.optimiser))
