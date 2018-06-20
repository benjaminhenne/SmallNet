import tensorflow as tf

class swish(object):
    def __init__(self):
        self.__name__ = 'swish'

    def __call__(self, x, ß):
        return x * tf.nn.sigmoid(x*ß)

class identity(object):
    def __init__(self):
        self.__name__ = 'identity'

    def __call__(self, x):
        return x