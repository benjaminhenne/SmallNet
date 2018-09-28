import tensorflow as tf
import os
import pickle

cifar_dict = {
    'encoded': True,
    'data': 'image/encoded',
    'labels': 'image/class/label',
    'resize_before_use': False,
    'hwd': [32, 32, 3],
    'eval_batch_size': 5000,
    'file_pattern': {
        'train': 'train*.tfre*',
        'validation': 'eval*.tfre*',
        'test': 'test*.tfre*'
    },
    'format': {
        'image/encoded':        tf.FixedLenFeature((), tf.string, ""),
        'image/format':         tf.FixedLenFeature((), tf.string, ""),
        'image/class/label':    tf.FixedLenFeature((), tf.int64, -1),
        'image/height':         tf.FixedLenFeature((), tf.int64, -1),
        'image/width':          tf.FixedLenFeature((), tf.int64, -1)
    }
}

imagenet_dict = {
    'encoded': True,
    'data': 'image/encoded',
    'labels': 'image/class/label',
    'resize_before_use': True,
    'hwd': [480, 480, 3],
    'eval_batch_size': 10000,
    'file_pattern': {
        'train': 'train/train-*',
        'validation': 'validation/validation-*',
        'test': None
    },
    'format_large': {
        'image/height':             tf.FixedLenFeature((), tf.int64, -1),
        'image/width':              tf.FixedLenFeature((), tf.int64, -1),
        'image/colorspace':         tf.FixedLenFeature((), tf.string, ""),
        'image/channels':           tf.FixedLenFeature((), tf.int64, -1),
        'image/class/label':        tf.FixedLenFeature((), tf.int64, -1),
        'image/class/synset':       tf.FixedLenFeature((), tf.string, ""),
        'image/class/text':         tf.FixedLenFeature((), tf.string, ""),
        'image/object/bbox/xmin':   tf.FixedLenFeature((), tf.float32, -1.0),
        'image/object/bbox/xmax':   tf.FixedLenFeature((), tf.float32, -1.0),
        'image/object/bbox/ymin':   tf.FixedLenFeature((), tf.float32, -1.0),
        'image/object/bbox/ymax':   tf.FixedLenFeature((), tf.float32, -1.0),
        'image/object/bbox/label':  tf.FixedLenFeature((), tf.int64, -1),
        'image/format':             tf.FixedLenFeature((), tf.string, ""),
        'image/filename':           tf.FixedLenFeature((), tf.string, ""),
        'image/encoded':            tf.FixedLenFeature((), tf.string, ""),
    },
    'format': {
        'image/height':             tf.FixedLenFeature((), tf.int64, -1),
        'image/width':              tf.FixedLenFeature((), tf.int64, -1),
        'image/channels':           tf.FixedLenFeature((), tf.int64, -1),
        'image/class/label':        tf.FixedLenFeature((), tf.int64, -1),
        'image/class/synset':       tf.FixedLenFeature((), tf.string, ""),
        'image/class/text':         tf.FixedLenFeature((), tf.string, ""),
        'image/encoded':            tf.FixedLenFeature((), tf.string, ""),
    }
}

with open('content.pickle', 'wb') as handle:
    pickle.dump(cifar_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
