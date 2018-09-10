import tensorflow as tf
import os
import pickle

info_dict = {
    'encoded': True,
    'data': 'image/encoded',
    'labels': 'image/class/label',
    'hwd': [32, 32, 3],
    'format': {
        'image/encoded': tf.FixedLenFeature((), tf.string, ""),
        'image/format': tf.FixedLenFeature((), tf.string, ""),
        'image/class/label': tf.FixedLenFeature((), tf.int64, -1),
        'image/height': tf.FixedLenFeature((), tf.int64, -1),
        'image/width': tf.FixedLenFeature((), tf.int64, -1)
    }
}

with open('content.pickle', 'wb') as handle:
    pickle.dump(info_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
