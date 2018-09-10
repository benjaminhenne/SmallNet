import tensorflow as tf
import os
import pickle

def get_parse_fn(mode, params):
    """
    Creates a parse function for dataset examples. Performs decoding, reshaping and casting on both train and eval
    data, performs preprocessing on train data if turned on.
    """
    def _parse_fn(example):
        """
        Method to parse dataset content. Decodes images, reshapes and casts them to float32, applies preprocessing
        when necessary.
        """

        """
        Reads dataset information from pickle file. Contents:
            'encoded':  True if dataset has been encoded, false if tf.decode_raw() is fine
            'data':     Name of image data feature in format dict
            'labels':   Name of label data feature in format dict
            'hwd':      Array containing heigth, width and depth of encoded image data
            'format':   Dictionary containing format of dataset to decode
        """
        with open(os.path.join(params.data_dir, 'content.pickle'), 'rb') as handle:
            info = pickle.load(handle)

        example_fmt = info['format']
        parsed = tf.parse_single_example(example, example_fmt)

        # decode data, dependant on encoding
        if info['encoded']:
            image = tf.image.decode_image(parsed[info['data']], channels=3, dtype=tf.float32)
        else:
            image = tf.cast(tf.decode_raw(parsed[info['data']], tf.uint8), tf.float32)

        # reshape data into desired shape
        height, width, depth = info['hwd']
        image.set_shape([height * width * depth])
        image = tf.reshape(image, [height, width, depth])
        label = parsed[info['labels']]

        # perform preprocessing if enabled
        if mode == 'train' and params.preprocess_data:
            zoom_factor = params.preprocess_zoom
            image = tf.image.resize_image_with_crop_or_pad(image, int(height*zoom_factor), int(width*zoom_factor))
            image = tf.random_crop(image, [height, width, depth])
            image = tf.image.random_flip_left_right(image)
        return image, label

    return _parse_fn

class DataHandler():
    """Class to provide datasets from serialised .tfrecords-files."""
    def __init__(self, mode, file_pattern, params, parse_fn=get_parse_fn):
        files = tf.data.Dataset.list_files(os.path.join(params.data_dir, file_pattern))
        self.dataset = files.interleave(tf.data.TFRecordDataset, 1) # cycle_length = num files
        self.params = params
        self.parse_fn = parse_fn
        self.mode = mode

    def prepare_for_train(self):
        """Performs mapping on dataset, forms batches, shuffles and repeats and performs prefetching."""
        self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.parse_fn(self.mode, self.params),
            batch_size=self.params.batch_size,
            num_parallel_batches=self.params.num_cores,
            drop_remainder=True))
        self.dataset = self.dataset.apply(tf.contrib.data.shuffle_and_repeat(
            buffer_size=self.params.batch_size,
            count=None))
        self.dataset = self.dataset.prefetch(buffer_size=1)
        return self.dataset

    def prepare_for_eval(self, eval_batch_size):
        """Performs mapping on dataset, forms batches, no shuffling or repeating, no prefetching."""
        self.dataset = self.dataset.apply(tf.contrib.data.map_and_batch(
            map_func=self.parse_fn(self.mode, self.params),
            batch_size=eval_batch_size,
            num_parallel_batches=self.params.num_cores,
            drop_remainder=True))
        return self.dataset