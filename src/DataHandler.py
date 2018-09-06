import tensorflow as tf
import os

def get_parse_fn(mode, params):
    """
    Creates a parse function for dataset examples. Performs raw decoding, reshaping and casting on both train and eval
    data, performs preprocessing on train data if turned on.
    """
    def _parse_fn(example):
        # format of serialised examples in .tfrecords
        example_fmt = {
            'image': tf.FixedLenFeature((), tf.string, ""),
            'label': tf.FixedLenFeature((), tf.int64, -1)
        }
        parsed = tf.parse_single_example(example, example_fmt)
        
        # decode raw tfrecords data and convert to uint8
        image = tf.decode_raw(parsed["image"], tf.uint8)
        
        # reshape data into desired shape, then cast to desired output type
        height, width, depth = params.data_dims
        image.set_shape([height * width * depth])
        image = tf.cast(tf.reshape(image, [height, width, depth]), tf.float32)
        label = tf.cast(parsed['label'], tf.int64)

        # perform preprocessing if turned on
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