import argparse
import os

import tensorflow as tf

from settings import Settings
import smallnet_architecture as net

# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator
# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/learn/README.md
# https://github.com/GoogleCloudPlatform/tf-estimator-tutorials/blob/master/05_Autoencoding/04.0%20-%20Dimensionality%20Reduction%20-%20Autoencoding%20%2B%20Custom%20Estimator%20with%20MNIST.ipynb

def get_model_fn(num_gpus, variable_strategy, num_workers):

    def _small_net_model_fn(features, labels, mode, params):
        settings = Settings()
        network = net.Smallnet(settings, features, labels, params)
        logits = network.logits

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        predicted_classes = tf.argmax(logits, 1)
        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'class': predicted_classes,
                'prob': tf.nn.softmax(logits)
            }
            return tf.estimator.EstimatorSpec(mode, predictions=predictions)

        # Create training op.
        if is_training:
            return tf.estimator.EstimatorSpec(
                mode, loss=network.xentropy, train_op=network.update)

        # Compute evaluation metrics.
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predicted_classes)
        }
        return tf.estimator.EstimatorSpec(
            mode, loss=network.xentropy,
            eval_metric_ops=eval_metric_ops)

    return _small_net_model_fn

def parse_fn(example):
    example_fmt = {
        "image": tf.FixedLenFeature((), tf.string),
        "label": tf.FixedLenFeature((), tf.int64)
    }
    parsed = tf.parse_single_example(example, example_fmt)
    image = tf.image.decode_image(parsed["image"])
    #image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
    #image = tf.random_crop(image, [HEIGHT, WIDTH, DEPTH])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, parsed["label"]

def _set_shapes(x, y):
    x.set_shape([32, 32, 3])
    y.set_shape([])
    return x, y

def get_input_fn(mode=None, params=None):

    def _input_fn(mode=None, params=None):
        with tf.device('/cpu:0'):
            #shapes = [tf.TensorShape([params.batch_size, 32, 32, None]), tf.TensorShape(params.batch_size)]
            if mode == 'train':
                files = tf.data.Dataset.list_files(os.path.join(params.data_dir, "train.tfrecord"))
                dataset = files.interleave(tf.data.TFRecordDataset, 1) # cyle_length = num files
                #dataset = dataset.apply(tf.contrib.data.map_and_batch(
                #    map_func=parse_fn,
                #    batch_size=hparams.batch_size,
                #    num_parallel_batches=hparams.num_cores,
                #    drop_remainder=True))
                dataset = dataset.map(map_func=parse_fn, num_parallel_calls=params.num_cores)
                dataset = dataset.map(map_func=_set_shapes)
                #dataset = dataset.apply(tf.contrib.data.assert_element_shape(shapes))
                dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
                #dataset = dataset.batch(hparams['batch_size'], drop_remainder=True)
                #dataset = dataset.batch(params.batch_size, drop_remainder=True)
                dataset = dataset.shuffle(params.batch_size).repeat()
                #dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(
                #    buffer_size=hparams.batch_size,
                #    count=None))
                dataset = dataset.prefetch(buffer_size=params.batch_size)
            elif mode == 'eval':
                files = tf.data.Dataset.list_files(os.path.join(params.data_dir, "eval.tfrecord"))
                dataset = files.interleave(tf.data.TFRecordDataset, 1)
                dataset = dataset.map(map_func=parse_fn, num_parallel_calls=params.num_cores)
                dataset = dataset.map(map_func=_set_shapes)
                dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(500))
            else:
                raise ValueError('Invalid mode type selected!')

            return dataset.make_one_shot_iterator().get_next()

    return _input_fn

def build_estimator(num_gpus, variable_strategy, run_config, hparams):
    classifier = tf.estimator.Estimator(model_fn=get_model_fn(
        num_gpus,
        variable_strategy,
        run_config.num_worker_replicas or 1),
        #model_dir=run_config._model_dir,
        config=run_config,
        params=hparams)
    return classifier

def main(variable_strategy,
         log_device_placement, num_intra_threads, **hparams):

    tf.logging.set_verbosity(tf.logging.INFO)

    session_config = tf.ConfigProto(
        allow_soft_placement=False,
        device_count={"CPU": hparams['num_cores'], "GPU": hparams['num_gpus']},
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True)
    )
    config = tf.estimator.RunConfig(
        model_dir=hparams['output_dir'],
        tf_random_seed=None,
        save_summary_steps=100,
        save_checkpoints_steps=1000,
        save_checkpoints_secs=None,
        session_config=session_config,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000
        #train_distribute=None  TODO was gibts hier zu holen?
    )

    classifier = build_estimator(
        hparams['num_gpus'], variable_strategy,
        run_config=config,
        hparams=tf.contrib.training.HParams(**hparams)
    )

    tf.estimator.train_and_evaluate(
        classifier,
        tf.estimator.TrainSpec(input_fn=get_input_fn(mode='train'), max_steps=hparams['train_steps']),
        tf.estimator.EvalSpec(input_fn=get_input_fn(mode='eval'), steps=None)
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir',
        type=str,
        required=True,
        help='The directory where the input data is stored.',
        dest='data_dir')
    parser.add_argument(
        '-o', '--output-dir',
        type=str,
        required=True,
        help='The directory where the output will be stored.',
        dest='output_dir')
    parser.add_argument(
        '-n', '--num-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.',
        dest='num_gpus')
    parser.add_argument(
        '-v', '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='CPU',
        help='Where to locate variable operations',
        dest='variable_strategy')
    parser.add_argument(
        '-c', '--num-cpu-cores',
        type=int,
        default=1,
        help='The number of cpu cores available for data preparation.',
        dest='num_cores')
    parser.add_argument(
        '-s', '--train-steps',
        type=int,
        default=100,
        help='The number of steps to use for training.',
        dest='train_steps')
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=128,
        help='Batch size.',
        dest='batch_size')
    parser.add_argument(
        '-l', '--learning-rate',
        type=float,
        default=0.1,
        help="""\
        This is the inital learning rate value. The learning rate will decrease
        during training. For more details check the model_fn implementation in
        this file.""",
        dest='learning_rate')
    parser.add_argument(
        '--num-intra-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for intra-op parallelism. When training on CPU
        set to 0 to have the system pick the appropriate number or alternatively
        set it to the number of physical CPU cores.\
        """)
    parser.add_argument(
        '--num-inter-threads',
        type=int,
        default=0,
        help="""\
        Number of threads to use for inter-op parallelism. If set to 0, the
        system will pick an appropriate number.\
        """)
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=False,
        help='Whether to log device placement.')
    args = parser.parse_args()

    main(**vars(args))
