import argparse
import os

import numpy as np
import tensorflow as tf

# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator
# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/learn/README.md

def get_model_fn(num_gpus, variable_strategy, num-workers):

    def _small_net_model_fn(features, labels, mode, params):
        pass

    return _small_net_model_fn

def get_input_fn(mode, hparams):
    if mode == 'train':
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={X_FEATURE: mnist.train.images},
            y=mnist.train.labels.astype(np.int32),
            batch_size=hparams.batch_size,
            num_epochs=None,
            shuffle=True)
    elif mode == 'test':
        input_fn = tf.estimator.inputs.numpy_input_fn(
            x={X_FEATURE: mnist.test.images},
            y=mnist.test.labels.astype(np.int32),
            num_epochs=1,
            shuffle=False)
    else:
        raise ValueError('Invalid mode type selected!')

    return input_fn

def build_estimator(data_dir, num_gpus, variable_strategy, run_config, hparams):
    classifier = tf.estimator.Estimator(model_fn=get_model_fn(
        num_gpus,
        variable_strategy,
        run_config.num_worker_replicas or 1)
    )
    return classifier

def main(data_dir, output_dir, num_gpus, variable_strategy,
         log_device_placement, num_intra_threads, **hparams):

    tf.logging.set_verbosity(tf.logging.INFO)

    session_config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=log_device_placement,
        intra_op_parallelism_threads=num_intra_threads,
        gpu_options=tf.GPUOptions(force_gpu_compatible=True)
    )
    config = tf.estimator.RunConfig(
        model_dir=output_dir,
        tf_random_seed=None,
        save_summary_steps=100,
        save_checkpoint_steps=1000,
        save_checkpoint_secs=None,
        session_config=session_config,
        keep_checkpoint_max=5,
        keep_checkpoint_every_n_hours=10000,
        train_distribute=None      # TODO was gibts hier zu holen?
    )

    classifier = build_estimator(
        data_dir, num_gpus, variable_strategy,
        run_config=config,
        hparams=tf.contrib.training.HParams(**hparams)
    )
    classifier.train(input_fn=get_input_fn('train', **hparams), hparams.batch_size)

    tf.estimator.train_and_evaluate(
        classifier,
        tf.estimator.TrainSpec(input_fn=get_input_fn('train', **hparams), max_steps=hparams.train_steps),
        tf.estimator.EvalSpec(input_fn=get_input_fn('test', **hparams), steps=None)
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
        '-v', '--variable-strategy',
        choices=['CPU', 'GPU'],
        type=str,
        default='CPU',
        help='Where to locate variable operations',
        dest='var_strat')
    parser.add_argument(
        '-n', '--num-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.',
        dest='num_gpus')
    parser.add_argument(
        '-s', '--train-steps',
        type=int,
        default=80000,
        help='The number of steps to use for training.',
        dest='train_steps')
    parser.add_argument(
        '-b', '--batch-size',
        type=int,
        default=128,
        help='Batch size.',
        dest='batch_size')
    parser.add_argument(
        '-m', '--momentum',
        type=float,
        default=0.9,
        help='Momentum for MomentumOptimizer.')
    parser.add_argument(
        '-w', '--weight-decay',
        type=float,
        default=2e-4,
        help='Weight decay for convolutions.',
        dest='weight_decay')
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
        '-d', '--use-distortion-for-training',
        type=bool,
        default=True,
        help='If doing image distortion for training.')
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
        '--data-format',
        type=str,
        default=None,
        help="""\
        If not set, the data format best for the training device is used.
        Allowed values: channels_first (NCHW) channels_last (NHWC).\
        """)
    parser.add_argument(
        '--log-device-placement',
        action='store_true',
        default=False,
        help='Whether to log device placement.')
    parser.add_argument(
        '--batch-norm-decay',
        type=float,
        default=0.997,
        help='Decay for batch norm.')
    parser.add_argument(
        '--batch-norm-epsilon',
        type=float,
        default=1e-5,
        help='Epsilon for batch norm.')
    args = parser.parse_args()

    main(**vars(args))
