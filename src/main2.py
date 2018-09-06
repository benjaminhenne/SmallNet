import argparse
import os

import tensorflow as tf

from settings import Settings
from DataHandler import DataHandler
import smallnet_architecture as net

# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10_estimator
# https://github.com/tensorflow/tensorflow/blob/r1.10/tensorflow/contrib/learn/README.md
# https://github.com/GoogleCloudPlatform/tf-estimator-tutorials/blob/master/05_Autoencoding/04.0%20-%20Dimensionality%20Reduction%20-%20Autoencoding%20%2B%20Custom%20Estimator%20with%20MNIST.ipynb

def get_model_fn(num_gpus):

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
            return tf.estimator.EstimatorSpec(mode, loss=network.xentropy, train_op=network.update)

        # Compute evaluation metrics.
        eval_metric_ops = {
            'accuracy': tf.metrics.accuracy(
                labels=labels, predictions=predicted_classes)
        }
        return tf.estimator.EstimatorSpec(mode, loss=network.xentropy, eval_metric_ops=eval_metric_ops)

    return _small_net_model_fn

def get_input_fn(mode=None, params=None):

    def _input_fn(mode=None, params=None):
        with tf.device('/cpu:0'):
            if mode == 'train':
                dataset = DataHandler(mode, "train.tfrecords", params).prepare_for_train()
            elif mode == 'eval':
                dataset = DataHandler(mode, "eval.tfrecords", params).prepare_for_eval(params.batch_size)
            elif mode == 'test':
                print('test access')
                dataset = DataHandler(mode, "test.tfrecords", params).prepare_for_eval(params.batch_size)
            else:
                raise ValueError('_input_fn received invalid MODE')
            return dataset.make_one_shot_iterator().get_next()

    return _input_fn

def build_estimator(num_gpus, run_config, hparams):
    classifier = tf.estimator.Estimator(model_fn=get_model_fn(num_gpus),
        config=run_config,
        params=hparams)
    return classifier

def main(**hparams):
    # conveniently convert data_dimensions argument from string to list now to save on computation later
    hparams['data_dims'] = list(map(int, hparams['data_dims'].split(',')))

    tf.logging.set_verbosity(tf.logging.INFO)

    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
        device_count={"CPU": hparams['num_cores'], "GPU": hparams['num_gpus']},
        gpu_options=tf.GPUOptions(force_gpu_compatible=True)
    )
    config = tf.estimator.RunConfig(
        model_dir=os.path.join(hparams['output_dir'], str(hparams['job_id'])),
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
        hparams['num_gpus'],
        run_config=config,
        hparams=tf.contrib.training.HParams(**hparams)
    )

    tf.estimator.train_and_evaluate(
        classifier,
        tf.estimator.TrainSpec(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.TRAIN), max_steps=hparams['train_steps']),
        tf.estimator.EvalSpec(input_fn=get_input_fn(mode=tf.estimator.ModeKeys.EVAL), steps=None)
    )

    classifier.evaluate(input_fn=get_input_fn(mode='test'), name='test')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--data-dir',
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
        '-d', '--dimensions',
        type=str,
        required=True,
        help='The dimensions of the dataset examples in HEIGHT,WIDTH,DEPTH format.',
        dest='data_dims')
    parser.add_argument(
        '-j', '--job-id',
        type=int,
        required=True,
        help='The id this job was assigned during submission, alternatively any unique number distinguish runs.',
        dest='job_id')
    parser.add_argument(
        '-n', '--num-gpus',
        type=int,
        default=1,
        help='The number of gpus used. Uses only CPU if set to 0.',
        dest='num_gpus')
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
        '-p', '--preprocess-data',
        type=bool,
        default=False,
        help='Whether or not the input data for training should be preprocessed',
        dest='preprocess_data')
    parser.add_argument(
        '-z', '--preprocess-zoom',
        type=float,
        default=1.25,
        help='Zoom factor for pad and crop performed during preprocessing.',
        dest='preprocess_zoom'
    )
    args = parser.parse_args()

    main(**vars(args))
