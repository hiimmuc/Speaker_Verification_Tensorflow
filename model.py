import importlib

import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import plot_model

from models import *


def get_net(args, **kwargs):
    if 'resnet' in args.model:
        model_name = 'resnet_beta'
    elif 'vgg' in args.model:
        model_name = 'vgg'
    else:
        raise ValueError('Unknown model name')
    net = importlib.import_module('models.{}'.format(
        model_name)).__getattribute__('construct_net')
    return net


def define_model(args, input_shape, num_classes, weights=None, include_top=True, plot_model_graph=True, summary=True):
    # optimizer
    learning_rate = args.learning_rate
    if args.optimizer == 'SGD':
        momentum = args.momentum
        nesterov = args.nesterov
        opt = SGD(learning_rate=learning_rate,
                  momentum=momentum, nesterov=nesterov)
    elif args.optimizer == 'Adam':
        epsilon = args.adam_epsilon
        opt = Adam(learning_rate=learning_rate, epsilon=epsilon)

    # Loss
    loss_function = tf.keras.losses.CategoricalCrossentropy()

    # metric
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.CategoricalAccuracy(
        name='train_accuracy')

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.CategoricalAccuracy(name='val_accuracy')

    metric_accuracy = tf.keras.metrics.Accuracy(name='accuracy')
    # get network

    net = get_net(args)
    model = net(args, input_shape, num_classes,
                weights=weights, include_top=include_top)

    if summary:
        model.summary()
    if plot_model_graph:
        save_img = args.save_dir + '/' + args.model + '_model.png'
        plot_model(model, to_file=save_img,
                   show_shapes=True, show_layer_names=True)
    # complile model
    model.compile(optimizer=opt, loss=loss_function, metrics=['accuracy'])

    return model


if __name__ == '__main__':
    pass
