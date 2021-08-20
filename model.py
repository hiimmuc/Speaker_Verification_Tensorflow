import importlib

import numpy as np
from tensorflow.keras.optimizers import SGD, Adam

from models import *


def get_net(args, **kwargs):
    if 'resnet' in args.model:
        model_name = 'resnet'
    elif 'vgg' in args.model:
        model_name = 'vgg'
    else:
        raise ValueError('Unknown model name')
    net = importlib.import_module('models.{}'.format(
        model_name)).__getattribute__('construct_net')
    return net


def define_model(args, input_shape, num_classes, summary=True):
    learning_rate = args.learning_rate
    if args.optimizer == 'SGD':
        momentum = args.momentum
        nesterov = args.nesterov
        opt = SGD(learning_rate=learning_rate,
                  momentum=momentum, nesterov=nesterov)
    elif args.optimizer == 'Adam':
        epsilon = args.adam_epsilon
        opt = Adam(learning_rate=learning_rate, epsilon=epsilon)
    net = get_net(args)
    model = net(args, input_shape, num_classes)
    if summary:
        model.summary()
    if 'resnet' in args.model:
        opt = args.optimizer
    model.compile(loss='categorical_crossentropy',
                  metrics=['accuracy'], optimizer=opt)

    return model


if __name__ == '__main__':
    pass
