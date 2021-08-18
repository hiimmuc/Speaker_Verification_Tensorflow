from models import vgg, resnet
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np

MODELS = {'vgg-m': vgg.VGG, 'vgg16': vgg.VGG, 'custom': vgg.VGG}


def define_model(args, input_shape, num_classes, summary=False):
    # TODO: need multi selections of model
    model = MODELS[args.backbone.lower()](input_shape, num_classes).build_model_vgg_m(summary)
    learning_rate = args.learning_rate
    if args.optimizer == 'SGD':
        momentum = args.momentum
        nesterov = args.nesterov
        opt = optimizers.SGD(learning_rate=learning_rate, momentum=momentum, nesterov=nesterov)
    elif args.optimizer == 'Adam':
        epsilon = args.adam_epsilon
        opt =optimizers.Adam(learning_rate=learning_rate, epsilon=epsilon)
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer=opt)
    return model
