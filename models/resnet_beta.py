import os

import tensorflow as tf
from MFCCs_config import NUM_CLASSES
from numpy.core.numeric import True_
from tensorflow import keras
from tensorflow.keras import backend, layers
from tensorflow.keras.applications.imagenet_utils import (decode_predictions,
                                                          preprocess_input)
from tensorflow.keras.utils import get_source_inputs
from tensorflow.python.keras.applications import imagenet_utils


def ResNet(stack_fn,
           preact=False,
           use_bias=True,
           model_name='resnet',
           include_top=True,
           weights=None,
           input_tensor=None,
           input_shape=None,
           pooling=None,
           classifier_activation='softmax',
           classes=NUM_CLASSES, **kwargs):
    """Instantiates the ResNet, ResNetV2, and ResNeXt architecture.

    Args:
      stack_fn: a function that returns output tensor for the
        stacked residual blocks.
      preact: whether to use pre-activation or not
        (True for ResNetV2, False for ResNet and ResNeXt).
      use_bias: whether to use biases for convolutional layers or not
        (True for ResNet and ResNetV2, False for ResNeXt).
      model_name: string, model name.
      include_top: whether to include the fully-connected
        layer at the top of the network.
      weights: one of `None` (random initialization),
        'imagenet' (pre-training on ImageNet),
        or the path to the weights file to be loaded.
      input_tensor: optional Keras tensor
        (i.e. output of `layers.Input()`)
        to use as image input for the model.
      input_shape: optional shape tuple, only to be specified
        if `include_top` is False (otherwise the input shape
        has to be `(224, 224, 3)` (with `channels_last` data format)
        or `(3, 224, 224)` (with `channels_first` data format).
        It should have exactly 3 inputs channels.
      pooling: optional pooling mode for feature extraction
        when `include_top` is `False`.
        - `None` means that the output of the model will be
            the 4D tensor output of the
            last convolutional layer.
        - `avg` means that global average pooling
            will be applied to the output of the
            last convolutional layer, and thus
            the output of the model will be a 2D tensor.
        - `max` means that global max pooling will
            be applied.
      classes: optional number of classes to classify images
        into, only to be specified if `include_top` is True, and
        if no `weights` argument is specified.
      classifier_activation: A `str` or callable. The activation function to use
        on the "top" layer. Ignored unless `include_top=True`. Set
        `classifier_activation=None` to return the logits of the "top" layer.
        When loading pretrained weights, `classifier_activation` can only
        be `None` or `"softmax"`.
      **kwargs: For backwards compatibility only.

    Returns:
      A `keras.Model` instance.
    """

    input_shape = imagenet_utils.obtain_input_shape(
        input_shape,
        default_size=224,
        min_size=32,
        data_format=backend.image_data_format(),
        require_flatten=include_top,
        weights=weights)

    if input_tensor is None:
        img_input = layers.Input(shape=input_shape)
    else:
        if not backend.is_keras_tensor(input_tensor):
            img_input = layers.Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    x = preprocess_input(img_input, mode='tf')  # add preprocessing input here
    x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(x)
    # x = layers.ZeroPadding2D(padding=(3, 3), name='conv1_pad')(img_input)
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(
        2, 2), use_bias=use_bias, name='conv1_conv')(x)
    if not preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='conv1_bn')(x)
        x = layers.Activation('relu', name='conv1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='pool1_pad')(x)
    x = layers.MaxPooling2D(3, strides=2, name='pool1_pool')(x)

    x = stack_fn(x)
    if preact:
        x = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name='post_bn')(x)
        x = layers.Activation('relu', name='post_relu')(x)

    if include_top:
        x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        # imagenet_utils.validate_activation(classifier_activation, weights)
        x = layers.Dense(units=classes, activation=str(classifier_activation),
                         name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D(name='avg_pool')(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D(name='max_pool')(x)
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    # create model
    model = keras.Model(inputs, x, name=model_name)

    # load weights
    if weights is not None:
        if os.path.exists(weights):
            model.load_weights(weights)

    return model


def residual_block1(x, filters,
                    kernel_size=3,
                    strides=1,
                    conv_shortcut=True,
                    name=None):
    """A residual block.
    1 x 1 64 conv -> 3 x 3 64 conv -> 1 x 1 64 conv
    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=strides, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, strides=strides, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(
        filters, kernel_size, padding='SAME', name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack_residual_block1(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = residual_block1(x, filters, strides=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = residual_block1(x, filters, conv_shortcut=False,
                            name=name + '_block' + str(i))
    return x


def residual_block2(x, filters,
                    kernel_size=3,
                    strides=1,
                    conv_shortcut=False,
                    name=None):
    """A residual block.
    1 x 1 64 conv -> 3 x 3 64 conv -> 1 x 1 64 conv
    Args:
        x: input tensor.
        filters: integer, filters of the bottleneck layer.
        kernel_size: default 3, kernel size of the bottleneck layer.
        stride: default 1, stride of the first layer.
        conv_shortcut: default False, use convolution shortcut if True,
          otherwise identity shortcut.
        name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    preact = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_preact_bn')(x)
    preact = layers.Activation('relu', name=name + '_preact_relu')(preact)

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=strides, name=name + '_0_conv')(preact)
    else:
        shortcut = layers.MaxPooling2D(
            1, strides=strides)(x) if strides > 1 else x

    x = layers.Conv2D(
        filters, 1, strides=1, use_bias=False, name=name + '_1_conv')(preact)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        use_bias=False,
        name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(4 * filters, 1, name=name + '_3_conv')(x)
    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack_residual_block2(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = residual_block2(x, filters, conv_shortcut=True, name=name + '_block1')
    for i in range(2, blocks):
        x = residual_block2(x, filters, name=name + '_block' + str(i))
    x = residual_block2(x, filters, strides=stride1,
                        name=name + '_block' + str(blocks))
    return x


def residual_block3(x, filters,
                    kernel_size=3,
                    strides=1,
                    groups=32,
                    conv_shortcut=False,
                    name=None):
    """A residual block.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer.
      kernel_size: default 3, kernel size of the bottleneck layer.
      stride: default 1, stride of the first layer.
      groups: default 32, group size for grouped convolution.
      conv_shortcut: default True, use convolution shortcut if True,
          otherwise identity shortcut.
      name: string, block label.

    Returns:
      Output tensor for the residual block.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            (64 // groups) * filters,
            1,
            strides=strides,
            use_bias=False,
            name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters, 1, use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    c = filters // groups
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name=name + '_2_pad')(x)
    x = layers.DepthwiseConv2D(
        kernel_size,
        strides=strides,
        depth_multiplier=c,
        use_bias=False,
        name=name + '_2_conv')(x)
    x_shape = backend.shape(x)[:-1]
    x = backend.reshape(x, backend.concatenate([x_shape, (groups, c, c)]))
    x = layers.Lambda(
        lambda x: sum(x[:, :, :, :, i] for i in range(c)),
        name=name + '_2_reduce')(x)
    x = backend.reshape(x, backend.concatenate([x_shape, (filters,)]))
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Conv2D(
        (64 // groups) * filters, 1, use_bias=False, name=name + '_3_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_3_bn')(x)

    x = layers.Add(name=name + '_add')([shortcut, x])
    x = layers.Activation('relu', name=name + '_out')(x)
    return x


def stack_residual_block3(x, filters, blocks, stride1=2, groups=32, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      groups: default 32, group size for grouped convolution.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = residual_block3(x, filters, strides=stride1,
                        groups=groups, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = residual_block3(
            x,
            filters,
            groups=groups,
            conv_shortcut=False,
            name=name + '_block' + str(i))
    return x


def basic_block(x, filters,
                kernel_size=3,
                strides=1,
                conv_shortcut=False,
                name=None):
    """Basic 3 X 3 convolution blocks.
    3x3 conv -> batch norm -> ReLU -> 3x3 conv -> batch norm -> ReLU
    Returns:
      Output tensor for the stacked blocks.
    """
    bn_axis = 3 if backend.image_data_format() == 'channels_last' else 1

    if conv_shortcut:
        shortcut = layers.Conv2D(
            4 * filters, 1, strides=strides, name=name + '_0_conv')(x)
        shortcut = layers.BatchNormalization(
            axis=bn_axis, epsilon=1.001e-5, name=name + '_0_bn')(shortcut)
    else:
        shortcut = x

    x = layers.Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=strides, padding='same',
                      use_bias=False, name=name + '_1_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_1_bn')(x)
    x = layers.Activation('relu', name=name + '_1_relu')(x)

    x = layers.Conv2D(filters=4 * filters,
                      kernel_size=kernel_size,
                      strides=(1, 1), padding='same',
                      use_bias=False, name=name + '_2_conv')(x)
    x = layers.BatchNormalization(
        axis=bn_axis, epsilon=1.001e-5, name=name + '_2_bn')(x)
    x = layers.Activation('relu', name=name + '_2_relu')(x)

    x = layers.Add(name=name + '_out')([shortcut, x])
    return x


def stack_basic_block(x, filters, blocks, stride1=2, name=None):
    """A set of stacked residual blocks.

    Args:
      x: input tensor.
      filters: integer, filters of the bottleneck layer in a block.
      blocks: integer, numbers of blocks in the stacked blocks.
      stride1: default 2, stride of the first layer in the first block.
      name: string, stack label.

    Returns:
      Output tensor for the stacked blocks.
    """
    x = basic_block(x, filters, conv_shortcut=True,
                    strides=stride1, name=name + '_block1')
    for i in range(2, blocks + 1):
        x = basic_block(x, filters, name=name + '_block' + str(i))

    return x


def ResNet18(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=NUM_CLASSES, **kwargs):
    """Instantiates the ResNet18 architecture."""

    def stack_fn(x):
        x = stack_basic_block(x, 64, 2, stride1=1, name='conv2')
        x = stack_basic_block(x, 128, 2, name='conv3')
        x = stack_basic_block(x, 256, 2, name='conv4')
        return stack_basic_block(x, 512, 2, name='conv5')

    return ResNet(stack_fn=stack_fn,
                  preact=False,
                  use_bias=True,
                  model_name='resnet18',
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  input_shape=input_shape,
                  pooling=pooling,
                  classes=classes, **kwargs)


def ResNet34(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=NUM_CLASSES,
             **kwargs):
    """Instantiates the ResNet34 architecture."""

    def stack_fn(x):
        x = stack_basic_block(x, 64, 3, stride1=1, name='conv2')
        x = stack_basic_block(x, 128, 4, name='conv3')
        x = stack_basic_block(x, 256, 6, name='conv4')
        return stack_basic_block(x, 512, 3, name='conv5')

    return ResNet(stack_fn=stack_fn,
                  preact=False,
                  use_bias=True,
                  model_name='resnet34',
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  input_shape=input_shape,
                  pooling=pooling,
                  classes=classes, **kwargs)


def ResNet50(include_top=True,
             weights=None,
             input_tensor=None,
             input_shape=None,
             pooling=None,
             classes=NUM_CLASSES,
             **kwargs):
    """Instantiates the ResNet50 architecture."""

    def stack_fn(x):
        x = stack_residual_block1(x, 64, 3, stride1=1, name='conv2')
        x = stack_residual_block1(x, 128, 4, name='conv3')
        x = stack_residual_block1(x, 256, 6, name='conv4')
        return stack_residual_block1(x, 512, 3, name='conv5')

    return ResNet(stack_fn=stack_fn,
                  preact=False,
                  use_bias=True,
                  model_name='resnet50',
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  input_shape=input_shape,
                  pooling=pooling,
                  classes=classes, **kwargs)


def ResNet101(include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=NUM_CLASSES,
              **kwargs):
    """Instantiates the ResNet101 architecture."""

    def stack_fn(x):
        x = stack_residual_block1(x, 64, 3, stride1=1, name='conv2')
        x = stack_residual_block1(x, 128, 4, name='conv3')
        x = stack_residual_block1(x, 256, 23, name='conv4')
        return stack_residual_block1(x, 512, 3, name='conv5')

    return ResNet(stack_fn=stack_fn,
                  preact=False,
                  use_bias=True,
                  model_name='resnet101',
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  input_shape=input_shape,
                  pooling=pooling,
                  classes=classes, **kwargs)


def ResNet152(include_top=True,
              weights=None,
              input_tensor=None,
              input_shape=None,
              pooling=None,
              classes=NUM_CLASSES,
              **kwargs):
    """Instantiates the ResNet152 architecture."""

    def stack_fn(x):
        x = stack_residual_block1(x, 64, 3, stride1=1, name='conv2')
        x = stack_residual_block1(x, 128, 8, name='conv3')
        x = stack_residual_block1(x, 256, 36, name='conv4')
        return stack_residual_block1(x, 512, 3, name='conv5')

    return ResNet(stack_fn=stack_fn,
                  preact=False,
                  use_bias=True,
                  model_name='resnet52',
                  include_top=include_top,
                  weights=weights,
                  input_tensor=input_tensor,
                  input_shape=input_shape,
                  pooling=pooling,
                  classes=classes, **kwargs)


def ResNet50V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=NUM_CLASSES,
        classifier_activation='softmax'):
    """Instantiates the ResNet50V2 architecture."""
    def stack_fn(x):
        x = stack_residual_block2(x, 64, 3, name='conv2')
        x = stack_residual_block2(x, 128, 4, name='conv3')
        x = stack_residual_block2(x, 256, 6, name='conv4')
        return stack_residual_block2(x, 512, 3, stride1=1, name='conv5')

    return ResNet(
        stack_fn,
        True,
        True,
        'resnet50v2',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation)


def ResNet101V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=NUM_CLASSES,
        classifier_activation='softmax'):
    """Instantiates the ResNet101V2 architecture."""
    def stack_fn(x):
        x = stack_residual_block2(x, 64, 3, name='conv2')
        x = stack_residual_block2(x, 128, 4, name='conv3')
        x = stack_residual_block2(x, 256, 23, name='conv4')
        return stack_residual_block2(x, 512, 3, stride1=1, name='conv5')

    return ResNet(
        stack_fn,
        True,
        True,
        'resnet101v2',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation)


def ResNet152V2(
        include_top=True,
        weights=None,
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=NUM_CLASSES,
        classifier_activation='softmax'):
    """Instantiates the ResNet152V2 architecture."""
    def stack_fn(x):
        x = stack_residual_block2(x, 64, 3, name='conv2')
        x = stack_residual_block2(x, 128, 8, name='conv3')
        x = stack_residual_block2(x, 256, 36, name='conv4')
        return stack_residual_block2(x, 512, 3, stride1=1, name='conv5')

    return ResNet(
        stack_fn,
        True,
        True,
        'resnet152v2',
        include_top,
        weights,
        input_tensor,
        input_shape,
        pooling,
        classes,
        classifier_activation=classifier_activation)


def preprocess_input_resnet(x, data_format=None):
    return preprocess_input(
        x, data_format=data_format, mode='tf')


def decode_predictions_resnet(preds, top=5):
    return decode_predictions(preds, top=top)


def construct_net(args, input_shape, num_classes, **kwargs):
    if args.model == 'resnet18':
        return ResNet18(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    elif args.model == 'resnet34':
        return ResNet34(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    elif args.model == 'resnet50':
        return ResNet50(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    elif args.model == 'resnet101':
        return ResNet101(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    elif args.model == 'resnet152':
        return ResNet152(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    elif args.model == 'resnet50v2':
        return ResNet50V2(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    elif args.model == 'resnet101v2':
        return ResNet101V2(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    elif args.model == 'resnet152v2':
        return ResNet152V2(input_shape=input_shape, weights=None, classes=num_classes, **kwargs)
    else:
        raise 'Invalid model name'
