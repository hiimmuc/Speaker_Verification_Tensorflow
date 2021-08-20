from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Flatten, GlobalAveragePooling2D, Input,
                                     MaxPooling2D)
from tensorflow.keras.models import Model


class VGG(object):
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes

    def conv_block(self, x, filters, kernel_size=3, stride=1, padding='same', activation='relu', n_conv=2, pooling=True,
                   batch_norm=False):
        for _ in range(n_conv):
            x = Conv2D(filters, kernel_size=kernel_size, strides=stride,
                       padding=padding, activation=activation)(x)
            if batch_norm:
                x = BatchNormalization()(x)
        if pooling:
            x = MaxPooling2D((2, 2), strides=(2, 2))(x)
        return x

    def fc_block(self, x, filters, activation, flatten=False, n_layers=2):
        if flatten:
            x = Flatten()(x)
        else:
            x = GlobalAveragePooling2D()(x)
        for i in range(n_layers):
            x = Dense(units=filters[i], activation=activation)(x)
        return x

    def build_model_vgg16(self, summary=False):
        input_layer = Input(shape=self.input_shape)
        block1 = self.conv_block(input_layer, 64)
        block2 = self.conv_block(block1, 128)
        block3 = self.conv_block(block2, 256, n_conv=3)
        block4 = self.conv_block(block3, 512, n_conv=3)
        block5 = self.conv_block(block4, 512, n_conv=3)
        batch_norm = BatchNormalization()(block5)
        fc_block = self.fc_block(batch_norm, [4096, 4096], 'relu')
        output = Dense(self.num_classes, activation='softmax')(fc_block)

        model = Model(inputs=input_layer, outputs=output)
        if summary:
            model.summary()
        return model

    def build_model_vgg_m(self, summary=False):
        input_layer = Input(shape=self.input_shape)
        block1 = self.conv_block(
            input_layer, 32, kernel_size=7, stride=2, n_conv=1, pooling=False, batch_norm=True)
        block2 = self.conv_block(
            block1, 64, kernel_size=5, stride=1, n_conv=1, pooling=False, batch_norm=True)
        block3 = self.conv_block(
            block2, 128, n_conv=1, pooling=False, batch_norm=True)
        block4 = self.conv_block(
            block3, 256, n_conv=2, pooling=False, batch_norm=True)
        # block5 = self.conv_block(block4, 512, n_conv=3, pooling=False)
        # batch_norm = BatchNormalization()(block4)
        fc_block = self.fc_block(block4, [1024, 256], 'relu')
        output = Dense(self.num_classes, activation='softmax')(fc_block)

        model = Model(inputs=input_layer, outputs=output)
        if summary:
            model.summary()
        return model

    def build_model_custom(self, summary=False):
        input_layer = Input(shape=self.input_shape)
        block1 = self.conv_block(
            input_layer, 16, kernel_size=7, n_conv=1, pooling=False, batch_norm=True)
        block2 = self.conv_block(block1, 32, n_conv=1,
                                 pooling=False, batch_norm=True)
        block3 = self.conv_block(block2, 64, n_conv=1,
                                 pooling=False, batch_norm=True)

        block4 = self.conv_block(
            block3, 128, n_conv=1, pooling=False, batch_norm=True)
        block5 = self.conv_block(
            block4, 256, n_conv=1, pooling=False, batch_norm=True)
        block6 = self.conv_block(
            block5, 512, kernel_size=1, n_conv=1, pooling=False, batch_norm=True)
        # batch_norm = BatchNormalization()(block4)
        # fc_block = self.fc_block(block6, [1024, 256], 'relu')
        output = Dense(self.num_classes, activation='softmax')(block6)

        model = Model(inputs=input_layer, outputs=output)
        if summary:
            model.summary()
        return model


def construct_net(args, input_shape, num_classes):
    my_model = VGG(input_shape, num_classes)
    if args.model == 'vgg16':
        return my_model.build_model_vgg16(summary=False)
    elif args.model == 'vgg-m':
        return my_model.build_model_vgg_m(summary=False)
    elif args.model == 'vgg-custom':
        return my_model.build_model_custom(summary=False)
    else:
        raise 'Unknown model'


if __name__ == '__main__':
    my_model = VGG((40, 256, 1), 400)
    my_model.build_model_custom(summary=True)
