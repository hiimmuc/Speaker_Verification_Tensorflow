import logging
import os

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


def load_dataset(args, mode):
    data_dir = os.path.join(args.data_dir, 'feature_vectors')
    X_load, y_load = np.load(os.path.join(data_dir, "data.npy")), np.load(
        os.path.join(data_dir, "label_encode.npy"))
    x_train, x_val, y_train, y_val = train_test_split(
        X_load, y_load, test_size=0.2, random_state=42)
    x_train = np.expand_dims(x_train, axis=-1)
    x_val = np.expand_dims(x_val, axis=-1)
    if mode == 'train':
        print('loaded train set')
        return x_train, y_train
    elif mode == 'dev':
        print('loaded dev set')
        return x_val, y_val
    elif mode == 'test':
        return None
    else:
        raise 'Wrong dataset type'


def load_model_no_top(model_path, summary=True):
    cnn_test = tf.keras.models.load_model(model_path)
    cnn_test.pop()
    if summary:
        cnn_test.summary()
    new_model = tf.keras.models.Model(
        inputs=cnn_test.input, outputs=cnn_test.layers[-1].output)
    new_model.compile(optimizer='adam', loss='categorical_crossentropy')
    return new_model
