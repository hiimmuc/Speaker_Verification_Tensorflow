import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import callbacks
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.models import load_model

from model import *

# GPU settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)


class Trainer:
    def __init__(self, args, train_dts, dev_dts, test_dts):
        self.args = args
        self.save_dir = self.args.save_dir
        self.ckpt_path = os.path.join(
            self.save_dir, '{}_weights_best.hdf5'.format(self.args.model))
        self.test_dts = test_dts
        self.x_val, self.y_val = dev_dts
        self.x_train, self.y_train = train_dts
        self.model = define_model(
            args, input_shape=self.x_train.shape[1:], num_classes=400, summary=False)

    def train(self, val_on_train=True):
        checkpoint_path = self.ckpt_path
        if self.args.use_pretrained:
            if os.path.exists(checkpoint_path):
                self.model.load_weights(checkpoint_path)

        # Calculate pre-training accuracy
        score = self.model.evaluate(self.x_val, self.y_val, verbose=1)
        accuracy = 100 * score[1]
        print("Pre-training accuracy: %.4f%%" % accuracy)
        # Train the model
        num_epochs = self.args.epochs
        num_batch_size = self.args.train_batch_size
        #  set callback
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)
        es_callback = EarlyStopping(
            monitor='val_loss', patience=10, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.85, patience=5, verbose=1, mode='min', min_lr=1e-6)
        callbacks = [checkpoint]
        if not self.args.early_stop_off:
            callbacks.append(es_callback)
        if not self.args.reduce_lr_off:
            callbacks.append(reduce_lr)
        # training
        if val_on_train:
            history = self.model.fit(self.x_train, self.y_train, batch_size=num_batch_size, epochs=num_epochs,
                                     validation_split=0.15,
                                     shuffle=True,
                                     callbacks=callbacks)
        else:
            history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                                     batch_size=num_batch_size,
                                     epochs=num_epochs,
                                     callbacks=callbacks)

        self.model.save(os.path.join(
            self.save_dir, '{}_last_epoch.h5'.format(self.args.model)))

        self.history = history

    def evaluate(self):
        # Evaluating the model on the training and testing set
        score = self.model.evaluate(
            self.x_train, self.y_train, batch_size=self.args.eval_batch_size, verbose=0)
        print("Training Accuracy: ", score[1])
        score = self.model.evaluate(
            self.x_val, self.y_val, batch_size=self.args.eval_batch_size, verbose=0)
        print("Validation Accuracy: ", score[1])

    def load_model(self):
        self.model = load_model(self.ckpt_path)

    def plot_graph(self):
        # history = np.load(os.path.join(self.save_dir, '/trainHistoryDict'), allow_pickle=True).items()
        history = self.history.history
        plt.figure()
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.savefig(self.save_dir+f'/{self.args.model}_accuracy.png')
        # plt.clf()
        # Plot training & validation loss values
        plt.figure()
        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        plt.show()
        plt.savefig(self.save_dir+f'/{self.args.model}_loss.png')
