import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
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
        self.ckpt_path = os.path.join(self.save_dir, 'weights.best.hdf5')
        self.train_dts = train_dts
        self.dev_dts = dev_dts
        self.test_dts = test_dts
        self.x_val, self.y_val = self.dev_dts
        self.x_train, self.y_train = self.train_dts
        self.model = define_model(
            args, input_shape=self.x_train[0].shape, num_classes=400, summary=True)

    def train(self, val_on_train=True):
        checkpoint_path = self.ckpt_path
        train_new = not self.args.use_pretrained
        if not train_new:
            if os.path.exists(checkpoint_path):
                self.model.load_weights(checkpoint_path)

        # Calculate pre-training accuracy
        score = self.model.evaluate(self.x_val, self.y_val, verbose=1)
        accuracy = 100 * score[1]
        print("Pre-training accuracy: %.4f%%" % accuracy)
        # Train the model
        num_epochs = self.args.epochs
        num_batch_size = 10
        #  set callback
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path, monitor='val_loss', verbose=1, save_best_only=True)
        es_callback = EarlyStopping(
            monitor='val_loss', patience=10, mode='min', verbose=1)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, verbose=0, mode='min', min_lr=0)

        if val_on_train:
            history = self.model.fit(self.x_train, self.y_train, batch_size=num_batch_size, epochs=num_epochs,
                                     validation_split=0.15,
                                     shuffle=True,
                                     callbacks=[checkpoint, es_callback, reduce_lr])
        else:
            history = self.model.fit(self.x_train, self.y_train, validation_data=(self.x_val, self.y_val),
                                     batch_size=num_batch_size,
                                     epochs=num_epochs,
                                     callbacks=[checkpoint, es_callback, reduce_lr])

        # with open(os.path.join(self.save_dir, '/trainHistoryDict'), 'wb') as file_pi:
            # pickle.dump(history.history, file_pi)
        self.model.save(os.path.join(self.save_dir, 'last_epoch.h5'))
        # np.save(os.path.join(self.save_dir, '/trainHistoryDict'), history.history)
        # json.dump(history.history, open(os.path.join(self.save_dir, 'trainHistory'), 'w'))
        self.history = history

    def evaluate(self):
        # Evaluating the model on the training and testing set
        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        print("Training Accuracy: ", score[1])
        score = self.model.evaluate(self.x_val, self.y_val, verbose=0)
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
        plt.savefig(self.save_dir+'/accuracy.png')
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
        plt.savefig(self.save_dir+'/loss.png')
