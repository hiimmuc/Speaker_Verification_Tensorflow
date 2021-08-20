import glob
import itertools
import os
from datetime import datetime

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pylab
import tensorflow as tf
from scipy import spatial
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dense,
                                     Dropout, Flatten, GlobalAveragePooling2D)
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model, to_categorical
from tqdm.notebook import tqdm

from data_prep import FeatureExtraction
from ultis import *


def get_audio_path(folder):
    return glob.glob(os.path.join(folder, '*.wav'))


feature_extraction_engine = FeatureExtraction()

data_files = []
with open('dataset/data.txt', 'r') as f:
    data_files = f.readlines()
data_files = list(map(lambda x: x.replace('\n', ''), data_files))
# get label of audio
candidates = []
for f in data_files:
    split_name = ''.join(os.path.split(f)[1].split('-'))
    non_split_name = os.path.split(f)[1]
    candidates.append(split_name)
test_files = []
with open('test.txt', 'r') as f:
    test_files = f.readlines()
test_files = list(map(lambda x: x.replace('\n', ''), test_files))

pair_compare = []
with open('Trial.txt', 'r') as f:
    pair_compare = f.readlines()
pair_compare = list(map(lambda x: x.replace('\n', ''), pair_compare))

pair_speaker = []
answer = []
for line in pair_compare:
    spk1, spk2, same = line.split(' ')
    same = same == 'target'
    pair_speaker.append([spk1, spk2])
    answer.append(same)
# print(len(pair_speaker), len(answer))
test_candidates = []
for f in test_files:
    split_name = ''.join(os.path.split(f)[1].split('-'))
    non_split_name = os.path.split(f)[1]
    test_candidates.append(split_name)
checkpoint_path = 'backup/weights.best.hdf5'
model_path = 'backup/last_epoch.h5'


def get_full_test_path(path):
    for f in test_files:
        if path in os.listdir(f):
            return os.path.join(f, path)


def get_pair_test_path():
    pair_test_path = []
    for i, pair in enumerate(pair_speaker):
        # print(pair)
        path1 = get_full_test_path(pair[0])
        path2 = get_full_test_path(pair[1])
        pair_test_path.append([path1, path2, answer[i]])
    return pair_test_path


pair_tests = get_pair_test_path()


def infer_hand_craft(wav_path, model):
    wav_features = feature_extraction_engine.extract_features_frame(wav_path)[
        0]
    wav_features = np.expand_dims(wav_features, axis=0)
    # print(wav_features.shape)
    prediction = model.predict(wav_features)
    # print(prediction.shape)
    return prediction


def pair_wav_compare(wav_path1, wav_path2):
    feature_vec1 = infer_hand_craft(wav_path1)
    feature_vec2 = infer_hand_craft(wav_path2)
    return abs(1 - spatial.distance.cosine(feature_vec1, feature_vec2))


def run_val(thres):
    attempts = len(pair_tests)
    true_answer = 0
    for i in tqdm(range(attempts)):
        answer = pair_tests[i][2]
        predict = pair_wav_compare(pair_tests[i][0], pair_tests[i][1]) >= thres
        if predict == answer:
            true_answer += 1
        # print(true_answer/attempts)
    return true_answer/attempts


def run_test(thres):
    attempts = len(pair_tests)
    print(attempts)
    falseReject = 0
    falseAccept = 0
    for i in tqdm(range(attempts)):
        answer = pair_tests[i][2]
        predict = pair_wav_compare(pair_tests[i][0], pair_tests[i][1]) >= thres
        if not answer and predict:
            falseAccept += 1
        elif not answer and not predict:
            falseReject += 1
    return falseAccept/attempts, falseReject/attempts


def find_min_thres():
    min_thres = 1.0
    for k in tqdm(range(len(pair_tests))):
        sim = pair_wav_compare(pair_tests[k][0], pair_tests[k][1])
        if pair_tests[k][2]:
            min_thres = min(sim, min_thres)
    return min_thres


def eer_eval():
    step = 1e-1
    diff = 1
    EER = 0
    EER_thres = 0
    EER_FAR = 0
    EER_FRR = 0
    for thres in [min_thres + step*i for i in range(int((1-min_thres)/step))]:

        FAR, FRR = run_test(thres)
        print(thres, FAR, FRR)
        # find min eer
        if diff > abs(FAR - FRR):
            diff = abs(FAR - FRR)
            EER = (FAR + FRR)/2
            EER_thres = thres
            EER_FAR = FAR
            EER_FRR = FRR
    print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" %
          (EER, EER_thres, EER_FAR, EER_FRR))
    return EER, EER_thres, EER_FAR, EER_FRR


min_thres = find_min_thres()


def test_model():
    eer = eer_eval()
    acc = run_val(eer[1])
    print(acc, eer, min_thres)


if __name__ == '__main__':
    test_model()
