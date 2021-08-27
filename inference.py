import csv
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from scipy import spatial
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.optimizers import Adam
from tqdm.notebook import tqdm

from data_prep import FeatureExtraction
from model import *
from ultis import *

print_interval = 10
START = 0
STOP = 40000

class Inference():
    def __init__(self, args, model_path, model_name, test_set=None, summary=False):
        self.args = args
        self.model_path = model_path
        self.model_name = model_name

        self.x_test, self.y_test = test_set
        self.extrac_engine = FeatureExtraction()
        self.model = self.load_model_no_top(summary=summary)

    def get_embedding(self, wav_path):
        feat, _ = self.extrac_engine.extract_feature_frame(wav_path)
        feat = np.expand_dims(feat, axis=0)
        # print(feat,'feature size:', feat.shape)
        feat = np.asanyarray(feat).astype(np.float32)
        embedding = self.model.predict(feat)
        return embedding

    def get_score_of_pair(self, wav1, wav2):
        emb1 = self.get_embedding(wav1)
        emb2 = self.get_embedding(wav2)
        return sum(cosine_similarity(emb1, emb2))

    def load_model_no_top(self, summary=False):
        net = define_model(
            self.args, self.x_test.shape[1:], 400, self.model_path, False, summary=summary)

        model_new = tf.keras.models.Sequential()
        model_new.add(net)
        model_new.add(tf.keras.layers.GlobalAveragePooling2D())
        model_new.compile(loss='categorical_crossentropy',
                          optimizer=Adam(lr=0.0001), metrics=['accuracy'])
        if summary:
            model_new.summary()
        return model_new

    def run_eval(self, listfilename,  threshold=0.5):
        lines = []
        files = []
        feats = {}
        tstart = time.time()
        # Read all lines
        with open(listfilename) as listfile:
            while True:
                line = listfile.readline()
                if not line:
                    break
                data = line.split()

                # Append random label if missing
                if len(data) == 2:
                    data = [random.randint(0, 1)] + data

                files.append(data[1])
                files.append(data[2])
                lines.append(line)

        setfiles = list(set(files))
        setfiles.sort()
        all_scores = []
        all_labels = []
        all_trials = []
        # Read files and compute all scores
        for idx, line in enumerate(lines):
            data = line.split()

            # Append random label if missing
            if len(data) == 2:
                data = [random.randint(0, 1)] + data

            score = self.get_score_of_pair(data[1], data[2])
            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + ' ' + data[2])

            if idx % 100 == 0:
                print('Processed {}/{} files'.format(idx, len(lines)))

            # NOTE: distance for training, normalized score for evaluating and testing

            all_scores.append(score)
            all_labels.append(int(data[0]))
            all_trials.append(data[1] + " " + data[2])

            if idx % print_interval == 0:
                telapsed = time.time() - tstart
                sys.stdout.write("\rComputing %d of %d: %.2f Hz - %.4f s" %
                                 (idx, len(lines), (idx + 1) / telapsed, telapsed / (idx + 1)))
                sys.stdout.flush()

        print('\n')
        print(all_scores, all_labels, all_trials)
        print('Done!')

    def run_test(self, threshold=0.5):
        root = self.args.data_dir
        data_root = os.path.join(root, 'public_test/data_test')
        read_file = os.path.join(root, 'public-test.csv')
        write_file = os.path.join(
            root, f'submission_model_{self.args.model}.csv')
        backup_file = os.path.join(
            root, f'backup_pair_of_features_{self.args.model}_backup.npz')
        lines = []
        files = []
        feats = {}
        tstart = time.time()
        with open(read_file, newline='') as rf:
            spamreader = csv.reader(rf, delimiter=',')
            next(spamreader, None)
            for row in tqdm(spamreader):
                files.append(row[0])
                files.append(row[1])
                lines.append(row)

        setfiles = list(set(files))
        setfiles.sort()
        print('Reading files times:{}'.format(time.time() - tstart))

        with open(write_file, 'w', newline='') as wf:
            spamwriter = csv.writer(wf, delimiter=',')
            spamwriter.writerow(['audio_1', 'audio_2', 'label', 'score'])
            for idx, data in tqdm(enumerate(lines[START:STOP])):
                feat1 = self.get_embedding(os.path.join(
                    os.path.join(data_root, data[0])))
                feat2 = self.get_embedding(os.path.join(
                    os.path.join(data_root, data[1])))
                pred = abs(1 - spatial.distance.cosine(feat1, feat2))

                if pred > threshold:
                    spamwriter.writerow([data[0], data[1], 1, pred])
                else:
                    spamwriter.writerow([data[0], data[1], 0, pred])
                feats[idx] = [feat1, feat2]

        np.savez_compressed(backup_file, feats=feats)
        print('Done!')


def run_inference(args, test_dataset=None):
    model_path = os.path.join(args.save_dir, f'{args.model}_weights_best.hdf5')
    infer_engine = Inference(args, model_path=model_path,
                             model_name=args.model, test_set=test_dataset, summary=False)
    if args.do_test:
        infer_engine.run_test()
    elif args.do_eval:
        infer_engine.run_eval(test_dataset)


# def eer_eval():
#     min_thres = 0.2
#     step = 1e-1
#     diff = 1
#     EER = 0
#     EER_thres = 0
#     EER_FAR = 0
#     EER_FRR = 0
#     for thres in [min_thres + step*i for i in range(int((1-min_thres)/step))]:

#         FAR, FRR = run_test(thres)
#         print(thres, FAR, FRR)
#         # find min eer
#         if diff > abs(FAR - FRR):
#             diff = abs(FAR - FRR)
#             EER = (FAR + FRR)/2
#             EER_thres = thres
#             EER_FAR = FAR
#             EER_FRR = FRR
#     print("\nEER : %0.2f (thres:%0.2f, FAR:%0.2f, FRR:%0.2f)" %
#           (EER, EER_thres, EER_FAR, EER_FRR))
#     return EER, EER_thres, EER_FAR, EER_FRR


if __name__ == '__main__':
    pass
