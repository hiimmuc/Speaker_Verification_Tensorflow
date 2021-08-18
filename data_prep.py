import glob
import os
import shutil
import librosa
import librosa.display
from tqdm.auto import tqdm
import numpy as np
import MFCCs_config as cfg
from tensorflow.keras.utils import to_categorical


def get_audio_path(folder):
    return glob.glob(os.path.join(folder, '*.wav'))

# create metadata and split dataset


class MetadataCreate(object):
    def __init__(self, raw_data_path, dst):
        self.raw_path = raw_data_path
        self.dest = dst
        self.filenames = os.listdir(self.raw_path)
        self.check_invalid_folder()
        self.filenames.sort(key=lambda x: int(list(x.split('-'))[0]))
        ratio = [.8 * len(self.filenames), .9 * .8 * len(self.filenames), len(self.filenames)]
        ratio = list(map(int, ratio))
        self.train_path = self.filenames[0:ratio[0]]
        self.val_path = self.filenames[ratio[0]:ratio[1]]
        self.test_path = self.filenames[ratio[1]:ratio[2]]

    def write_files(self, f_type, f):
        with open(os.path.join(self.dest, f"{f_type}.txt"), 'w+') as fi:
            for txt in f:
                fi.write(str(os.path.join(self.raw_path, txt)) + '\n')
            fi.close()

    def check_invalid_folder(self):
        for f in tqdm(self.filenames):
            path = os.path.join(self.raw_path, f)
            for invalid in os.listdir(path):
                path_invalid = os.path.join(path, invalid)
                if os.path.isdir(path_invalid):
                    print(path_invalid, end=' ')
                    if len(os.listdir(path_invalid)) == 0:
                        print('empty', end='\n')
                        os.rmdir(path_invalid)  # remove empty folder
                    else:
                        if os.path.isdir(os.path.join(self.raw_path, invalid)):
                            for audio in os.listdir(path_invalid):
                                if audio not in os.listdir(os.path.join(self.raw_path, invalid)):
                                    # print(audio)
                                    shutil.move(src=os.path.join(path_invalid, audio),
                                                dst=os.path.join(os.path.join(self.raw_path, invalid), audio))
                            shutil.rmtree(path_invalid)

    def export_metadata(self):
        self.write_files('train', self.train_path)
        self.write_files('val', self.val_path)
        self.write_files('test', self.test_path)
        self.write_files('data', self.filenames)


# create feature vectors directory
class FeatureExtraction:
    def __init__(self, model_path, save_path):
        self.config = cfg.mfcc_config
        self.data_files = []
        self.speakers = self.get_labels()
        self.model_path = model_path
        self.save_path = save_path

    def get_labels(self, data_path='./dataset'):
        with open(os.path.join(data_path, 'data.txt'), 'r') as f:
            self.data_files = f.readlines()
        self.data_files = list(map(lambda x: x.replace('\n', ''), self.data_files))
        # get label of audio
        candidates = []
        for f in self.data_files:
            split_name = ''.join(os.path.split(f)[1].split('-'))
            non_split_name = os.path.split(f)[1]
            candidates.append(split_name)
        return candidates

    def extract_feature_frame(self, audio_path):
        try:
            """
                Load and preprocess the audio
            """
            audio, sample_rate = librosa.load(audio_path, sr=self.config.sampling_rate)
            y = audio
            """
                Convert to MFCC numpy array
            """
            max_pad_length = self.config.max_pad_length  # calculate by avg time length / time overlap - 1 for st :v
            n_mfcc = self.config.n_mfcc
            n_fft = self.config.n_fft
            hop_length = self.config.hop_length
            n_mels = self.config.n_mels
            mfccs = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length,
                                         n_mels=n_mels)
            pad_width = max_pad_length - mfccs.shape[1]
            pad_width = pad_width if pad_width >= 0 else 0
            mfccs = np.pad(mfccs[:, :max_pad_length], pad_width=((0, 0), (0, pad_width)), mode='constant')
            # print(mfccs.shape)
        except Exception as e:
            print("Error encountered while parsing file: ", e)
            return None, self.config.sampling_rate
        return mfccs, sample_rate

    def process_raw_dataset(self):
        print("start text independent utterance feature extraction")
        if not isinstance(self.data_files, list):
            print("wrong format of folder path")
            return None, None
        total_speaker_num = len(self.data_files)
        print("total speaker number : %d" % total_speaker_num)
        features = []
        labels = []
        for i, folder in enumerate(self.data_files):
            audio_files = get_audio_path(folder=folder)  # same speaker -> same label
            label = self.speakers[i]
            print(f"\n{i}th speaker, id {label}, is processing, total {len(audio_files)} files")
            for j, audio_path in tqdm(enumerate(audio_files)):
                data, sr = self.extract_feature_frame(audio_path=audio_path)
                if data is not None:
                    features.append(data)
                    labels.append(label)
                else:
                    print("Data is empty: ", audio_path)
        X = np.array(features)
        Y = np.array(labels)
        # one hot label
        y_l = Y.copy()
        y_ = [self.speakers.index(x) for x in Y.tolist()]
        y_encode = to_categorical(y_, len(self.speakers))
        # print(y_encode.shape)
        Y = np.array(y_encode)
        return X, Y, y_l

    def save_as_ndarray(self, X, y, y_encode):
        os.makedirs(self.model_path, exist_ok=True)   # make folder to save train file
        os.makedirs(self.save_path, exist_ok=True)    # make folder to save test file
        np.save(os.path.join(self.save_path, "data.npy"), X)
        np.save(os.path.join(self.save_path, "label_encode.npy"), y_encode)
        np.save(os.path.join(self.save_path, "label.npy"), y)


if __name__ == '__main__':
    model_path = './backup/model'
    save_path = './dataset/feature_vectors'
    # metadata_gen = MetadataCreate('dataset/raw_dataset', 'dataset')
    # metadata_gen.export_metadata()
    # data_gen = FeatureExtraction(model_path, save_path)
    # x, y, y_l = data_gen.process_raw_dataset()
    # data_gen.save_as_ndarray(x, y_l, y)
    pass
