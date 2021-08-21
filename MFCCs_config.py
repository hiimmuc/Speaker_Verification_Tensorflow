class mfcc_config:
    sampling_rate = 16000
    max_pad_length = 400  # calculate by avg time length / time overlap - 1 for st :v
    n_mfcc = 400  # win_length
    n_fft = 512
    hop_length = 160
    n_mels = 64


NUM_CLASSES = 400
