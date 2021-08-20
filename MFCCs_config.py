class mfcc_config:
    sampling_rate = 16000
    max_pad_length = 256  # calculate by avg time length / time overlap - 1 for st :v
    n_mfcc = 40
    n_fft = 1024
    hop_length = 512
    n_mels = 256
    num_classes = 400
