# SpeakerVerificaiton
Speaker verification with zalo ai challenge dataset
## Dependencies
```
pip install -r requirements.txt
```

## Data preparation
1. Download the [public dataset](https://dl.challenge.zalo.ai/voice-verification/data/Train-Test-Data_v2.zip)
2. Put to the raw_dataset folder wiht format raw_dataset/speakerid/*.wav
3. Prepare dataset (split, extract feature):
```
python dataprep.py
```

## Training
Training from scratch
```
python main.py
```

## Testing and inference
[in progress] 😅

## Citation
...
