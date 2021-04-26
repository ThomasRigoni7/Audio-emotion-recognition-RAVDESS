import torch
from torch.utils import data
import librosa
import numpy as np
import pandas as pd
import random
from pathlib import Path
import utils


class MOSEI_DATA(data.Dataset):
    def _load_csv(self, csv_path):
        try:
            table = pd.read_csv(csv_path)
        except OSError as oserr:
                raise ValueError(
                    "csv file not found in '{}', check the path.".format(csv_path)) from oserr
        return table.values.tolist()
    
    def _load_files(self, rows):
        print("Loading dataset...")
        files = []
        minlen = 2000000
        maxlen = 0
        for i, row in enumerate(rows):
            file, start_time, end_time, sentiment, *emotions = row
            emotions = torch.tensor(emotions)
            # put 0 if the emotion is not present, 1 otherwise
            emotions = (emotions > 0).float()
            start_time = 0 if start_time < 0 else start_time
            print("{}/{}".format(i, len(rows)), end="\r")
            filepath = Path(file)
            if self.data_dir is not None:
                filepath = (Path(self.data_dir) / filepath).with_suffix(self.in_suffix)
            else:
                filepath = Path(filepath).with_suffix(self.in_suffix)
            x, sr = utils.load_file(filepath, self.sr, start_time=start_time, end_time=end_time)
            # discard the utterances of less than 1 second and divide the ones longer than 10 secs
            splits = utils.divide_and_discard(x, sr, 1 * sr, 10 * sr)
            for split in splits:
                s = utils.apply_transformations(split, self.transformations, sr, max_len=5 * sr)
                minlen = min(minlen, s.shape[-1])
                maxlen = max(maxlen, s.shape[-1])
                files.append((s, emotions))
        print("MinLen: ", minlen)
        print("MaxLen: ", maxlen)
        if self.chunk_len is None:
            self.chunk_len = minlen
        print("---DONE---")
        return files
    
    def __init__(self, csv_path="./CMU-MOSEI_dataset/Audio/Full/labels.csv", data_dir="./CMU-MOSEI_dataset/Audio/Full/WAV_16000/", chunk_len=None, random_load=False, in_suffix=".wav", transformations=["mel", "power_to_db"], sr=None, save_folder=None):
        super(MOSEI_DATA, self).__init__()
        self.chunk_len = chunk_len
        self.random_load = random_load
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.in_suffix = in_suffix
        self.transformations = transformations
        self.sr = sr
        self.classes = ["happy","sad","anger","surprise","disgust","fear"]
        rows = self._load_csv(csv_path)
        self.files = self._load_files(rows)
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        X, y = self.files[index]
        if self.random_load:
            index = random.randint(0, X.shape[-1] - self.chunk_len)
            if X.dim() == 1:
                X_cut = X[index:index+self.chunk_len]
            elif X.dim() == 2:
                X_cut = X[:, index:index+self.chunk_len]
            return X_cut, y
        return X, y

if __name__ == "__main__":
    dataset = MOSEI_DATA(csv_path="./CMU-MOSEI_dataset/Audio/Full/prova.csv")
    for x, y in dataset:
        print(x)
        print(y)
        print(x.shape, y.shape)
        break