import torch
from torch.utils import data
from scipy import signal
import librosa
import numpy as np
import pandas as pd
import random
from pathlib import Path




class RAVDESS_DATA(data.Dataset):
    def _load_csv(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
        except OSError as oserr:
                raise RuntimeError(
                    "csv file not found in '{}', did you download the dataset and generate the files with genfiles.py?".format(csv_path)) from oserr
        return data.values.tolist()
    
    def _load_files(self, filenames):
        print("Loading dataset...")
        files = []
        for i, (file, label) in enumerate(filenames):
            print("{}/{}".format(i, len(filenames)), end="\r")
            filepath = Path(file)
            if self.data_dir is not None:
                filepath = (Path(self.data_dir) / filepath).with_suffix(self.in_suffix)
            try:
                f = open(filepath, 'rb')
            except OSError as oserr:
                raise RuntimeError(
                    "Data file not found in '{}', did you download the dataset and generate the files with genfiles.py?".format(filepath)) from oserr
            with f:
                if self.in_suffix == ".pt":
                    x = torch.load(f)
                elif self.in_suffix == ".wav":
                    x, sr = librosa.load(f, sr=1000)
                    # print("#samples, sample rate = ", x.shape[0], sr)
                else:
                    raise RuntimeError(
                    "The suffix '{}' is not valid.".format(self.in_suffix)) from oserr
                if self.transformations == "none":
                    pass
                elif self.transformations == "mel":
                    x = librosa.feature.melspectrogram(
                        y=waveform, sr=sr, n_mels=40)
                    x = librosa.power_to_db(mel_spectogram)
                else:
                    raise RuntimeError(
                    "The transformations '{}' are not valid.".format(self.transformations)) from oserr
                if self.in_suffix == ".wav":
                    x = torch.from_numpy(x)
                if self.device is not None:
                    x.to(self.device)
                files.append((x, int(label) - 1))
        print("---DONE---")
        return files
    
    def __init__(self, csv_path, device=None, data_dir="./RAVDESS_dataset/mels/", chunk_len=153, random_load=True, in_suffix=".pt", transformations="none"):
        super(RAVDESS_DATA, self).__init__()
        self.chunk_len = chunk_len
        self.random_load = random_load
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.device = device
        self.in_suffix = in_suffix
        self.transformations = transformations
        filenames = self._load_csv(csv_path)
        self.files = self._load_files(filenames)
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        X, y = self.files[index]
        if self.random_load:
            index = random.randint(0, 98)
            if X.dim() == 1:
                X_cut = X[index:index+self.chunk_len]
            elif X.dim() == 2:
                X_cut = X[:, index:index+self.chunk_len]
            return X_cut, y
        return X, y


if __name__ == "__main__":
    mydata = RAVDESS_DATA('./RAVDESS_dataset/train_data.csv',
                          device=torch.device("cpu"))
    print(mydata)
    print(mydata.__len__())

    params = {'batch_size': 5,
              'shuffle': False}
    test_set_generator = data.DataLoader(mydata, **params)

    for x, y in test_set_generator:
        print(y)
        print(x.shape)
        break
