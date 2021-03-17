import torch
from torch.utils import data
from scipy import signal
import librosa
import numpy as np
import pandas as pd
import random
from pathlib import Path


class RAVDESS_DATA(data.Dataset):
    def __init__(self, csv_path, device, data_dir="./RAVDESS_dataset/mels/", chunk_len=153, random_load=True):
        super(RAVDESS_DATA, self).__init__()
        self.chunk_len = chunk_len
        self.random_load = random_load
        try:
            data = pd.read_csv(csv_path)
        except OSError as oserr:
                raise RuntimeError(
                    "csv file not found in '{}', did you download the dataset and generate the files with genfiles.py?".format(csv_path)) from oserr
        filenames = data.values.tolist()
        self.files = []
        for file, label in filenames:
            filepath = Path(file)
            if data_dir is not None:
                filepath = Path(data_dir) / filepath
            try:
                f = open(filepath, 'rb')
            except OSError as oserr:
                raise RuntimeError(
                    "Data file not found in '{}', did you download the dataset and generate the files with genfiles.py?".format(filepath)) from oserr
            with f:
                self.files.append((torch.load(f), int(label) - 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        X, y = self.files[index]
        if self.random_load:
            index = random.randint(0, 98)
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
