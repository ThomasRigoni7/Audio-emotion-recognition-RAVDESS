import torch
from torch.utils import data
import numpy as np
import pandas as pd
import random
from pathlib import Path
import utils
import h5py


class MOSEI20_DATA(data.Dataset):
    def _load_data(self, data_path, labels_path):
        audio = h5py.File(data_path, 'r')
        labels = h5py.File(labels_path, 'r')
        return (audio["d1"], labels["d1"])

    
    def __init__(self, data_path, labels_path):
        super(MOSEI20_DATA, self).__init__()
        self.classes = ["Anger", "Disgust", "Fear", "Happy", "Sad", "Surprise"]
        self.audio, self.labels = self._load_data(data_path, labels_path)

        

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, index):
        X = self.audio[index]
        y = self.labels[index]
        return X, y


if __name__ == "__main__":
    mydata = MOSEI20_DATA('CMU-MOSEI_dataset/prova/immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_20/data/audio_train.h5', 'CMU-MOSEI_dataset/prova/immortal.multicomp.cs.cmu.edu/raw_datasets/processed_data/cmu-mosei/seq_length_20/data/ey_train.h5')
    print(mydata)
    print(mydata.__len__())

    params = {'batch_size': 5,
              'shuffle': False}
    test_set_generator = data.DataLoader(mydata, **params)

    for x, y in test_set_generator:
        print(y)
        print(torch.argmax(y, dim=0))
        print(x.shape)
        break