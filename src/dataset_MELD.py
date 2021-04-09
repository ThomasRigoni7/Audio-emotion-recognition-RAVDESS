import torch
from torch.utils import data
import librosa
import numpy as np
import pandas as pd
import random
from pathlib import Path
import utils


class MELD_DATA(data.Dataset):
    def _load_csv(self, csv_path):
        try:
            data = pd.read_csv(csv_path)
        except OSError as oserr:
                raise OSError(
                    "csv file not found in '{}', did you download the dataset and extract it?".format(csv_path)) from oserr
        return data
    
    def _load_files(self, data_files):
        print("Loading dataset...")
        files = []
        for i in files:
            print(i)
        minlen = 2000000
        for i, f in enumerate(data_files):
            print("{}/{}".format(i, len(data_files)), end="\r")
            # extract the emotion and convert it into a number
            label = self.emotion_to_code[f[3]]

            # f[5] and f[6] contain the dialogue and utterance of the file
            filename = "dia{}_utt{}.wav".format(f[5], f[6])
            filepath = Path(filename)
            if self.data_dir is not None:
                filepath = (Path(self.data_dir) / filepath).with_suffix(self.in_suffix)
            else:
                filepath = Path(filepath).with_suffix(self.in_suffix)
            try:
                x, sr = utils.load_file(filepath, self.sr)
                x = utils.apply_transformations(x, self.transformations, sr)
                minlen = min(minlen, x.shape[-1])
                files.append((x, int(label)))
            except RuntimeError as re:
                print(re)
        if self.chunk_len is None:
            self.chunk_len = minlen
        print("---DONE---")
        return files
    
    def __init__(self, csv_path, data_dir=None, chunk_len=None, random_load=True, in_suffix=".wav", transformations=[], sr=22050, save_path=None):
        super(MELD_DATA, self).__init__()
        self.chunk_len = chunk_len
        self.random_load = random_load
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.in_suffix = in_suffix
        self.transformations = transformations
        self.sr = sr
        self.classes = ["neutral", "joy", "sadness", "anger", "fear", "disgust", "surprise"]
        self.emotion_to_code = {"neutral": 0, "joy": 1, "sadness": 2, "anger": 3, "fear": 4, "disgust": 5, "surprise": 6}
        dataframe = self._load_csv(csv_path)
        self.files = self._load_files(dataframe.values)

        

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
    
    def get_class_sample_count(self):
        count =np.zeros(len(self.classes), dtype=int)
        for s in self.files:
            count[s[1]] +=1
        weight = 1. /count
        sample_weight = []
        for s in self.files:
            sample_weight.append(weight[s[1]]) 
        return count, torch.Tensor(sample_weight)


if __name__ == "__main__":
    mydata = MELD_DATA('./MELD_dataset/MELD.Raw/test_sent_emo.csv', data_dir="./MELD_dataset/test_data/")
    
    params = {'batch_size': 5,
              'shuffle': False}
    test_set_generator = data.DataLoader(mydata, **params)

    for x, y in test_set_generator:
        print(y)
        print(x.shape)
        break