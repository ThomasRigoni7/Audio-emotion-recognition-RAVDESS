import torch
from torch.utils import data
import numpy as np
import pandas as pd
import random
from pathlib import Path
import utils
from tabulate import tabulate
import os
import specaugment.spec_augment_pytorch as specaugment

def get_class_count(wav_path):
    totcount = [0. for i in range(8)]
    for subdir, dirs, files in os.walk(wav_path):
        for file in files:
            label = int(file[6:8])
            totcount[label - 1] += 1

    print(tabulate([totcount],
                    headers=[i for i in range(1, 9)]))


class RAVDESS_DATA(data.Dataset):
    def _load_csv(self, csv_path):
        try:
            table = pd.read_csv(csv_path)
        except OSError as oserr:
                raise ValueError(
                    "csv file not found in '{}', did you download the dataset and generate the files with genfiles.py?".format(csv_path)) from oserr
        return table.values.tolist()
    
    def _load_files(self, filenames):
        print("Loading dataset...")
        files = []
        minlen = 2000000
        maxlen_before = 0
        for i, (file, label) in enumerate(filenames):
            # label goes from 1 to 8
            label = int(label) - 1

            if label in self.classes_to_use:
                print("{}/{}".format(i, len(filenames)), end="\r")
                filepath = Path(file)
                if self.data_dir is not None:
                    filepath = (Path(self.data_dir) / filepath).with_suffix(self.in_suffix)
                else:
                    filepath = Path(filepath).with_suffix(self.in_suffix)
                x, sr = utils.load_file(filepath, self.sr)
                if sr is not None:
                    maxlen_before = max(maxlen_before, x.shape[-1])
                    x = utils.apply_transformations(x, self.transformations, sr, max_len= 5 * sr)
                minlen = min(minlen, x.shape[-1])
                files.append((x, torch.as_tensor(self.classes_to_use.index(label), dtype=torch.long)))
        print("MinLen: ",minlen)
        print("max before:", maxlen_before)
        if self.chunk_len is None:
            self.chunk_len = minlen
        print("---DONE---")
        return files
    
    def __init__(self, csv_path, data_dir="./RAVDESS_dataset/mels_noise2/", chunk_len=None, random_load=True, in_suffix=".pt",
                    transformations=["shift","cut","noise","mel","power_to_db"], sr=22050, classes_to_use=[0,1,2,3,4,5,6,7]):
        super(RAVDESS_DATA, self).__init__()
        self.chunk_len = chunk_len
        self.random_load = random_load
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.in_suffix = in_suffix
        self.transformations = transformations
        self.sr = sr
        self.classes = ["neutral", "calm", "happy", "sad", "angry", "fearful", "disgust", "surprised"]
        self.classes_to_use = classes_to_use
        filenames = self._load_csv(csv_path)
        self.files = self._load_files(filenames)
        
        # modify classes for cross-dataset
        self.classes = [self.classes[c] for c in classes_to_use]
        

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        x, y = self.files[index]
        if self.random_load:
            index = random.randint(0, x.shape[-1] - self.chunk_len)
            if x.dim() == 1:
                x_cut = x[index:index+self.chunk_len]
            elif x.dim() == 2:
                x_cut = x[:, index:index+self.chunk_len]
            return x_cut, y
        return x, y
    
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
    mydata = RAVDESS_DATA('./RAVDESS_dataset/csv/divided_with_validation/test_data.csv', data_dir='./RAVDESS_dataset/wav/', in_suffix=".wav", transformations=["noise", "mel", "power_to_db"], sr = 22050)
    print(mydata)
    print(mydata.__len__())

    params = {'batch_size': 5,
              'shuffle': False}
    test_set_generator = data.DataLoader(mydata, **params)
    for i,(x, y) in enumerate(test_set_generator):
        print("iteration ",i)

    '''
    for x, y in test_set_generator:
        print(y)
        print(x.shape)

        mel_spec = librosa.feature.inverse.db_to_power(x[0].numpy())
        wave_obj = librosa.feature.inverse.mel_to_audio(mel_spec)
        play_obj = sa.play_buffer(wave_obj, 1, 4, 22050)
        play_obj.wait_done()
        break

    wave_obj, sr = librosa.load("RAVDESS_dataset/wav/Audio_Song_Actors_01-24/Actor_01/03-02-01-01-01-01-01.wav")
    print("sr:", sr)
    mel_spec = utils.apply_transformations(wave_obj, ["mel", "power_to_db"], sr)
    # wave_obj = x[0].numpy()
    mel_spec = librosa.feature.inverse.db_to_power(mel_spec.numpy())
    wave_obj = librosa.feature.inverse.mel_to_audio(mel_spec)
    play_obj = sa.play_buffer(wave_obj, 1, 4, sr)
    play_obj.wait_done()
    '''