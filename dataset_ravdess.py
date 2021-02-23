import torch
from torch.utils import data
from scipy import signal
import librosa
import numpy as np
import pandas as pd


class RAVDESS_DATA(data.Dataset):
    def __init__(self, csv_path, max_len=192000):
        super(RAVDESS_DATA, self).__init__()
        self.max_len=max_len
        data = pd.read_csv(csv_path)
        self.files = data.values.tolist()

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        filepath, label = self.files[index]
        try:
            #Load librosa array, pad it, obtain mel and return couple X, y
            waveform, sample_rate = librosa.load(filepath, res_type='kaiser_fast')
            waveform = waveform[:self.max_len]
            if len(waveform) < self.max_len: 
                ff = np.pad(waveform, [(0, self.max_len - waveform.shape[0]),], mode='constant')
                waveform=ff
            mfcc = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=40)
            label = int(label) - 1
            t_mfcc = torch.from_numpy(mfcc)
            return t_mfcc, label
        # If the file is not valid, skip it
        except ValueError:
            raise RuntimeError("File {} is not valid.".format(filepath))


if __name__ == "__main__":
    mydata = RAVDESS_DATA('RAVDESS_dataset/train_data.csv', 192000)
    print(mydata)
    print(mydata.__len__())

    params = {'batch_size': 5,
              'shuffle': False}
    test_set_generator = data.DataLoader(mydata,**params)

    for x, y in test_set_generator:
        print(y)
        print(x.shape)
        break
