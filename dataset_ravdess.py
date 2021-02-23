import torch
from torch.utils import data
from scipy import signal
import librosa
import numpy as np
import pandas as pd


class RAVDESS_DATA(data.Dataset):
    def __init__(self, csv_path, device, change_dir = "/mels/"):
        super(RAVDESS_DATA, self).__init__()
        data = pd.read_csv(csv_path)
        filenames = data.values.tolist()
        self.files = []
        for file, label in filenames:
            if change_dir is not None
                file = file.replace("/wav/", change_dir)
            file = file + ".npy"
            with open(file, 'rb') as f:
                numpy_data = np.load(f)
                self.files.append((torch.from_numpy(numpy_data).to(device) ,int(label) - 1))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        X, y = self.files[index]
        return X, y

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
