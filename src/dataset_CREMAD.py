import csv
import torch
from torch.utils import data
import numpy as np
import pandas as pd
import random
from pathlib import Path
import utils
import os
from tabulate import tabulate


def generate_csv(wav_path, destination_dir):

    emotions2labels = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}

    totcount = [0. for i in range(8)]
    traincount = [0. for i in range(8)]
    testcount = [0. for i in range(8)]
    validcount = [0. for i in range(8)]

    perm = np.random.permutation(91)
    perm += 1001
    train_actors = perm[:65]
    valid_actors = perm[65:73]
    test_actors = perm[73:]

    os.makedirs(destination_dir, exist_ok=True)

    # split the files in train and test set and save the names in csv files
    with open(destination_dir / "train_data.csv", 'w', newline='') as trainfile:
        with open(destination_dir / "test_data.csv", 'w', newline='') as testfile:
            with open(destination_dir / "valid_data.csv", 'w', newline='') as validfile:
                train_writer = csv.writer(trainfile)
                test_writer = csv.writer(testfile)
                valid_writer = csv.writer(validfile)
                train_writer.writerow(["filepath", "label"])
                test_writer.writerow(["filepath", "label"])
                valid_writer.writerow(["filepath", "label"])
                for subdir, dirs, files in os.walk(wav_path):
                    for file in files:
                        filepath = Path(subdir) / Path(file)
                        relative_path = filepath.relative_to(wav_path)
                        label = emotions2labels[file[9:12]]
                        actor = int(file[0:4])
                        totcount[label] += 1
                        '''
                        # cross dataset
                        r = random.randint(0, 2)
                        if r == 0 or r == 1:
                            test_writer.writerow([relative_path, label])
                            testcount[label] += 1
                        elif r == 2:
                            valid_writer.writerow([relative_path, label])
                            validcount[label] += 1
                        train_writer.writerow([relative_path, label])
                        traincount[label] += 1
                        '''
                        
                        # speaker independent
                        if actor in test_actors:
                            test_writer.writerow([relative_path, label])
                            testcount[label - 1] += 1
                        elif actor in valid_actors:
                            valid_writer.writerow([relative_path, label])
                            validcount[label - 1] += 1
                        elif actor in train_actors:
                            train_writer.writerow([relative_path, label])
                            traincount[label - 1] += 1
                        '''
                        # speaker dependent
                        r = random.randint(0, 10)
                        if r == 8 or r == 9:
                            test_writer.writerow([relative_path, label])
                            testcount[label] += 1
                        elif r == 7:
                            valid_writer.writerow([relative_path, label])
                            validcount[label] += 1
                        else:
                            train_writer.writerow([relative_path, label])
                            traincount[label] += 1
                        '''

def get_class_count(wav_path):
    emotions2labels = {"NEU": 0, "HAP": 1, "SAD": 2, "ANG": 3, "FEA": 4, "DIS": 5}

    totcount = [0. for i in range(6)]
    
    for subdir, dirs, files in os.walk(wav_path):
        for file in files:
            label = emotions2labels[file[9:12]]
            totcount[label] += 1
    
    print(tabulate([totcount],
                headers=[i for i in range(1, 7)]))

class CREMAD_DATA(data.Dataset):

    def _load_csv(self, csv_path):
        try:
            table = pd.read_csv(csv_path)
        except OSError as oserr:
            raise ValueError(
                "csv file not found in '{}', did you download the dataset and generate csv files?".format(csv_path)) from oserr
        return table.values.tolist()

    def _load_files(self, filenames):
        print("Loading dataset...")
        files = []
        minlen = 2000000
        for i, (file, label) in enumerate(filenames):
            if int(label) in self.classes_to_use:
                print("{}/{}".format(i, len(filenames)), end="\r")
                filepath = Path(file)
                if self.data_dir is not None:
                    filepath = (Path(self.data_dir) /
                                filepath).with_suffix(self.in_suffix)
                else:
                    filepath = Path(filepath).with_suffix(self.in_suffix)
                x, sr = utils.load_file(filepath, self.sr)
                if sr is not None:
                    x = utils.apply_transformations(
                        x, self.transformations, sr, max_len=5 * sr)
                minlen = min(minlen, x.shape[-1])
                files.append(
                    (x, torch.as_tensor(self.classes_to_use.index(int(label)), dtype=torch.long)))
        print("MinLen: ", minlen)
        if self.chunk_len is None:
            self.chunk_len = minlen
        print("---DONE---")
        return files

    def __init__(self, csv_path, data_dir="./CREMAD_dataset/AudioWAV/", chunk_len=None, random_load=True, in_suffix=".wav",
                    transformations=["cut", "mel", "power_to_db"], sr=None, classes_to_use=[0,1,2,3,4,5]):
        super(CREMAD_DATA, self).__init__()
        self.chunk_len = chunk_len
        self.random_load = random_load
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.in_suffix = in_suffix
        self.transformations = transformations
        self.sr = sr
        self.classes_to_use = classes_to_use
        self.classes = ["neutral", "happy", "sad",
                        "angry", "fearful", "disgust"]
        filenames = self._load_csv(csv_path)
        self.files = self._load_files(filenames)

        # modify classes for cross-dataset
        self.classes = [self.classes[c] for c in classes_to_use]

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
        count = np.zeros(len(self.classes), dtype=int)
        for s in self.files:
            count[s[1]] += 1
        weight = 1. / count
        sample_weight = []
        for s in self.files:
            sample_weight.append(weight[s[1]])
        return count, torch.Tensor(sample_weight)


if __name__ == "__main__":
    # generate_csv(Path("CREMAD_dataset/AudioWAV/"), Path("CREMAD_dataset/csv/random/"))
    
    dataset = CREMAD_DATA("CREMAD_dataset/csv/random/valid_data.csv","CREMAD_dataset/AudioWAV")
    print(dataset)
    print(len(dataset))

    params = {'batch_size': 5,
              'shuffle': False}
    generator = data.DataLoader(dataset, **params)

    for x, y in generator:
        print(y)
        print(x.shape)
        break

    #generate_csv(Path("CREMAD_dataset/AudioWAV/"), Path("CREMAD_dataset/csv/divided/"))
