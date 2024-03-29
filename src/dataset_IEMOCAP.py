import csv
import torch
from pathlib import Path
import random
import shutil
import pandas as pd
from tabulate import tabulate
from torch.utils import data
import utils
import numpy as np

emotions2labels_4 = {"ang": 3, "hap": 0, "sad": 1, "neu":2, "exc":0}
emotions2labels_8 = {"ang": 3, "hap": 0, "sad": 1, "neu":2, "fru":5, "exc":4, "fea":6, "sur":7}

def get_names_labels(file, emotions2labels):
    files = []
    with open(file, "r") as f:
        for line in f.readlines():
            # line contains filename and emotion
            if line[0] == "[":
                tokens = line.split()
                filename = tokens[3]
                emotion = tokens[4]
                if emotion in emotions2labels.keys():
                    files.append((filename, emotion))
    return files

# I decided to copy the audio files in a single folder to have an easier time loading them later
def copy_files(files_path, dest_path):
    files_path = Path(files_path)
    dest_path = Path(dest_path)
    files = []
    # get the audio filenames and labels from the text files
    for folder in [files_path / ("Session" + str(i)) for i in range(1,6)]:
        txt_folder = folder / "dialog" / "EmoEvaluation"
        for file in txt_folder.iterdir():
            if file.is_file():
                files += get_names_labels(file, emotions2labels_8)
        wav_folder = folder / "sentences" / "wav"
        for item in wav_folder.iterdir():
            for file in item.iterdir():
                shutil.copy2(file, dest_path)
        
    with open(dest_path / "info.csv", 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["filepath", "emotion"])
        for filename, emo in files:
            writer.writerow([filename, emo])

def _load_csv(csv_path):
        try:
            table = pd.read_csv(csv_path)
        except OSError as oserr:
                raise ValueError(
                    "csv file not found in {}".format(csv_path)) from oserr
        return table.values.tolist()

def generate_csv(files_path, dest_path, speaker_dependent=False, emotions2labels=emotions2labels_4):
    files_path = Path(files_path)
    dest_path = Path(dest_path)
    len_emotions = 4

    totcount = [0. for i in range(len_emotions)]
    traincount = [0. for i in range(len_emotions)]
    testcount = [0. for i in range(len_emotions)]
    validcount = [0. for i in range(len_emotions)]

    files = _load_csv(files_path / "info.csv")
    # divide the files into train, valid, test
    with open(dest_path / "train_data.csv", 'w', newline='') as trainfile, open(dest_path / "test_data.csv", 'w', newline='') as testfile, open(dest_path / "valid_data.csv", 'w', newline='') as validfile:
        train_writer = csv.writer(trainfile)
        test_writer = csv.writer(testfile)
        valid_writer = csv.writer(validfile)
        train_writer.writerow(["filepath", "label"])
        test_writer.writerow(["filepath", "label"])
        valid_writer.writerow(["filepath", "label"])
        for filename, emotion in files:
            if emotion in emotions2labels.keys() and "impro" in filename:
                label = emotions2labels[emotion]
                totcount[label] += 1
                r = random.randint(0,2)
                if r == 0 or r == 1:
                    test_writer.writerow([filename, label])
                    testcount[label] += 1
                elif r == 2:
                    valid_writer.writerow([filename, label])
                    validcount[label] += 1
                train_writer.writerow([filename, label])
                traincount[label] += 1
                '''
                if speaker_dependent:
                    r = random.randint(0,9)
                    if r == 8 or r == 9:
                        test_writer.writerow([filename, label])
                        testcount[label] += 1
                    elif r == 7:
                        valid_writer.writerow([filename, label])
                        validcount[label] += 1
                    else:
                        train_writer.writerow([filename, label])
                        traincount[label] += 1
                else:
                    # speaker independent
                    # female: 1-5, male: 6-10
                    actor = int(filename[3:5]) + 5 if filename[6] == "M" else int(filename[3:5])
                    if actor == 1 or actor == 6:
                        r = random.randint(0,1)
                        if r == 0:
                            test_writer.writerow([filename, label])
                            testcount[label] += 1
                        elif r == 1:
                            valid_writer.writerow([filename, label])
                            validcount[label] += 1
                    else:
                        train_writer.writerow([filename, label])
                        traincount[label] += 1
                '''
    tot_train = sum(traincount)
    tot_test = sum(testcount)
    tot_valid = sum(validcount)
    for i in range(len_emotions):
        traincount[i] = "{:.2f}".format(100 * traincount[i]/tot_train) + "%"
        testcount[i] = "{:.2f}".format(100 * testcount[i]/tot_test) + "%"
        validcount[i] = "{:.2f}".format(100 * validcount[i]/tot_valid) + "%"
    print(tabulate([totcount, traincount, validcount, testcount],
                   headers=[i for i in range(1, len_emotions + 1)]))


class IEMOCAP_DATA(data.Dataset):    
    def _load_files(self, filenames):
        print("Loading dataset...")
        files = []
        minlen = 2000000
        for i, (file, label) in enumerate(filenames):
            if int(label) in self.classes_to_use:
                print("{}/{}".format(i, len(filenames)), end="\r")
                filepath = Path(file)
                if self.data_dir is not None:
                    filepath = (Path(self.data_dir) / filepath).with_suffix(self.in_suffix)
                else:
                    filepath = Path(filepath).with_suffix(self.in_suffix)
                x, sr = utils.load_file(filepath, self.sr)
                if sr is not None:
                    x = utils.apply_transformations(x, self.transformations, sr, max_len= 5 * sr)
                minlen = min(minlen, x.shape[-1])
                files.append((x, torch.as_tensor(self.classes_to_use.index(int(label)), dtype=torch.long)))
        print("MinLen: ",minlen)
        if self.chunk_len is None:
            self.chunk_len = minlen
        print("---DONE---")
        return files
    
    def __init__(self, csv_path, data_dir="./IEMOCAP_dataset/wav/", chunk_len=None, random_load=True, in_suffix=".wav",
                    transformations=["cut","mel","power_to_db"], sr=None, classes_to_use=[0,1,2,3]):
        super(IEMOCAP_DATA, self).__init__()
        self.chunk_len = chunk_len
        self.random_load = random_load
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.in_suffix = in_suffix
        self.transformations = transformations
        self.sr = sr
        self.classes_to_use = classes_to_use
        self.classes = ["hap", "sad", "neu", "ang"]
        filenames = _load_csv(csv_path)
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
        count =np.zeros(len(self.classes), dtype=int)
        for s in self.files:
            count[s[1]] +=1
        weight = 1. /count
        sample_weight = []
        for s in self.files:
            sample_weight.append(weight[s[1]]) 
        return count, torch.Tensor(sample_weight)



if __name__ == '__main__':
    # copy_files("IEMOCAP_dataset/full/","IEMOCAP_dataset/wav")
    generate_csv("IEMOCAP_dataset/wav", "IEMOCAP_dataset/csv/cross_dataset/", speaker_dependent=True)
    '''
    mydata = IEMOCAP_DATA('./IEMOCAP_dataset/csv/random/test_data.csv', data_dir='./IEMOCAP_dataset/wav/', in_suffix=".wav", transformations=["cut", "mel", "power_to_db"], sr = 22050)
    print(mydata)
    print(mydata.__len__())

    params = {'batch_size': 5,
              'shuffle': False}
    test_set_generator = data.DataLoader(mydata, **params)

    for x, y in test_set_generator:
        print(y)
        print(x.shape)
        break'''