import soundfile as sf
import numpy as np
import glob
import os
import matplotlib.pyplot as plot
import librosa
import argparse
import torch
import random
import csv
from tabulate import tabulate
import pathlib

parser = argparse.ArgumentParser(
    description='Generate the preprocessed files used for training, the audio files must be in a folder called "wav" inside the main dataset directory')
parser.add_argument('-p', '--pathdataset', type=str,
                    help='path of the main dataset directory', default='./RAVDESS_dataset/')
parser.add_argument('-l', '--maxlen', type=int,
                    help="maximum file length in the cutting procces", default=128000)

arg = parser.parse_args()
max_len = arg.maxlen
path_dataset = pathlib.Path(arg.pathdataset)

f_files = []
fcut_files = []
maximum = 0
minimum = 2000000
maximum_trim = 0
minimum_trim = 2000000

wav_path = path_dataset / "wav"

totcount = [0. for i in range(8)]
traincount = [0. for i in range(8)]
testcount = [0. for i in range(8)]

# split the files in train and test set and save the names in csv files
with open(path_dataset / "train_data.csv", 'w', newline='') as trainfile:
    with open(path_dataset / "test_data.csv", 'w', newline='') as testfile:
        train_writer = csv.writer(trainfile)
        test_writer = csv.writer(testfile)
        train_writer.writerow(["filepath", "label"])
        test_writer.writerow(["filepath", "label"])
        for subdir, dirs, files in os.walk(wav_path):
            for file in files:
                filepath = pathlib.Path(subdir) / pathlib.Path(file)
                relative_path = filepath.relative_to(wav_path)
                relative_path = relative_path.with_suffix(".pt")
                label = int(file[6:8])
                totcount[label - 1] += 1
                if random.randint(1, 5) == 1:
                    test_writer.writerow([relative_path, label])
                    testcount[label - 1] += 1
                else:
                    train_writer.writerow([relative_path, label])
                    traincount[label - 1] += 1

tot_train = sum(traincount)
tot_test = sum(testcount)
for i in range(8):
    traincount[i] = "{:.2f}".format(100 * traincount[i]/tot_train) + "%"
    testcount[i] = "{:.2f}".format(100 * testcount[i]/tot_test) + "%"
print(tabulate([totcount, traincount, testcount],
               headers=[i for i in range(1, 9)]))

# compute transformations and save the files
i = 1
for subdir, dirs, files in os.walk(wav_path):
    for filename in files:
        print(i, end='\r')
        filepath = pathlib.Path(subdir) / pathlib.Path(filename)
        f, sr = librosa.load(filepath)
        f_files.append(len(f))
        maximum = max(maximum, len(f))
        minimum = min(minimum, len(f))

        relative_path = filepath.relative_to(wav_path)
        relative_dir = relative_path.parent
        filename = pathlib.Path(filename).with_suffix(".pt")

        dir_name_mfcc = path_dataset / "mfcc" / relative_dir
        dir_name_mfcc128 = path_dataset / "mfcc128" / relative_dir
        dir_name_mel = path_dataset / "mels" / relative_dir
        dir_name_mel128 = path_dataset / "mels128" / relative_dir

        os.makedirs(dir_name_mfcc, exist_ok=True)
        os.makedirs(dir_name_mfcc128, exist_ok=True)
        os.makedirs(dir_name_mel, exist_ok=True)
        os.makedirs(dir_name_mel128, exist_ok=True)

        # DEFAULT  frame_length=2048, hop_length=512
        fcut, index = librosa.effects.trim(
            f, frame_length=2048, hop_length=512)
        fcut_files.append(len(fcut))

        maximum_trim = max(maximum_trim, len(fcut))
        minimum_trim = min(minimum_trim, len(fcut))

        fcut = fcut[:max_len]
        if len(fcut) < max_len:
            ff = np.pad(
                fcut, [(0, max_len - fcut.shape[0]), ], mode='constant')
        waveform = ff
        mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
        mfcc128 = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=128)
        mel_spectogram = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=40)
        mel_spectogram128 = librosa.feature.melspectrogram(
            y=waveform, sr=sr, n_mels=128)
        scaled_mel = librosa.power_to_db(mel_spectogram)
        scaled_mel128 = librosa.power_to_db(mel_spectogram128)

        torch.save(torch.from_numpy(mfcc), dir_name_mfcc / filename)
        torch.save(torch.from_numpy(scaled_mel), dir_name_mel / filename)
        torch.save(torch.from_numpy(mfcc128), dir_name_mfcc128 / filename)
        torch.save(torch.from_numpy(scaled_mel128), dir_name_mel128 / filename)
        i += 1

print("max before trim:", maximum)
print("min before trim:", minimum)
print("max after trim:", maximum_trim)
print("min after trim:", minimum_trim)

# plotting histogram
plot.hist(f_files, bins='auto')
plot.xlim(0, 150000)
plot.title("Histogram full wav files")
plot.savefig("./plots/full_wav.png")

plot.hist(fcut_files, bins='auto')
plot.title("Histogram cut wav files")
plot.xlim(0, 150000)
plot.savefig("./plots/cut_wav.png")
