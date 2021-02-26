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

parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-l', '--maxlen', type = int, help = "maximum file length", default=128000)

arg = parser.parse_args()
max_len = arg.maxlen

f_files=[]
fcut_files=[]
maximum = 0
minimum = 200000


totcount = [0. for i in range(8)]
traincount = [0. for i in range(8)]
testcount = [0. for i in range(8)]
with open("RAVDESS_dataset/train_data.csv", 'w', newline='') as trainfile:
    with open("RAVDESS_dataset/test_data.csv", 'w', newline='') as testfile:
        train_writer = csv.writer(trainfile)
        test_writer = csv.writer(testfile)
        train_writer.writerow(["filepath", "label"])
        test_writer.writerow(["filepath", "label"])
        path = pathlib.Path("RAVDESS_dataset/wav/")
        for subdir, dirs, files in os.walk(path):
            for file in files:
                filepath = pathlib.Path(subdir) / pathlib.Path(file)
                thirdparent = filepath.parent.parent.parent
                print(filepath, thirdparent)
                relative_path = filepath.relative_to(thirdparent)
                relative_path = relative_path.with_suffix(".pt")
                label = int(file[6:8])
                totcount[label - 1] += 1
                if random.randint(1,5) == 1:
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
print(tabulate([totcount, traincount, testcount], headers=[i for i in range(1,9)]))


#compute transformations and save the files
for i, file in enumerate(glob.glob('RAVDESS_dataset/wav/**/**/*.wav')):
    print(i,end='\r')
    f, sr = librosa.load(file)
    f_files.append(len(f))
    maximum = max(maximum, len(f))
    minimum = min(minimum, len(f))
    
    dir_name = os.path.dirname(file)
    
    dir_name_mfcc = dir_name.replace('wav','mfcc') 
    dir_name_mfcc128 = dir_name.replace('wav','mfcc128') 
    dir_name_mel = dir_name.replace('wav','mels')
    dir_name_mel128 = dir_name.replace('wav','mels128')
    
    os.makedirs(dir_name_mfcc, exist_ok=True)
    os.makedirs(dir_name_mfcc128, exist_ok=True)
    os.makedirs(dir_name_mel, exist_ok=True)
    os.makedirs(dir_name_mel128, exist_ok=True)
    filename = os.path.basename(file)
    
    
    fcut,index= librosa.effects.trim(f, frame_length=2048, hop_length=512)  ### DEFAULT  frame_length=2048, hop_length=512
    fcut_files.append(len(fcut))

    fcut = fcut[:max_len]
    if len(fcut) < max_len: 
        ff = np.pad(fcut, [(0, max_len - fcut.shape[0]),], mode='constant')
    waveform=ff
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
    mfcc128 = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=128)
    mel_spectogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=40)
    mel_spectogram128 = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=40)
    scaled_mel = librosa.power_to_db(mel_spectogram)
    scaled_mel128 = librosa.power_to_db(mel_spectogram128)

    filename = os.path.splitext(filename)[0] + ".pt"
    
    torch.save(torch.from_numpy(mfcc), dir_name_mfcc+'/'+ filename)
    torch.save(torch.from_numpy(scaled_mel), dir_name_mel+'/'+filename)
    torch.save(torch.from_numpy(mfcc128), dir_name_mfcc128+'/'+ filename)
    torch.save(torch.from_numpy(scaled_mel128), dir_name_mel128+'/'+filename)

print("max:", maximum)
print("min:", minimum)

### plotting histogram
plot.hist(f_files, bins='auto')  
plot.xlim(0, 150000)
plot.title("Histogram full wav files")
plot.savefig("plots/full_wav.png")     

plot.hist(fcut_files, bins='auto')  
plot.title("Histogram cut wav files")
plot.xlim(0, 150000)
plot.savefig("plots/cut_wav.png")      
