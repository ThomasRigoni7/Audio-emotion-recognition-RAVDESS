import soundfile as sf
import numpy as np
import glob   
import os 
import matplotlib.pyplot as plot
import librosa 
import argparse

parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-l', '--maxlen', type = int, help = "maximum file length", default=128000)

arg = parser.parse_args()
max_len = arg.maxlen

f_files=[]
fcut_files=[]

for i, file in enumerate(glob.glob('RAVDESS_dataset/wav_original/**/**/*.wav')):
    print(i,end='\r')
    f, sr = librosa.load(file)
    f_files.append(len(f))
    
    dir_name = os.path.dirname(file)
    dir_name_mfcc = dir_name.replace('wav_original','mfcc') 
    dir_name_mel = dir_name.replace('wav_original','mels') 
    
    os.makedirs(dir_name_mfcc, exist_ok=True)
    os.makedirs(dir_name_mel, exist_ok=True)
    filename = os.path.basename(file)
    new_filename = filename.replace('.wav', '')
    
    fcut,index= librosa.effects.trim(f,frame_length=2048, hop_length=512)  ### DEFAULT  frame_length=2048, hop_length=512
    fcut_files.append(len(fcut))

    fcut = fcut[:max_len]
    if len(fcut) < max_len: 
        ff = np.pad(fcut, [(0, max_len - fcut.shape[0]),], mode='constant')
    waveform=ff
    mfcc = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=40)
    mel_spectogram = librosa.feature.melspectrogram(y=waveform, sr=sr, n_mels=40)
    scaled_mel = librosa.power_to_db(mel_spectogram)
    
    np.save(dir_name_mfcc+'/'+ filename, mfcc)
    np.save(dir_name_mel+'/'+filename, scaled_mel)
        

### plotting histogram
plot.hist(f_files, bins='auto')  
plot.xlim(0, 100000)
plot.title("Histogram full wav files")
plot.savefig("plots/full_wav.png")     

plot.hist(fcut_files, bins='auto')  
plot.title("Histogram cut wav files")
plot.xlim(0, 100000)
plot.savefig("plots/cut_wav.png")       