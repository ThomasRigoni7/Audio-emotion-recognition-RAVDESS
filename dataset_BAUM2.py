import pandas as pd
import torch
from torch.utils import data
import numpy as np
import librosa
import os
from pathlib import Path

baum2ravdess = {0:0, 1:4, 2:1, 3:6, 4:5, 5:2, 6:3, 7:7}

class BAUM2_DATA(data.Dataset):
    def get_data(self):
        print("Loading dataset...")
        files = []
        table = pd.read_excel(self.excel_path)
        data = table[["Clip Name", "Emotion Code", "Audio useful?"]]
        data_len = len(data.values)
        for i, row in enumerate(data.values):
            print("{}/{}".format(i, data_len), end="\r")
            # load only the values with useful audio
            if row[2] == 1:
                rel_path = row[0].replace("_","/wav/")
                path = Path(self.data_path) / rel_path
                path = path.with_suffix(".wav")
                waveform, sr = librosa.load(path, sr=8000)

                # cut and calculate mel
                max_len = int(12.5 * 8000)   # in seconds * sample rate

                wav_cut, index = librosa.effects.trim(
                waveform, frame_length=2048, hop_length=512)
                if len(wav_cut) < max_len:
                    #padlen = (max_len - fcut.shape[0]) // 2
                    ff = np.pad(
                        wav_cut, [(0, max_len - wav_cut.shape[0]), ], mode='constant')
                    waveform = ff
                
                mel_spectogram = librosa.feature.melspectrogram(
                    y=waveform, sr=sr, n_mels=40)
                scaled_mel = librosa.power_to_db(mel_spectogram)
                files.append((scaled_mel, baum2ravdess[row[1]]))
        print("---DONE---")
        return files

    def __init__(self, data_path, excel_path):
        super(BAUM2_DATA, self).__init__()
        self.data_path = data_path
        self.excel_path = excel_path
        self.files = self.get_data()
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        return self.files[index]



if __name__ == '__main__':
    baum = BAUM2_DATA("BAUM2_dataset/data/", "BAUM2_dataset/Annotations.xlsx")
    print(baum.__getitem__(7))