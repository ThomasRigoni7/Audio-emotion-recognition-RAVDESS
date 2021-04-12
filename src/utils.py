import numpy as np
import librosa
import torch
import math
from pathlib import Path

def load_file(filepath : Path, sr):
    try:
        f = open(filepath, 'rb')
    except OSError as oserr:
        raise ValueError(
            "Data file not found in '{}', check the path.".format(filepath)) from oserr
    with f:
        if filepath.suffix == ".pt":
            x = torch.load(f)
            sr = None
        elif filepath.suffix == ".wav":
            x, sr = librosa.load(f, sr=sr)
        else:
            raise ValueError(
            "The suffix '{}' is not valid.".format(self.in_suffix))
        return (x, sr)

def apply_transformations(data, transformations, sr, max_len=None):
    '''
    This function applies the transformations listed as input to the data, but only if the data is a np array (loaded with librosa from audio), otherwise (torch Tensor) it does nothing.
    
    transformations is a List containing the transformations, possible values: ["cut", "noise", "mel", "mfcc", "power_to_db"]. 

    mel and mfcc cannot be specified both. 
    To cut the data it is mandatory to supply a max_len
    '''
    if isinstance(data, np.ndarray):
        if "cut" in transformations:
            if max_len is None:
                raise ValueError(
                "Cannot cut the files if max_len is not specified.")
            # fcut = data[:max_len]
            fcut = data
            if len(fcut) < max_len:
                padlen = (max_len - fcut.shape[0]) // 2
                ff = np.pad(
                    fcut, [(padlen, max_len - fcut.shape[0] - padlen), ], mode='constant')
                data = ff
        if "noise" in transformations:
            noise = np.random.normal(0, 0.002, data.shape[0])
            data = data + noise
        if "mel" in transformations and "mfcc" in transformations:
            raise ValueError(
            "Cannot transform both mel spectrogram and mfcc.")
        if "mel" in transformations:
            data = librosa.feature.melspectrogram(
                y=data, sr=sr, n_mels=40)
        if "mfcc" in transformations:
            data = librosa.feature.mfcc(
                y=data, sr=sr, n_mfcc=40)
        if "power_to_db" in transformations:
            data = librosa.power_to_db(data)
        data = torch.from_numpy(data)
    return data

def divide_and_discard(x, sr, minlen, maxlen):
    # if we have loaded the file with librosa
    if sr is not None:
        if len(x) < minlen:
            # discard the sample
            return []
        if len(x) > maxlen:
            # split the sample into (almost) equal parts
            num_splits = math.ceil(len(x) / maxlen)
            return np.array_split(x, num_splits)
        # return the sample if its len is in between min and max
        return [x]
    return [x]

if __name__ == '__main__':
    data, sr = load_file(Path("RAVDESS_dataset/wav/Audio_Song_Actors_01-24/Actor_01/03-02-01-01-01-01-01.wav"), 22050)
    data = apply_transformations(data, ["cut","noise", "mfcc", "power_to_db"], sr, max_len=7000)