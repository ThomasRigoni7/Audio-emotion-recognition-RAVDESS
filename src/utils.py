import numpy as np
import librosa
import torch
import math
from pathlib import Path
import specaugment.spec_augment_pytorch as specaugment
from pysndfx import AudioEffectsChain


def load_file(filepath : Path, sr, start_time=0.0, end_time=None):
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
            if end_time is None:
                duration = None
            else:
                duration = end_time - start_time
            x, sr = librosa.load(f, sr=sr, offset=start_time, duration=duration)
        else:
            raise ValueError(
            "The suffix '{}' is not valid.".format(filepath.suffix))
        return (x, sr)

def apply_transformations(data, transformations, sr, max_len=None) -> torch.Tensor:
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
        if "reverb" in transformations:
            fx = (AudioEffectsChain().reverb())
            data = fx(data)
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
        if "spec_augment" in transformations:
            data = torch.from_numpy(data)
            data = specaugment.spec_augment(data)
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

# training/testing metrics
def get_predictions(model, generator, device):
    model.eval()
    x, y = generator.dataset.__getitem__(0)
    num_elements = len(generator.dataset)
    num_classes = len(generator.dataset.classes)
    num_batches = len(generator)
    batch_size = generator.batch_size
    if y.dim() == 0:
        predictions = torch.zeros(num_elements, num_classes, dtype=torch.float)
        ground_truths = torch.zeros(num_elements, dtype=torch.long)
    elif y.dim() == 1:
        predictions = torch.zeros(num_elements, num_classes, dtype=torch.float)
        ground_truths = torch.zeros(num_elements, num_classes, dtype=torch.float)
    else:
        raise RuntimeError("The dimension of the labels is not 0 or 1: dim={}".format(y.dim()))
    for i, (inputs, labels) in enumerate(generator):
        start = i * batch_size
        end = start + batch_size
        if i == num_batches - 1:
            end = num_elements
        outputs = model(inputs.float().to(device))
        predictions[start:end] = outputs.detach().cpu()
        ground_truths[start:end] = labels.detach().cpu()
    return predictions, ground_truths


if __name__ == '__main__':
    data, sr = load_file(Path("RAVDESS_dataset/wav/Audio_Song_Actors_01-24/Actor_01/03-02-01-01-01-01-01.wav"), 22050)
    data = apply_transformations(data, ["cut","noise", "mfcc", "power_to_db"], sr, max_len=7000)