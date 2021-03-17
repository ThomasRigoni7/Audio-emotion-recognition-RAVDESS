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
                    help="maximum file length in the cutting procces", default=192000)


def inject_noise(data, noise_factor):
    noise = np.random.randn(len(data))
    augmented_data = data + noise_factor * noise
    # Cast back to same data type
    augmented_data = augmented_data.astype(type(data[0]))
    return augmented_data


def shift(data, sampling_rate, shift_max, shift_direction):
    shift = np.random.randint(sampling_rate * shift_max)
    if shift_direction == 'right':
        shift = -shift
    elif shift_direction == 'both':
        direction = np.random.randint(0, 2)
        if direction == 1:
            shift = -shift

    augmented_data = np.roll(data, shift)
    # Set to silence for heading/ tailing
    if shift > 0:
        augmented_data[:shift] = 0
    else:
        augmented_data[shift:] = 0
    return augmented_data


def change_pitch(data, sampling_rate, pitch_max):
    pitch_factor = (2 * np.random.uniform() - 1) * pitch_max
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)


def change_speed(data, speed_min, speed_max):
    speed_factor = np.random.uniform() * (speed_max - speed_min) + speed_min
    return librosa.effects.time_stretch(data, speed_factor)


def apply_transformations(data, sr, shift=False, pitch=False, speed=False, noise=False):
    aug = data
    if shift:
        aug = shift(data, sr, 1, "both")
    if pitch:
        aug = change_pitch(data, sr, 5)
    if speed:
        aug = change_speed(aug, 0.7, 1.3)
    if noise:
        aug = inject_noise(aug, 0.0005)
    return aug


def generate_csv(wav_path, destination_dir):
    totcount = [0. for i in range(8)]
    traincount = [0. for i in range(8)]
    testcount = [0. for i in range(8)]
    validcount = [0. for i in range(8)]

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
                        filepath = pathlib.Path(subdir) / pathlib.Path(file)
                        relative_path = filepath.relative_to(wav_path)
                        relative_path = relative_path.with_suffix(".pt")
                        label = int(file[6:8])
                        actor = int(file[18:20])
                        original = int(file[21:23]) == 0
                        aug1 = int(file[21:23]) == 1
                        totcount[label - 1] += 1
                        if original and (actor == 23 or actor == 24):
                            test_writer.writerow([relative_path, label])
                            testcount[label - 1] += 1
                        elif original and (actor == 21 or actor == 22):
                            valid_writer.writerow([relative_path, label])
                            validcount[label - 1] += 1
                        elif actor < 21:
                            train_writer.writerow([relative_path, label])
                            traincount[label - 1] += 1

    tot_train = sum(traincount)
    tot_test = sum(testcount)
    tot_valid = sum(validcount)
    for i in range(8):
        traincount[i] = "{:.2f}".format(100 * traincount[i]/tot_train) + "%"
        testcount[i] = "{:.2f}".format(100 * testcount[i]/tot_test) + "%"
        validcount[i] = "{:.2f}".format(100 * validcount[i]/tot_valid) + "%"
    print(tabulate([totcount, traincount, validcount, testcount],
                   headers=[i for i in range(1, 9)]))

# compute transformations and save the files
def generate_transformed(wav_path, dest_dir, max_len, put_noise=False):
    f_files = []
    fcut_files = []
    maximum = 0
    minimum = 2000000
    maximum_trim = 0
    minimum_trim = 2000000
    first = True
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

            final_dir = dest_dir / relative_dir

            os.makedirs(final_dir, exist_ok=True)

            # DEFAULT  frame_length=2048, hop_length=512
            fcut, index = librosa.effects.trim(
                f, frame_length=2048, hop_length=512)
            fcut_files.append(len(fcut))

            maximum_trim = max(maximum_trim, len(fcut))
            minimum_trim = min(minimum_trim, len(fcut))

            fcut = fcut[:max_len]
            if len(fcut) < max_len:
                padlen = (max_len - fcut.shape[0]) // 2
                ff = np.pad(
                    fcut, [(padlen, max_len - fcut.shape[0] - padlen), ], mode='constant')
            waveform = ff
            actor = int(str(filename)[18:20])
            if put_noise:
                noise = np.random.normal(0, 0.002, waveform.shape[0])
                waveform = waveform + noise
            mel_spectogram = librosa.feature.melspectrogram(
                y=waveform, sr=sr, n_mels=40)
            scaled_mel = librosa.power_to_db(mel_spectogram)
            torch.save(torch.from_numpy(scaled_mel),
                       final_dir / filename)

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


def generate_transformed_wavs(wav_path, dest_dir):
    i = 1
    for subdir, dirs, files in os.walk(wav_path):
        for filename in files:
            print(i, end='\r')
            filepath = pathlib.Path(subdir) / pathlib.Path(filename)
            f, sr = librosa.load(filepath)
            aug1 = apply_transformations(f, sr, pitch=True)
            aug2 = apply_transformations(f, sr, pitch=True)

            relative_path = filepath.relative_to(wav_path)
            relative_dir = relative_path.parent

            transformed_dir = dest_dir / relative_dir
            os.makedirs(transformed_dir, exist_ok=True)
            name = pathlib.Path(filename).with_suffix("")

            original_path = (transformed_dir / (str(name) + "-00")
                             ).with_suffix(".wav")
            aug1_path = (transformed_dir / (str(name) + "-01")
                         ).with_suffix(".wav")
            aug2_path = (transformed_dir / (str(name) + "-02")
                         ).with_suffix(".wav")

            sf.write(original_path, f, sr)
            sf.write(aug1_path, aug1, sr)
            sf.write(aug2_path, aug2, sr)
            i += 1

def plot_waveform(path, name):
    filepath = pathlib.Path(path)
    filename = filepath.stem
    if filepath.suffix == ".wav":
        waveform, sr = librosa.load(path)
    elif filepath.suffix == ".pt":
        tensor = torch.load(path)
        spec = tensor.numpy()
        pow_spec = librosa.db_to_power(spec)
        waveform = librosa.feature.inverse.mel_to_audio(pow_spec)
    plot.plot(waveform)
    plot.title(filename)
    plot.savefig(pathlib.Path("plots")/ name)
    plot.clf()

def save_wav(path, name):
    filepath = pathlib.Path(path)
    if filepath.suffix == ".wav":
        waveform, sr = librosa.load(path)
    elif filepath.suffix == ".pt":
        tensor = torch.load(path)
        spec = tensor.numpy()
        pow_spec = librosa.db_to_power(spec)
        waveform = librosa.feature.inverse.mel_to_audio(pow_spec)
        sr = 22050
    sf.write(pathlib.Path("test_audio")/ name, waveform, sr)

arg = parser.parse_args()
_max_len = arg.maxlen
_path_dataset = pathlib.Path(arg.pathdataset)

_wav_path = _path_dataset / "wav"

if __name__ == "__main__":
    generate_transformed_wavs(_wav_path, _path_dataset / "wav_pitch")
    generate_csv(_path_dataset / "wav_pitch", _path_dataset / "csv" / "pitch")
    generate_transformed(_path_dataset / "wav_pitch", _path_dataset / "pitch", _max_len)
    
