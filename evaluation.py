import copy
import torch
from torch.utils import data
import soundfile as sf
import numpy as np
from scipy import signal
import librosa
from models import TCN
from dataset_ravdess import RAVDESS_DATA
import torch.optim as optim
import torch.nn
from torch.autograd import Variable
import argparse
import os
from tabulate import tabulate
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# reading params
parser = argparse.ArgumentParser(description='''Evaluate the training and test performance of a model or a directory containing models.
    If the model file has its own configuration it will override the one given as input.''')
parser.add_argument('-m', '--model', type=str,
                    help="model path or directory", required=True)
parser.add_argument('-b', '--blocks', type=int, help='blocks', default=5)
parser.add_argument('-r', '--repeats', type=int, help='repeats', default=2)
parser.add_argument('-w', '--workers', type=int, help='workers', default=0)
parser.add_argument('-p', '--pathdataset', type=str,
                    help='pathdataset', default='./RAVDESS_dataset/')
parser.add_argument('--batch_size', type=int, help='', default=50)
parser.add_argument('-in', '--in_classes', type=int,
                    help='number of output classes', default=40)
parser.add_argument('-out', '--out_classes', type=int,
                    help='number of output classes', default=8)
parser.add_argument('-t', '--type', type=str,
                    help='type of the input files: mfcc/mfcc128/mel/mel128/', default="mel")

settings = parser.parse_args()
model_name = settings.model

directories = {"mfcc": "mfcc/", "mfcc128": "mfcc128/",
               "mel": "mels/", "mel128": "mels128/"}

classes = ['neutral', 'calm', 'happy', 'sad',
           'angry', 'fearful', 'disgust', 'surprised']


def accuracy(model, generator):
    correct = []
    for data in generator:
        inputs, label = data
        outputs = model(inputs.float().to(device))
        _, pred = torch.max(outputs.detach().cpu(), dim=1)
        correct.append((pred == label).float())
    acc = (np.mean(np.hstack(correct)))
    return 100 * acc


def class_accuracy(model, generator, modelsettings):
    class_correct = list(0. for i in range(modelsettings.out_classes))
    class_total = list(0. for i in range(modelsettings.out_classes))
    with torch.no_grad():
        for data in generator:
            inputs, labels = data
            outputs = model(inputs.float().to(device))
            _, pred = torch.max(outputs.detach().cpu(), 1)
            c = (pred == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

    for i in range(modelsettings.out_classes):
        print('%10s : %2.2f %%' %
              (classes[i], 100 * class_correct[i] / class_total[i]))


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)


models = []
if os.path.isdir(model_name):
    for dir, subdir, files in os.walk(model_name):
        for file in files:
            path = os.path.join(model_name, file)
            models.append(path)
else:
    models.append(model_name)


training_acc = []
test_acc = []
for i, modelpath in enumerate(models):
    print("evaluating model {} of {}".format(i + 1, len(models)))
    loaded = torch.load(modelpath)
    if "args" in loaded:
        modelsettings = loaded["args"]
        modeldict = loaded["model"]
    else:
        modelsettings = settings
        modeldict = loaded

    files_directory = directories[modelsettings.type]

    test_data = RAVDESS_DATA(modelsettings.pathdataset + 'test_data.csv', device,
                             data_dir=modelsettings.pathdataset + files_directory, random_load=False)
    params = {'batch_size': modelsettings.batch_size,
              'shuffle': False, 'num_workers': modelsettings.workers}
    test_set_generator = data.DataLoader(test_data, **params)

    training_data = RAVDESS_DATA(modelsettings.pathdataset + 'train_data.csv', device,
                                 data_dir=modelsettings.pathdataset + files_directory, random_load=False)
    training_set_generator = data.DataLoader(training_data, **params)

    model = TCN(n_blocks=modelsettings.blocks, n_repeats=modelsettings.repeats,
                out_chan=modelsettings.out_classes, in_chan=modelsettings.in_classes)
    model.load_state_dict(modeldict)
    model.to(device)
    training_acc.append(accuracy(model, training_set_generator))
    test_acc.append(accuracy(model, test_set_generator))

print(tabulate(list(zip(*[test_acc, training_acc, models][::-1])),
               headers=["Model", "Training", "Test"]))

if len(models) == 1:
    # print detailed statistics about the model
    print("Test performance:")
    class_accuracy(model, test_set_generator, modelsettings)
