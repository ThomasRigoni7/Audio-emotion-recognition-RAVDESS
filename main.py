import copy
import torch
from torch.utils import data
import soundfile as sf
import numpy as np
from scipy import signal
import librosa
from TCN import TCN
from dataset_ravdess import RAVDESS_DATA
import torch.optim as optim
import torch.nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import argparse
import os
from pathlib import Path

# bool parsing function


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


# reading params
parser = argparse.ArgumentParser(
    description='Train a TCN model with the parameters specified, the input files must have been previously generated with genfiles.py.')
parser.add_argument('-m', '--model', type=str,
                    help="model name", default='./models/TCN/best_model.pkl')
parser.add_argument('-b', '--blocks', type=int,
                    help='number of blocks in the TCN', default=5)
parser.add_argument('-r', '--repeats', type=int,
                    help='repeats in the TCN', default=2)
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='learning rate', default=0.001)
parser.add_argument('-e', '--epochs', type=int, help='epochs', default=100)
parser.add_argument('-w', '--workers', type=int,
                    help='workers on the data load', default=0)
parser.add_argument('-p', '--pathdataset', type=str,
                    help='path of the dataset', default='./RAVDESS_dataset/')
parser.add_argument('--batch_size', type=int,
                    help='minibatch size', default=100)
parser.add_argument('-in', '--in_classes', type=int,
                    help='number of input classes', default=40)
parser.add_argument('-out', '--out_classes', type=int,
                    help='number of output classes', default=8)
parser.add_argument('--step_size', type=int,
                    help='number of epochs between each scheduler step', default=10)
parser.add_argument('-g', '--gamma', type=float,
                    help='multiplicative factor of the scheduler step (how much does the learning rate shrink)', default=0.8)
parser.add_argument('--dropout_prob', type=float,
                    help='probability in the dropout layers', default=0.2)
parser.add_argument('--netpath', type=str,
                    help='path of partially trained network', default=None)
parser.add_argument('-t', '--type', type=str,
                    help='type of the input files: mfcc/mfcc128/mel/mel128/', default="mel")
parser.add_argument("--random_load", type=str2bool, nargs='?',
                    help="Load the training data with random init", const=True, default=False)
parser.add_argument("--wandb", type=str2bool, nargs='?',
                    help="Log the run with wandb", const=True, default=True)


arg = parser.parse_args()
path_dataset = arg.pathdataset
numworkers = arg.workers
tcnBlocks = arg.blocks
tcnRepeats = arg.repeats
learning_rate = arg.learning_rate
epochs = arg.epochs
modelname = arg.model
batch_size = arg.batch_size
in_classes = arg.in_classes
out_classes = arg.out_classes
netpath = arg.netpath
inputfiles_type = arg.type
random_load = arg.random_load
use_wandb = arg.wandb
sc_step_size = arg.step_size
sc_gamma = arg.gamma
dropout_prob = arg.dropout_prob

if Path(modelname).is_file():
    ans = input("The specified model file %s already exists, do you really want to overwrite it (y/n)? " % modelname)
    if not str2bool(ans):
        quit()

if use_wandb:
    import wandb
    wandb.init(config=arg)
else:
    wandb = None

directories = {"mfcc": "mfcc/", "mfcc128": "mfcc128/",
               "mel": "mels/", "mel128": "mels128/"}

files_directory = directories[inputfiles_type]


def accuracy(model, generator, device):
    model.eval()
    correct = []
    for data in generator:
        inputs, label = data
        outputs = model(inputs.float().to(device))
        _, pred = torch.max(outputs.detach().cpu(), dim=1)
        correct.append((pred == label).float())
    acc = (np.mean(np.hstack(correct)))
    return 100 * acc


# Set device as cuda if available, otherwise cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

train_data = RAVDESS_DATA(path_dataset + 'train_data.csv', device=device,
                          data_dir=path_dataset + files_directory, random_load=random_load)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': numworkers}
train_set_generator = data.DataLoader(train_data, **params)

test_data = RAVDESS_DATA(path_dataset + 'test_data.csv', device=device,
                         data_dir=path_dataset + files_directory, random_load=False)
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': numworkers}
test_set_generator = data.DataLoader(test_data, **params)

model = TCN(n_blocks=tcnBlocks, n_repeats=tcnRepeats,
            in_chan=in_classes, out_chan=out_classes, dropout_prob=dropout_prob)
if netpath is not None:
    model.load_state_dict(torch.load(netpath))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=sc_step_size, gamma=sc_gamma)

criterion = torch.nn.CrossEntropyLoss()

best_accuracy = 0

train_gen_len = len(train_set_generator)
if wandb is not None:
    wandb.watch(model)

for e in range(epochs):
    for i, d in enumerate(train_set_generator):
        model.train()
        f, l = d
        f = torch.squeeze(f, dim=1)
        y = model(f.float().to(device))
        loss = criterion(y, l.to(device))

        print("Iteration %d in epoch %d--> loss = %f" %
              (i, e + 1, loss.item()), end='\r')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % train_gen_len == train_gen_len - 1:
            scheduler.step()
            model.eval()
            acc_train = accuracy(model, train_set_generator, device)
            acc_test = accuracy(model, test_set_generator, device)
            # accuracy
            iter_acc = 'iteration %d epoch %d--> %f (%f)' % (
                i, e + 1, acc_test, best_accuracy)
            print(iter_acc)
            if wandb is not None:
                wandb.log({"loss": loss, "training accuracy": acc_train,
                           "testing accuracy": acc_test})

            if acc_test > best_accuracy:
                improved_accuracy = 'Current accuracy = %f (%f), updating best model' % (
                    acc_test, best_accuracy)
                print(improved_accuracy)
                best_accuracy = acc_test
                best_epoch = e
                best_model = copy.deepcopy(model)
                torch.save(
                    {"args": arg, "model": model.state_dict()}, modelname)

print("Best accuracy reached at epoch %d with %2.2f%%" %
      (best_epoch + 1, best_accuracy))
