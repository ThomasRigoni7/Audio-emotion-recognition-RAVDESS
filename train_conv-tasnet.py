import copy
import torch
from torch.utils import data
import soundfile as sf
import numpy as np
from scipy import signal
import librosa
from conv_tasnet import ConvTasNet
from dataset_ravdess import RAVDESS_DATA
import torch.optim as optim
import torch.nn
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
    description='Train a Conv-TasNet model with the parameters specified, the input files must have been previously generated with genfiles.py.')
# Network architecture
parser.add_argument('--N', default=256, type=int,
                    help='Number of filters in autoencoder')
parser.add_argument('--L', default=20, type=int,
                    help='Length of the filters in samples (40=5ms at 8kHZ)')
parser.add_argument('--B', default=256, type=int,
                    help='Number of channels in bottleneck 1 × 1-conv block')
parser.add_argument('--H', default=512, type=int,
                    help='Number of channels in convolutional blocks')
parser.add_argument('--P', default=3, type=int,
                    help='Kernel size in convolutional blocks')
parser.add_argument('--X', default=8, type=int,
                    help='Number of convolutional blocks in each repeat')
parser.add_argument('--R', default=4, type=int,
                    help='Number of repeats')
parser.add_argument('--C', default=8, type=int,
                    help='Number of out classes')
parser.add_argument('--norm_type', default='gLN', type=str,
                    choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
parser.add_argument('--causal', type=int, default=0,
                    help='Causal (1) or noncausal(0) training')
parser.add_argument('--mask_nonlinear', default='relu', type=str,
                    choices=['relu', 'softmax'], help='non-linear to generate mask')
#configuration
parser.add_argument('-m', '--model', type=str,
                    help="model name", default=None)
parser.add_argument('--netpath', type=str,
                    help='path of partially trained network', default=None)
parser.add_argument('-t', '--type', type=str,
                    help='type of the input files: mfcc/mfcc128/mel/mel128/mel_noise/augmented', default="mel")
parser.add_argument("--random_load", type=str2bool, nargs='?',
                    help="Load the training data with random init", const=True, default=False)
parser.add_argument("--wandb", type=str2bool, nargs='?',
                    help="Log the run with wandb", const=True, default=True)
parser.add_argument('-csv', '--csv_location', type=str,
                    help='directory where to find the csv files test_data.csv, train_data.csv, valid_data.csv', default="./RAVDESS_dataset/csv/divided_with_validation/")
#learning parameters
parser.add_argument('-lr', '--learning_rate', type=float,
                    help='learning rate', default=0.001)
parser.add_argument('-e', '--epochs', type=int, help='epochs', default=40)
parser.add_argument('-w', '--workers', type=int,
                    help='workers on the data load', default=0)
parser.add_argument('-p', '--pathdataset', type=str,
                    help='path of the dataset', default='./RAVDESS_dataset/')
parser.add_argument('--batch_size', type=int,
                    help='minibatch size', default=8)
parser.add_argument('--step_size', type=int,
                    help='number of epochs between each scheduler step', default=10)
parser.add_argument('-g', '--gamma', type=float,
                    help='multiplicative factor of the scheduler step (how much does the learning rate shrink)', default=0.9)



arg = parser.parse_args()
path_dataset = arg.pathdataset
numworkers = arg.workers
learning_rate = arg.learning_rate
epochs = arg.epochs
modelname = arg.model
batch_size = arg.batch_size
netpath = arg.netpath
inputfiles_type = arg.type
random_load = arg.random_load
use_wandb = arg.wandb
sc_step_size = arg.step_size
sc_gamma = arg.gamma
csv_location = arg.csv_location


if modelname is not None and Path(modelname).is_file():
    ans = input(
        "The specified model file %s already exists, do you really want to overwrite it (y/n)? " % modelname)
    if not str2bool(ans):
        quit()

if use_wandb:
    import wandb
    wandb.init(config=arg, project="conv-tasnet", save_code=True)
    wandb.save("*.py")
    if modelname is None:
        modelname = Path("./models/ConvTasNet/WandB/") / wandb.run.name
else:
    wandb = None
    if modelname is None:
        raise RuntimeError("If wandb is not active you MUST specify a model name!")

files_directory = Path(inputfiles_type) / ""


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

train_data = RAVDESS_DATA(csv_location + 'train_data.csv', 
                          data_dir=Path(path_dataset) / files_directory, random_load=random_load, chunk_len=300, in_suffix = ".wav")
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': numworkers}
train_set_generator = data.DataLoader(train_data, **params)

valid_data = RAVDESS_DATA(csv_location + 'valid_data.csv', 
                          data_dir=Path(path_dataset) / files_directory, random_load=False, in_suffix = ".wav")
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': numworkers}
valid_set_generator = data.DataLoader(valid_data, **params)

test_data = RAVDESS_DATA(csv_location + 'test_data.csv', 
                         data_dir=Path(path_dataset) / files_directory, random_load=False, in_suffix = ".wav")
test_set_generator = data.DataLoader(test_data, **params)

model = ConvTasNet(arg.N, arg.L, arg.B, arg.H, arg.P, arg.X, arg.R,
                       arg.C, norm_type=arg.norm_type, causal=arg.causal,
                       mask_nonlinear=arg.mask_nonlinear)

if netpath is not None:
    model.load_state_dict(torch.load(netpath))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(
    optimizer, step_size=sc_step_size, gamma=sc_gamma)

criterion = torch.nn.CrossEntropyLoss()

best_accuracy = 0

train_gen_len = len(train_set_generator)
if wandb is not None:
    wandb.watch(model)

for e in range(epochs):
    sum_loss = 0
    for i, d in enumerate(train_set_generator):
        model.train()
        f, l = d
        f = torch.squeeze(f, dim=1)
        # print("x shape: ", f.shape)
        y = model(f.float().to(device))
        # print("y shape: ", y.shape)
        loss = criterion(y, l.to(device))
        sum_loss += loss

        print("Iteration %d in epoch %d--> loss = %f" %
              (i, e + 1, loss.item()), end='\r')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % train_gen_len == train_gen_len - 1:
            scheduler.step()
            model.eval()
            #acc_train = accuracy(model, train_set_generator, device)
            acc_valid = accuracy(model, valid_set_generator, device)
            # accuracy
            iter_acc = 'iteration %d epoch %d--> %f (%f)' % (
                i, e + 1, acc_valid, best_accuracy)
            print(iter_acc)
            if wandb is not None:
                wandb.log({"loss": sum_loss/train_gen_len, #"training accuracy": acc_train,
                           "validation accuracy": acc_valid})
            sum_loss = 0

            if acc_valid > best_accuracy:
                improved_accuracy = 'Current accuracy = %f (%f), updating best model' % (
                    acc_valid, best_accuracy)
                print(improved_accuracy)
                best_accuracy = acc_valid
                best_epoch = e
                best_model = copy.deepcopy(model)
                torch.save(
                    {"args": arg, "model": model.state_dict()}, modelname)

model.eval()
test_acc_best = accuracy(best_model, test_set_generator, device)
test_acc_last = accuracy(model, test_set_generator, device)
if wandb is not None:
    wandb.log({"best accuracy": best_accuracy})

print("Best accuracy on validation set reached at epoch %d with %2.2f%%" %
      (best_epoch + 1, best_accuracy))
if test_acc_best > test_acc_last:
    print("Best accuracy on test set is on the BEST model with %2.2f%%, (vs %2.2f%%)" % (
        test_acc_best, test_acc_last))
else:
    print("Best accuracy on test set is on the LAST model with %2.2f%%, (vs %2.2f%%)" % (
        test_acc_last, test_acc_best))
    modelpath = Path(modelname)
    torch.save({"args": arg, "model": model.state_dict()},
               modelpath.with_name(modelpath.stem + "_LAST.pkl"))
