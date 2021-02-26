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
import matplotlib.pyplot as plt
import argparse
import os
import wandb

#reading params
parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-m', '--model', type = str, help = "model name", default='./models/TCN/best_model.pkl')
parser.add_argument('-b', '--blocks', type = int, help='number of blocks in the TCN',default = 5)
parser.add_argument('-r', '--repeats', type = int, help='repeats in the TCN', default = 2)
parser.add_argument('-lr', '--learning_rate', type = float, help = 'learning rate', default = 0.001)
parser.add_argument('-e', '--epochs', type = int, help = 'epochs', default = 100)
parser.add_argument('-w', '--workers', type = int, help='workers on the data load',default = 0)
parser.add_argument('-p', '--pathdataset', type = str, help='path of the dataset', default = './RAVDESS_dataset/')
parser.add_argument('--batch_size', type = int, help='minibatch size', default = 100)
parser.add_argument('-in','--in_classes', type = int, help='number of output classes', default = 40)
parser.add_argument('-out','--out_classes', type = int, help='number of output classes', default = 8)
parser.add_argument('--netpath', type = str, help='path of partially trained network', default = None)
parser.add_argument('-t', '--type', type = str, help='type of the input files: mfcc/mfcc128/mel/mel128/', default="mel")

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
out_classes=arg.out_classes
netpath=arg.netpath
inputfiles_type=arg.type

wandb.init(config=arg)

directories = {"mfcc":"mfcc/", "mfcc128":"mfcc128/","mel":"mels/", "mel128":"mels128/"}

files_directory=directories[inputfiles_type]

def accuracy(model, generator, device):
    model.eval()
    correct=[]
    for data in generator:
        inputs, label = data
        outputs = model(inputs.float().to(device))
        _, pred = torch.max(outputs.detach().cpu(), dim=1)
        correct.append((pred == label).float())
    acc= (np.mean(np.hstack(correct)))
    return 100 * acc

##Set device as cuda if available, otherwise cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

train_data = RAVDESS_DATA(path_dataset + 'train_data.csv', device=device, data_dir=path_dataset + files_directory, random_load=False)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': numworkers}
train_set_generator=data.DataLoader(train_data,**params)

test_data = RAVDESS_DATA(path_dataset + 'test_data.csv', device=device, data_dir=path_dataset + files_directory, random_load=False)
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': numworkers}
test_set_generator=data.DataLoader(test_data,**params)

model = TCN(n_blocks=tcnBlocks, n_repeats=tcnRepeats, in_chan=in_classes, out_chan=out_classes)
if netpath is not None:
    model.load_state_dict(torch.load(netpath))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

criterion = torch.nn.CrossEntropyLoss()

best_accuracy = 0

train_gen_len = len(train_set_generator)

wandb.watch(model)

for e in range(epochs):
    for i, d in enumerate(train_set_generator):
        model.train()
        f, l = d
        y = model(f.float().to(device))
        loss = criterion(y,l.to(device))

        print("Iteration %d in epoch %d--> loss = %f"%(i,e,loss.item()),end='\r')
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if i % train_gen_len == train_gen_len - 1:
            acc_train = accuracy(model, train_set_generator, device)
            acc_test = accuracy(model, test_set_generator, device)
            iter_acc = 'iteration %d epoch %d--> %f (%f)'%(i, e + 1, acc_test, best_accuracy)  #accuracy
            print(iter_acc)   
            wandb.log({"loss": loss, "training accuracy": acc_train, "testing accuracy": acc_test})
       
            if acc_test > best_accuracy:
                improved_accuracy = 'Current accuracy = %f (%f), updating best model'%(acc_test,best_accuracy)
                print(improved_accuracy)
                best_accuracy = acc_test
                best_epoch = e
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), modelname)

