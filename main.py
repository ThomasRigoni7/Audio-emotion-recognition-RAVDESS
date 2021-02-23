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

#reading params
parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-m', '--model', type = str, help = "model name", default='best_model.pkl')
parser.add_argument('-b', '--blocks', type = int, help='blocks',default = 5)
parser.add_argument('-r', '--repeats', type = int, help='repeats', default = 2)
parser.add_argument('-lr', '--learning_rate', type = float, help = 'learning rate', default = 0.001)
parser.add_argument('-e', '--epochs', type = int, help = 'epochs', default = 100)
parser.add_argument('-w', '--workers', type = int, help='workers',default = 0)
parser.add_argument('-p', '--pathdataset', type = str, help='pathdataset', default = './RAVDESS_dataset')
parser.add_argument('--batch_size', type = int, help='batch_size',default = 100)
parser.add_argument('--n_classes', type = int, help='number of output classes',default = 8)
parser.add_argument('--netpath', type = str, help='path of partially trained network', default = None)

arg = parser.parse_args()
path_dataset = arg.pathdataset
numworkers = arg.workers
tcnBlocks = arg.blocks
tcnRepeats = arg.repeats
learning_rate = arg.learning_rate
epochs = arg.epochs
modelname = arg.model
batch_size = arg.batch_size
n_classes=arg.n_classes
netpath=arg.netpath

##Set device as cuda if available, otherwise cpu
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

train_data = RAVDESS_DATA( path_dataset + '/train_data.csv', max_len = 128000)
params = {'batch_size': batch_size,
          'shuffle': True,
          'num_workers': numworkers}
train_set_generator=data.DataLoader(train_data,**params)

test_data = RAVDESS_DATA(path_dataset + '/test_data.csv',max_len = 128000)
params = {'batch_size': batch_size,
          'shuffle': False,
          'num_workers': numworkers}
test_set_generator=data.DataLoader(test_data,**params)

model = TCN(n_blocks=tcnBlocks, n_repeats=tcnRepeats, out_chan=n_classes)
if netpath is not None:
    model.load_state_dict(torch.load(netpath))
model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr = learning_rate)

criterion = torch.nn.CrossEntropyLoss()

best_accuracy = 0

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
        if i%20 == 0:            
            model.eval()
            correct = []
            for j,eval_data in enumerate(test_set_generator):
                feat,label = eval_data

                y_eval = model(feat.float().to(device))
                _, pred = torch.max(y_eval.detach().cpu(),dim=1)

                correct.append((pred == label).float())
                if j > 10:
                    break
            acc = (np.mean(np.concatenate(correct)))
            iter_acc = 'iteration %d epoch %d--> %f (%f)'%(i, e, acc, best_accuracy)  #accuracy
            print(iter_acc)   
            
       
            if acc > best_accuracy:
                improved_accuracy = 'Current accuracy = %f (%f), updating best model'%(acc,best_accuracy)
                print(improved_accuracy)
                best_accuracy = acc
                best_epoch = e
                best_model = copy.deepcopy(model)
                torch.save(model.state_dict(), modelname)