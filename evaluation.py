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
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#reading params
parser = argparse.ArgumentParser(description='Desciption')
parser.add_argument('-m', '--model', type = str, help = "model name", required=True)
parser.add_argument('-b', '--blocks', type = int, help = 'blocks', default=5)
parser.add_argument('-r', '--repeats', type = int, help='repeats', default=2)
parser.add_argument('-w', '--workers', type = int, help='workers',default=0)
parser.add_argument('-p', '--pathdataset', type = str, help='pathdataset', default = './RAVDESS_dataset/')
parser.add_argument('--batch_size', type = int, help='',default = 50)
parser.add_argument('--n_classes', type = int, help='number of output classes',default = 8)

#storing params 
arg = parser.parse_args()
model_name = arg.model
path_dataset= arg.pathdataset
batch_size=arg.batch_size
n_classes=arg.n_classes

test_data = RAVDESS_DATA(path_dataset + 'test_data.csv',max_len = 128000)
params = {'batch_size': batch_size,'shuffle': False,'num_workers': arg.workers}
test_set_generator=data.DataLoader(test_data,**params)

training_data = RAVDESS_DATA(path_dataset + 'train_data.csv',max_len = 128000)
training_set_generator=data.DataLoader(training_data,**params)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using device %s' % device)

model = TCN(n_blocks=arg.blocks,n_repeats=arg.repeats,out_chan=n_classes)

#loading the model 
model.load_state_dict(torch.load(model_name))
model.eval()
model.to(device)

correct_training=[]
for i, d in enumerate(training_set_generator):
    print('Iter %d (%d/%d)'%(i,i*batch_size,len(training_data)),end='\r')
    feat,label=d

    a_eval = model(feat.float().to(device))
    _, pred_test = torch.max(a_eval.detach().cpu(),dim=1)
    correct_training.append((pred_test == label).float())


acc_training= (np.mean(np.hstack(correct_training)))
print("The accuracy on the training set is %2.2f %%" %(100 *acc_training))

correct_test = []
for i, d in enumerate(test_set_generator):

    feat,label=d
    print('Iter %d (%d/%d)'%(i,i*batch_size,len(test_data)),end='\r')
    z_eval = model(feat.float().to(device))
    _, pred_test = torch.max(z_eval.detach().cpu(),dim=1)
    correct_test.append((pred_test == label).float())


acc_test= (np.mean(np.hstack(correct_test)))  
print("The accuracy on test set is %2.2f %%" %(100 * acc_test))
