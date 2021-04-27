import torch
from torch.utils import data
import numpy as np
import librosa
import data_loader
import torch.optim as optim
import torch.nn
import argparse
from pathlib import Path
from train import train
from test import test
import yaml


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

def get_arguments():
    # reading params
    parser = argparse.ArgumentParser(
        description='Train a model with the parameters specified.')
    # config files
    parser.add_argument('--dataset_config', default=None, type=str,
                        help='path to the dataset config file', required=True)
    parser.add_argument('--model_config', default=None, type=str,
                        help='path to the model config file', required=True)
    parser.add_argument('--training_config', default=None, type=str,
                        help='path to the training config file', required=True)
    # wandb
    parser.add_argument("--wandb", type=str2bool, nargs='?',
                        help="Log the run with wandb", const=True, default=True)
    parser.add_argument("--tags", type=str, 
                        help="Tags to add in the wandb run", default=None)

    ###
    # All the arguments below  are set to None as default because usually the values on the config files are used. The command-line ones override the config files
    ###

    # TCN Network architecture
    parser.add_argument('-b', '--blocks', type=int,
                        help='number of blocks in the TCN', default=None)
    parser.add_argument('-r', '--repeats', type=int,
                        help='repeats in the TCN', default=None)
    parser.add_argument('--dropout_prob', default=None, type=float,
                        help='dropout probability')
    parser.add_argument('-in', '--in_classes', type=int,
                        help='number of input classes (for TCN)', default=None)
    parser.add_argument('-out', '--out_classes', type=int,
                        help='number of output classes (for TCN, use C for conv-tasnet)', default=None)

    # Conv-TasNet Network architecture
    parser.add_argument('--N', default=None, type=int,
                        help='Number of filters in autoencoder')
    parser.add_argument('--L', default=None, type=int,
                        help='Length of the filters in samples (40=5ms at 8kHZ)')
    parser.add_argument('--B', default=None, type=int,
                        help='Number of channels in bottleneck 1 × 1-conv block')
    parser.add_argument('--H', default=None, type=int,
                        help='Number of channels in convolutional blocks')
    parser.add_argument('--P', default=None, type=int,
                        help='Kernel size in convolutional blocks')
    parser.add_argument('--X', default=None, type=int,
                        help='Number of convolutional blocks in each repeat')
    parser.add_argument('--R', default=None, type=int,
                        help='Number of repeats')
    parser.add_argument('--C', default=None, type=int,
                        help='Number of out classes')
    parser.add_argument('--norm_type', default=None, type=str,
                        choices=['gLN', 'cLN', 'BN'], help='Layer norm type')
    parser.add_argument('--causal', type=int, default=None,
                        help='Causal (1) or noncausal(0) training')
    parser.add_argument('--mask_nonlinear', default=None, type=str,
                        choices=['relu', 'softmax'], help='non-linear to generate mask')
    # dataset configuration
    parser.add_argument('--data_location', type=str,
                        help='directory of the data files', default=None)
    parser.add_argument('-sr', "--sample_rate", type=int,
                        help='sample rate used to load wav files', default=None)
    parser.add_argument("--random_load", type=str2bool, nargs='?',
                        help="Load the training data with random init", const=True, default=None)
    parser.add_argument('-csv', '--csv_location', type=str,
                        help='directory where to find the csv files test_data.csv, train_data.csv, valid_data.csv', default=None)
    parser.add_argument("--data_suffix", type=str,
                        help="suffix of the data files", default=None)
    # learning parameters
    parser.add_argument('-m', '--model_save_path', type=str,
                        help="path to the file to save the model", default=None)
    parser.add_argument('--model_to_load_path', type=str,
                        help='path of partially trained network to load', default=None)
    parser.add_argument('-lr', '--learning_rate', type=float,
                        help='learning rate', default=None)
    parser.add_argument('-e', '--epochs', type=int, help='epochs', default=None)
    parser.add_argument('--batch_size', type=int,
                        help='minibatch size', default=None)
    parser.add_argument('--step_size', type=int,
                        help='number of epochs between each scheduler step', default=None)
    parser.add_argument('-g', '--gamma', type=float,
                        help='multiplicative factor of the scheduler step (how much does the learning rate shrink)', default=None)

    args = parser.parse_args()
    return args


def override_config_with_args(config, args):
    for c in config:
        if hasattr(args, c) and getattr(args, c) != None:
            print("Override di {} con il valore {}".format(c, getattr(args, c)))
            config[c] = getattr(args, c)
    return config

def main():
    args = get_arguments()
    # check if the file specified to save the model exists
    if args.model_save_path is not None and Path(args.model_save_path).is_file():
        ans = input(
            "The specified model file %s already exists, do you really want to overwrite it (y/n)? " % args.model_save_path)
        if not str2bool(ans):
            quit()

    # load the config files
    with open(args.dataset_config) as f:
        dataset_config = yaml.safe_load(f)
    with open(args.model_config) as f:
        model_config = yaml.safe_load(f)
    with open(args.training_config) as f:
        training_config = yaml.safe_load(f)

    # TODO: cosa faccio con gli arg per i dataset?
    #override_config_with_args(dataset_config, args)
    override_config_with_args(model_config, args)
    override_config_with_args(training_config, args)

    if args.wandb:
        import wandb
        tags = [dataset_config["TAG"], model_config["MODEL"], args.tags] if args.tags is not None else [dataset_config["TAG"], model_config["MODEL"]]
        wandb.init(config={"dataset": dataset_config, "model": model_config, "training":training_config}, project="Audio Emotion Recognition", 
            save_code=True, tags=tags)
        wandb.save("src/*.py")
        if args.model_save_path is None:
            modelname = (Path("./models/WandB/") /
                        wandb.run.name).with_suffix(".pt")
            training_config["model_save_path"] = modelname
    else:
        wandb = None
        if args.model_save_path is None:
            raise RuntimeError(
                "If wandb is not active you MUST specify a model name!")

    # Set device as cuda if available, otherwise cpu
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('Using device %s' % device)

    # load the datasets
    train_set_generator, valid_set_generator, test_set_generator = data_loader.load_datasets(dataset_config)

    # load the model
    model = data_loader.load_model(model_config, training_config["model_to_load_path"])
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=training_config["learning_rate"])

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=training_config["step_size"], gamma=training_config["gamma"])

    if training_config["criterion"] == "cross_entropy":
        criterion = torch.nn.CrossEntropyLoss()
    elif training_config["criterion"] == "BCE_logits":
        criterion = torch.nn.BCEWithLogitsLoss()

    try:
        multiclass_labels = training_config["multiclass_labels"]
    except KeyError:
        multiclass_labels = False

    best_model, last_model = train(model, criterion, optimizer, scheduler, train_set_generator,
        valid_set_generator, device, wandb, dataset_config, model_config, training_config)

    test(best_model, last_model, device, test_set_generator, wandb, multiclass_labels)

if __name__ == '__main__':
    main()