import torch
from torch.utils import data
from pathlib import Path
from torch.utils.data import WeightedRandomSampler

from models.TCN import TCN
from models.conv_tasnet import ConvTasNet

from dataset_ravdess import RAVDESS_DATA
from dataset_BAUM2 import BAUM2_DATA
from dataset_MELD import MELD_DATA

######################
# MODEL
######################

def load_model(model_config, model_to_load_path):
    if model_config["MODEL"] == "TCN":
        model = TCN(n_blocks=model_config["blocks"], n_repeats=model_config["repeats"],
                    in_chan=model_config["in_classes"], out_chan=model_config["out_classes"], dropout_prob=model_config["dropout_prob"])

    elif model_config["MODEL"] == "CONV-TASNET":
        model = ConvTasNet(model_config["N"], model_config["L"], model_config["B"], model_config["H"], model_config["P"], model_config["X"], model_config["R"],
                           model_config["C"], norm_type=model_config["norm_type"], causal=model_config["causal"],
                           mask_nonlinear=model_config["mask_nonlinear"], dropout_prob=model_config["dropout_prob"])

    if model_to_load_path is not None:
        model.load_state_dict(torch.load(model_to_load_path)["model"])

    return model

######################
# DATASET
######################

def _load_generator(dataset, batch_size, sampler, shuffle):
    params = {'batch_size': batch_size,
              'num_workers': 0}
    set_generator = None
    if sampler == True:
        _, samples_weight = dataset.get_class_sample_count()
        sampler = WeightedRandomSampler(samples_weight, len(samples_weight))
        set_generator = data.DataLoader(
            dataset, sampler=sampler, shuffle=False, **params)
    else:
        set_generator = data.DataLoader(dataset, **params, shuffle=shuffle)
    return set_generator

def _load_ravdess(config):
    return RAVDESS_DATA(config["csv_location"],
                        data_dir=Path(config["data_location"]) / "", random_load=config["random_load"], in_suffix=config["data_suffix"], sr=config["sample_rate"], transformations=config["transformations"])

def _load_baum2(config):
    raise NotImplementedError("BAUM2 LOADER NOT IMPLEMENTED YET")

def _load_meld(config):
    return MELD_DATA(config["csv_location"], data_dir=config["data_location"], in_suffix=config["data_suffix"], sr=config["sample_rate"],transformations=config["transformations"], chunk_len=config["chunk_len"])
    


def _switch_dataset(config,  training=False):
    dataset = None
    if config["DATASET"] == "RAVDESS":
        dataset = _load_ravdess(config)
    elif config["DATASET"] == "BAUM2":
        dataset = _load_baum2(config)
    elif config["DATASET"] == "MELD":
        dataset = _load_meld(config)
    return _load_generator(dataset, config["batch_size"], config["sampler"], training)



def load_datasets(dataset_config):
    train = _switch_dataset(dataset_config["train"], training=True)
    valid = _switch_dataset(dataset_config["valid"], training=False)
    test = _switch_dataset(dataset_config["test"], training=False)

    return (train, valid, test)
