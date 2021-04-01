import torch
from torch.utils import data
from TCN import TCN
from conv_tasnet import ConvTasNet
from pathlib import Path
from dataset_ravdess import RAVDESS_DATA
from dataset_BAUM2 import BAUM2_DATA

# lo faccio con file di configurazione o con gli argomenti da riga di comando?

# 3 file di config: model, dataset e training

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

# metti sampler
def load_dataset():
    # coose
    raise NotImplementedError()
    return train, valid, test