import copy
import torch
from pathlib import Path
import numpy as np
from metrics import *
from utils import get_predictions


def test(model_best, model_last, device, test_set_generator, wandb, multiclass_labels):
    model_best.eval()
    model_last.eval()
    print("")
    print("########################")
    print("")
    print("Results on the TEST set:")

    metrics_single_label = ["wacc", "uacc"]
    if wandb is not None:
        metrics_single_label.append("conf_matrix")

    if not multiclass_labels:
        metrics_best = get_metrics(model_best, test_set_generator, device, metrics_single_label)
        metrics_last = get_metrics(model_last, test_set_generator, device, metrics_single_label)
        print("weighted accuracy (best): %2.2f%%" % metrics_best["wacc"])
        print("unweighted accuracy (best): %2.2f%%" % metrics_best["uacc"])
        print("")
        print("weighted accuracy (last): %2.2f%%" % metrics_last["wacc"])
        print("unweighted accuracy (last): %2.2f%%" % metrics_last["uacc"])

        if wandb is not None:
            wandb.log({"weighted test accuracy best": metrics_best["wacc"],
                    "weighted test accuracy last": metrics_last["wacc"],
                    "unweighted test accuracy best": metrics_best["uacc"],
                    "unweighted test accuracy last": metrics_last["uacc"],
                    "test confusion matrix best": metrics_best["conf_matrix"],
                    "test confusion matrix last": metrics_last["conf_matrix"]})
    else:
        metrics_best = get_metrics(model_best, test_set_generator, device, ["acc_multilabel", "f1"])
        metrics_last = get_metrics(model_best, test_set_generator, device, ["acc_multilabel", "f1"])
        print("multilabel accuracy (best): %2.2f%%" % metrics_best["acc_multilabel"])
        print("f1 score (best): %2.2f" % metrics_best["f1"])
        print("")
        print("multilabel accuracy (best): %2.2f%%" % metrics_best["acc_multilabel"])
        print("f1 score (best): %2.2f" % metrics_best["f1"])

        if wandb is not None:
            wandb.log({"multilabel accuracy best": metrics_best["acc_multilabel"],
                    "multilabel accuracy last": metrics_last["acc_multilabel"],
                    "f1 score best": metrics_best["f1"],
                    "f1 score last": metrics_last["f1"]})

        