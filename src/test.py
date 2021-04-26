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

    if not multiclass_labels:
        metrics_best = get_metrics(model_best, test_set_generator, device, ["wacc", "uacc", "conf_matrix"])
        metrics_last = get_metrics(model_best, test_set_generator, device, ["wacc", "uacc", "conf_matrix"])
        print("weighted accuracy (best): %2.2f%%" % metrics_best["wacc"])
        print("unweighted accuracy (best): %2.2f%%" % metrics_best["uacc"])
        print("cofusion matrix (best)")
        print(metrics_best["conf_matrix"])
        print("")
        print("weighted accuracy (last): %2.2f%%" % metrics_last["wacc"])
        print("unweighted accuracy (last): %2.2f%%" % metrics_last["uacc"])
        print("cofusion matrix (last)")
        print(metrics_last["conf_matrix"])

        if wandb is not None:
            wandb.log({"weighted_test_accuracy_best": metrics_best["wacc"],
                    "weighted_test_accuracy_last": metrics_last["wacc"],
                    "unweighted_test_accuracy_best": metrics_best["uacc"],
                    "unweighted_test_accuracy_last": metrics_last["uacc"],
                    "test_confusion_matrix_best": metrics_best["conf_matrix"],
                    "test_confusion_matrix_last": metrics_last["conf_matrix"]})
    else:
        metrics_best = get_metrics(model_best, test_set_generator, device, ["acc_multilabel", "f1"])
        metrics_last = get_metrics(model_best, test_set_generator, device, ["acc_multilabel", "f1"])
        print("multilabel accuracy (best): %2.2f%%" % metrics_best["wacc"])
        print("f1 score (best): %2.2f" % metrics_best["f1"])
        print("")
        print("multilabel accuracy (best): %2.2f%%" % metrics_best["wacc"])
        print("f1 score (best): %2.2f" % metrics_best["f1"])

        if wandb is not None:
            wandb.log({"multilabel_accuracy_best": metrics_best["acc_multilabel"],
                    "multilabel_accuracy_last": metrics_last["acc_multilabel"],
                    "f1_score_best": metrics_best["f1"],
                    "f1_score_last": metrics_last["f1"]})

        