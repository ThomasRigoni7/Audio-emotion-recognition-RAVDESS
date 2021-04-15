import copy
import torch
from pathlib import Path
import numpy as np
from utils import *


def test(model_best, model_last, device, test_set_generator, wandb):
    model_best.eval()
    model_last.eval()
    wacc_best = weighted_accuracy(model_best, test_set_generator, device, test_set_generator.dataset.classes)
    wacc_last = weighted_accuracy(model_last, test_set_generator, device, test_set_generator.dataset.classes)
    uacc_best = unweighted_accuracy(model_best, test_set_generator, device)
    uacc_last = unweighted_accuracy(model_last, test_set_generator, device)

    predicted_best, ground_best = conf_matrix(model_best, test_set_generator, device)
    predicted_last, ground_last = conf_matrix(model_last, test_set_generator, device)
    print("")
    print("########################")
    print("")

    print("Results on the TEST set:")
    print("weighted accuracy (best): %2.2f%%" % wacc_best)
    print("weighted accuracy (last): %2.2f%%" % wacc_last)
    print("unweighted accuracy (best): %2.2f%%" % uacc_best)
    print("unweighted accuracy (last): %2.2f%%" % uacc_last)

    if wandb is not None:
        wandb.log({"weighted_test_accuracy_best": wacc_best,
                   "weighted_test_accuracy_last": wacc_last,
                   "unweighted_test_accuracy_best": uacc_best,
                   "unweighted_test_accuracy_last": uacc_last,
                   "test_confusion_matrix_best": wandb.plot.confusion_matrix(preds=predicted_best, y_true = ground_best, class_names=test_set_generator.dataset.classes),
                   "test_confusion_matrix_last": wandb.plot.confusion_matrix(preds=predicted_last, y_true = ground_last, class_names=test_set_generator.dataset.classes)})