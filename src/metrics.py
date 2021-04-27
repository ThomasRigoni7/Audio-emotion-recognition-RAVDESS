import torch
import numpy as np
from utils import *
import sklearn

# for the 1 dimensional case
def unweighted_accuracy(predictions, ground_truths):
    _, pred = torch.max(predictions.detach().cpu(), dim=1)
    correct = (pred == ground_truths)
    acc = torch.mean(correct.float())
    return acc * 100

def unweighted_accuracy_wrapper(model, generator, device):
    model.eval()
    pred, ground = get_predictions(model, generator, device)
    return weighted_accuracy(pred, ground, classes)


def weighted_accuracy(predictions, ground_truths, classes=None):
    _, pred = torch.max(predictions.detach().cpu(), dim=1)
    # get the number of classes in the ground truths if it's not given
    if classes is None:
        classes = torch.unique(ground_truths)
    correct = torch.zeros(len(classes))
    total = torch.zeros(len(classes))
    for p, g in zip(pred, ground_truths):
        correct[int(g)] += (p == g)
        total[int(g)] += 1
    acc = torch.mean(correct/total)
    return 100 * acc

def weighted_accuracy_wrapper(model, generator, device):
    model.eval()
    classes = generator.dataset.classes
    pred, ground = get_predictions(model, generator, device)
    return weighted_accuracy(pred, ground, classes)


# multilabel
def accuracy_multilabel(predictions, ground_truths):
    # an emotion is predicted only if the prediction is > 0
    pred = predictions > 0
    ground = ground_truths > 0
    correct = (pred == ground)
    acc = torch.mean(correct.float())
    return acc * 100

def accuracy_multilabel_wrapper(model, generator, device):
    model.eval()
    pred, ground = get_predictions(model, generator, device)
    return unweighted_accuracy_multiclass_labels(pred, ground)

def f1_score(predictions, ground_truths):
    pred = predictions > 0
    ground = ground_truths > 0
    return sklearn.metrics.f1_score(ground, pred, average=None)

def conf_matrix(predictions, ground_truths, classes):
    import wandb as w_b
    _, pred = torch.max(predictions.detach().cpu(), dim=1)
    pred_np = pred.numpy()
    ground_truths_np = ground_truths.numpy()

    return w_b.plot.confusion_matrix(preds=pred_np, y_true = ground_truths_np, class_names=classes)


def get_metrics(model, generator, device, metrics: list, criterion=None):
    '''
    metrics is a list of strings indicating the metrics to be calculated, the choices are:
    - wacc -> weighted accuracy
    - uacc -> unweighted accuracy
    - acc_multilabel -> multilabel accuracy 
    - f1 -> f1 score
    - conf_matrix -> returns the confusion matrix calculated by wandb
    - loss -> returns the loss based on the criterion specified

    returns a dictionary containing the metrics requested
    '''
    classes = generator.dataset.classes
    pred, ground = get_predictions(model, generator, device)
    res = {}
    for metric in metrics:
        if metric == "wacc":
            res["wacc"] = weighted_accuracy(pred, ground)
        elif metric == "uacc":
            res["uacc"] = unweighted_accuracy(pred, ground)
        elif metric == "acc_multilabel":
            res["acc_multilabel"] = accuracy_multilabel(pred, ground)
        elif metric == "f1":
            res["f1"] = f1_score(pred, ground)
        elif metric == "conf_matrix":
            res["conf_matrix"] = conf_matrix(pred, ground, classes)
        elif metric == "loss":
            if criterion is None:
                raise ValueError("Cannot calculate the loss if a criterion is not specified!")
            res["loss"] = criterion(pred, ground)
        else:
            raise ValueError("Invalid metric {}".format(metric))
    return res

    

if __name__ == "__main__":
    pred = torch.Tensor([[ 3, -1],
                         [ 1, 1]])
    truth = torch.Tensor([[ 1, 2],
                          [ 0, -2]])
    acc = unweighted_accuracy_multiclass_labels(pred, truth)
    f1_score = f1_score(pred, truth)
    metrics = get_metrics()
    print("Acc: ", acc)
    print("F1 : ", f1_score)