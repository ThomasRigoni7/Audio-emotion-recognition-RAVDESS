import copy
import torch
from utils import *
from metrics import *


def train(model, criterion, optimizer, scheduler, train_set_generator, valid_set_generator, device, wandb, epochs, model_save_path, multiclass_labels):
    
    best_accuracy = 0
    train_gen_len = len(train_set_generator)
    if wandb is not None:
        wandb.watch(model)

    for e in range(epochs):
        for i, d in enumerate(train_set_generator):
            model.train()
            f, l = d
            f = torch.squeeze(f, dim=1)
            y = model(f.float().to(device))
            loss = criterion(y, l.to(device))
            
            print("Iteration %d in epoch %d--> loss = %f" %
                  (i, e + 1, loss.item()), end='\r')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        model.eval()
        # accuracy
        acc_valid = 0
        if not multiclass_labels:
            metrics_train = get_metrics(model, train_set_generator, device, ["uacc", "wacc", "loss"], criterion=criterion)
            metrics_valid = get_metrics(model, valid_set_generator, device, ["uacc", "wacc", "loss"], criterion=criterion)
            acc_train = metrics_train["uacc"]
            acc_valid = metrics_valid["uacc"]
            if wandb is not None:
                wandb.log({
                    "training loss": metrics_train["loss"],
                    "validation loss": metrics_valid["loss"],
                    "training weighted accuracy": metrics_train["wacc"],
                    "validation weighted accuracy": metrics_valid["wacc"],
                    "training unweighted accuracy": metrics_train["uacc"],
                    "validation unweighted accuracy": metrics_valid["uacc"]
                })
        else:
            metrics_train = get_metrics(model, train_set_generator, device, ["acc_multilabel", "f1", "loss"], criterion=criterion)
            metrics_valid = get_metrics(model, valid_set_generator, device, ["acc_multilabel", "f1", "loss"], criterion=criterion)
            acc_train = metrics_train["acc_multilabel"]
            acc_valid = metrics_valid["acc_multilabel"]
            if wandb is not None:
                wandb.log({
                    "training loss": metrics_train["loss"],
                    "validation loss": metrics_valid["loss"],
                    "training multilabel accuracy": metrics_train["acc_multilabel"],
                    "validation multilabel accuracy": metrics_valid["acc_multilabel"],
                    "training F1 score ": metrics_train["f1"],
                    "validation F1 score": metrics_valid["f1"]
                })

        iter_acc = 'iteration %d epoch %d--> %f (%f)' % (i, e + 1, acc_valid, best_accuracy)
        print(iter_acc)

        if acc_valid > best_accuracy:
            improved_accuracy = 'Current accuracy = %f (%f), updating best model' % (
                acc_valid, best_accuracy)
            print(improved_accuracy)
            best_accuracy = acc_valid
            best_epoch = e
            best_model = copy.deepcopy(model)
            torch.save(
                {"model": model.state_dict()}, model_save_path)
        
        # stop if heavy overfitting
        if acc_train > 99 and acc_valid < 50:
            print("Training stopped for overfitting! Training acc: %2.2f, Validation acc: %2.2f" % (
                acc_train, acc_valid))
            break
    
    return best_model, model

