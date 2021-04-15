import copy
import torch
from pathlib import Path
import numpy as np
from utils import *


def train(model, criterion, optimizer, scheduler, train_set_generator, valid_set_generator, device, wandb, dataset_config, model_config, training_config):

    best_accuracy = 0
    train_gen_len = len(train_set_generator)
    if wandb is not None:
        wandb.watch(model)

    for e in range(training_config["epochs"]):
        sum_loss = 0
        for i, d in enumerate(train_set_generator):
            model.train()
            f, l = d
            f = torch.squeeze(f, dim=1)
            y = model(f.float().to(device))
            loss = criterion(y, l.to(device))
            sum_loss += loss

            print("Iteration %d in epoch %d--> loss = %f" %
                  (i, e + 1, loss.item()), end='\r')
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        scheduler.step()
        model.eval()
        acc_train = unweighted_accuracy(model, train_set_generator, device)
        acc_valid = unweighted_accuracy(model, valid_set_generator, device)
        # accuracy
        iter_acc = 'iteration %d epoch %d--> %f (%f)' % (
            i, e + 1, acc_valid, best_accuracy)
        print(iter_acc)
        if wandb is not None:
            wandb.log({"loss": sum_loss/train_gen_len, "training accuracy": acc_train,
                       "validation accuracy": acc_valid})
        sum_loss = 0

        if acc_valid > best_accuracy:
            improved_accuracy = 'Current accuracy = %f (%f), updating best model' % (
                acc_valid, best_accuracy)
            print(improved_accuracy)
            best_accuracy = acc_valid
            best_epoch = e
            best_model = copy.deepcopy(model)
            torch.save(
                {"dataset_config": dataset_config, "model_config": model_config, "training_config": training_config, "model": model.state_dict()}, training_config["model_save_path"])
        # stop if heavy overfitting
        if acc_train > 99 and acc_valid < 50:
            print("Training stopped for overfitting! Training acc: %2.2f, Validation acc: %2.2f" % (
                acc_train, acc_valid))
            break

    wacc_last = weighted_accuracy(model, valid_set_generator, device, valid_set_generator.dataset.classes)
    wacc_best = weighted_accuracy(best_model, valid_set_generator, device, valid_set_generator.dataset.classes)
    print("weighted accuracy (last): %2.2f%%" % wacc_last)
    print("weighted accuracy (best): %2.2f%%" % wacc_best)
    if wandb is not None:
        wandb.log({"best validation accuracy": best_accuracy, "weighted validation accuracy last": wacc_last, "weighted validation accuracy best": wacc_best})
        predicted, ground = conf_matrix(model, valid_set_generator, device)
        wandb.log({"confusion_matrix_last": wandb.plot.confusion_matrix(preds=predicted, y_true = ground, class_names=valid_set_generator.dataset.classes)})
    print("Best accuracy on validation set reached at epoch %d with %2.2f%%" %
          (best_epoch + 1, best_accuracy))
    return best_model, model
