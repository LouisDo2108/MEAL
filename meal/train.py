import os
import sys
import json
import cv2
import numpy as np
from pathlib import Path
from copy import deepcopy
from natsort import natsorted
import albumentations as A
from sklearn.metrics import f1_score

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from utils.general import LOGGER
from meal.models import MEAL_with_masked_attention, MEAL, init_model
from meal.datasets import get_dataloaders, get_transforms


def train(model, params, model_name):

    num_epochs = params["num_epochs"]
    loss_func = params["loss_func"]
    opt = params["optimizer"]
    train_dl = params["train_dl"]
    val_dl = params["val_dl"]
    lr_scheduler = params["lr_scheduler"]
    save_path = params["save_path"]

    # history of loss values in each epoch
    loss_history = {
        "train": [],
        "val": [],
    }
    # history of metric values in each epoch
    metric_history = {
        "train": {"micro": [], "weighted": []},
        "val": {"micro": [], "weighted": []},
    }

    # a deep copy of weights for the best performing model
    best_model_wts = deepcopy(model.state_dict())

    # initialize best loss to a large value
    best_loss = float("inf")

    model = model.to(model.device)
    
    def metrics_batch(output, target):
        output = torch.argmax(output, dim=1).cpu().numpy().tolist()
        target = target.cpu().numpy().tolist()
        weighted_score = f1_score(output, target, average="weighted")
        micro_score = f1_score(output, target, average="micro")
        return [weighted_score, micro_score]


    def loss_batch(loss_func, output, target, opt=None):
        loss = loss_func(output, target)
        with torch.no_grad():
            weighted_score, score_score = metrics_batch(output, target)
        if opt is not None:
            opt.zero_grad()
            loss.backward()
            opt.step()
        return loss.item(), (weighted_score, score_score)


    def loss_epoch(model, loss_func, dataset_dl, opt=None):
        running_loss = 0.0
        micro_score = 0.0
        weighted_score = 0.0
        count = 0
        for ix, (x, y) in enumerate(dataset_dl):
            if model_name == "MEAL":
                _, _, _, bbox, _ = x
                ylist = []
                for b, cls in zip(bbox, y):
                    ylist.extend([cls] * b.shape[0])
                y = torch.tensor(ylist, dtype=torch.long)
            elif model_name == "MEAL_with_masked_attention":
                pass
            else:
                print("FATAL ERROR: NO MODEL NAMED {}".format(model_name))
                return
            y = y.to(model.device)
            output = model(x, debug=True)
            loss_b, metric_b = loss_batch(loss_func, output, y, opt)
            running_loss += loss_b
            if metric_b is not None:
                micro_score += metric_b[0]
                weighted_score += metric_b[1]
            count += 1

        # average loss value
        loss = running_loss / count
        # average metric value
        micro_metric = micro_score / count
        weighted_metric = weighted_score / count
        return loss, (micro_metric, weighted_metric)


    def get_lr(opt):
        for param_group in opt.param_groups:
            return param_group["lr"]

    # main loop
    for epoch in range(num_epochs):
        # get current learning rate
        current_lr = get_lr(opt)
        LOGGER.info(
            "Epoch {}/{}, current lr={}".format(epoch, num_epochs - 1, current_lr)
        )
        # train model on training dataset
        model.train()
        train_loss, train_metric = loss_epoch(model, loss_func, train_dl, opt)
        # collect loss and metric for training dataset
        loss_history["train"].append(train_loss)
        metric_history["train"]["micro"].append(train_metric[0])
        metric_history["train"]["weighted"].append(train_metric[1])

        model.eval()
        with torch.no_grad():
            val_loss, val_metric = loss_epoch(model, loss_func, val_dl)
            # collect loss and metric for validation dataset
            loss_history["val"].append(val_loss)
            metric_history["val"]["micro"].append(val_metric[0])
            metric_history["val"]["weighted"].append(val_metric[1])

        # store best model
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_wts = deepcopy(model.state_dict())
            # store weights into a local file
            torch.save(model.state_dict(), save_path)
            LOGGER.info("Copied best model weights!")

        lr_scheduler.step(val_loss)
        if current_lr != get_lr(opt):
            LOGGER.info("Loading best model weights!")
            model.load_state_dict(best_model_wts)

        LOGGER.info(
            "train loss: {:.6f}, val loss: {:.6f}, f1-micro: {:.2f}, f1-weighted: {:.2f}".format(
                train_loss, val_loss, 100 * val_metric[0], 100 * val_metric[1]
            )
        )
        LOGGER.info("-" * 10)

    return model, loss_history, metric_history


if __name__ == "__main__":
    
    train_transforms, val_transforms = get_transforms()
    
    train_dl, val_dl = get_dataloaders(
        train_transforms, val_transforms,
        batch_size=256,
        fold=0,
    )

    model, loss_func, opt, lr_scheduler = init_model(model_name="MEAL")
    params = {
        "num_epochs": 100,
        "optimizer": opt,
        "loss_func": loss_func,
        "train_dl": train_dl,
        "val_dl": val_dl,
        "lr_scheduler": lr_scheduler,
        "save_path": "~/checkpoints/new_model.pt"
    }
    
    model, loss_hist, metric_hist = train(
        model=model,
        params=params,
        model_name=model_name,
    )
    
    
    