# run.py
import argparse
import datetime
import json
import os
import sys
import time

import torch
import torch.nn as nn
import yaml


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from scripts.trainer import Trainer
from data_processing.data_handler import load_and_prepare_data
from model_arch.mlcaformer import MLCAFormer
from utils.helpers import (CustomJSONEncoder, print_log, seed_everything,
                           set_cpu_num)
from utils.losses import MaskedMAELoss


def main(cfg, dataset_name):

    seed = cfg.get("seed")
    seed_everything(seed)
    set_cpu_num(1)

    GPU_ID = cfg.get("gpu_num",1)
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{GPU_ID}"
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_path = f"./data/{dataset_name}"
    model_name = "MLCAFormer"

    now = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_path = "../logs"
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    log_file = os.path.join(log_path, f"{model_name}-{dataset_name}-{now}.log")
    log = open(log_file, "a")
    log.seek(0)
    log.truncate()

    print_log(f"Loading dataset: {dataset_name}", log=log)
    (
        trainset_loader,
        valset_loader,
        testset_loader,
        scaler,
    ) = load_and_prepare_data(
        data_path,
        tod=cfg.get("time_of_day"),
        dow=cfg.get("day_of_week"),
        batch_size=cfg.get("batch_size", 16),
        log=log,
    )


    model = MLCAFormer(**cfg["model_args"])
    model.to(DEVICE)


    save_path = "../saved_models"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_file = os.path.join(save_path, f"{model_name}-{dataset_name}-{now}.pt")


    if dataset_name in ("METRLA", "PEMSBAY"):
        criterion = MaskedMAELoss()
    elif dataset_name in ("PEMS04", "PEMS08"):
        criterion = nn.HuberLoss()
    else:
        raise ValueError("Unsupported dataset.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0),
        eps=cfg.get("eps", 1e-8),
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["milestones"],
        gamma=cfg.get("lr_decay_rate", 0.1),
        verbose=False,
    )


    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        scaler=scaler,
        device=DEVICE,
        train_loader=trainset_loader,
        val_loader=valset_loader,
        test_loader=testset_loader,
        log=log
    )

    trainer.train_epochs(
        max_epochs=cfg.get("max_epochs", 200),
        early_stop=cfg.get("early_stop", 10),
        clip_grad=cfg.get("clip_grad", 0.0),
        save_path=save_file
    )

    print_log(f"Saved Model: {save_file}", log=log)

    trainer.test_model()

    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, default="METRLA", help="Dataset name, e.g., METRLA, PEMS04")
    parser.add_argument("-g", "--gpu_num", type=int, default=0, help="GPU device ID")
    args = parser.parse_args()

    dataset_name = args.dataset.upper()

    # Load config file
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", "model_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if dataset_name not in config:
        raise ValueError(f"Dataset '{dataset_name}' not found in model_config.yaml")

    dataset_cfg = config[dataset_name]
    dataset_cfg['gpu_num'] = args.gpu_num

    main(dataset_cfg, dataset_name)