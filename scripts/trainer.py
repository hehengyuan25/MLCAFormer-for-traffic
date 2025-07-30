import copy
import datetime
import time

import numpy as np
import torch
import matplotlib.pyplot as plt

from .metrics import RMSE_MAE_MAPE
from utils.helpers import print_log


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, scaler, device,
                 train_loader, val_loader, test_loader, log=None):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.scaler = scaler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.log = log

    def _train_one_epoch(self, clip_grad=0.0):
        self.model.train()
        batch_loss_list = []

        for x_batch, y_batch in self.train_loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)
            out_batch = self.model(x_batch)
            out_batch = self.scaler.inverse_transform(out_batch)

            loss = self.criterion(out_batch, y_batch)
            batch_loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            if clip_grad > 0.0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
            self.optimizer.step()

        epoch_loss = np.mean(batch_loss_list)
        self.scheduler.step()
        return epoch_loss

    @torch.no_grad()
    def _evaluate(self, loader):
        self.model.eval()
        batch_loss_list = []
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = self.model(x_batch)
            out_batch = self.scaler.inverse_transform(out_batch)
            loss = self.criterion(out_batch, y_batch)
            batch_loss_list.append(loss.item())
        return np.mean(batch_loss_list)

    @torch.no_grad()
    def _predict(self, loader):
        self.model.eval()
        y_true = []
        y_pred = []

        for x_batch, y_batch in loader:
            x_batch = x_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            out_batch = self.model(x_batch)
            out_batch = self.scaler.inverse_transform(out_batch)

            out_batch = out_batch.cpu().numpy()
            y_batch = y_batch.cpu().numpy()
            y_pred.append(out_batch)
            y_true.append(y_batch)

        y_pred = np.vstack(y_pred).squeeze()
        y_true = np.vstack(y_true).squeeze()
        return y_true, y_pred

    def train_epochs(self, max_epochs, early_stop, clip_grad=0.0, verbose=1, plot=False, save_path=None):
        wait = 0
        min_val_loss = np.inf
        best_epoch = 0
        best_state_dict = None

        train_loss_list = []
        val_loss_list = []

        for epoch in range(max_epochs):
            train_loss = self._train_one_epoch(clip_grad=clip_grad)
            train_loss_list.append(train_loss)

            val_loss = self._evaluate(self.val_loader)
            val_loss_list.append(val_loss)

            if (epoch + 1) % verbose == 0:
                print_log(
                    datetime.datetime.now(),
                    "Epoch", epoch + 1,
                             "\tTrain Loss = %.5f" % train_loss,
                             "Val Loss = %.5f" % val_loss,
                    log=self.log,
                )

            if val_loss < min_val_loss:
                wait = 0
                min_val_loss = val_loss
                best_epoch = epoch
                best_state_dict = copy.deepcopy(self.model.state_dict())
            else:
                wait += 1
                if wait >= early_stop:
                    break

        self.model.load_state_dict(best_state_dict)
        train_rmse, train_mae, train_mape = RMSE_MAE_MAPE(*self._predict(self.train_loader))
        val_rmse, val_mae, val_mape = RMSE_MAE_MAPE(*self._predict(self.val_loader))

        out_str = f"Early stopping at epoch: {epoch + 1}\n"
        out_str += f"Best at epoch {best_epoch + 1}:\n"
        out_str += "Train Loss = %.5f\n" % train_loss_list[best_epoch]
        out_str += "Train RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (train_rmse, train_mae, train_mape)
        out_str += "Val Loss = %.5f\n" % val_loss_list[best_epoch]
        out_str += "Val RMSE = %.5f, MAE = %.5f, MAPE = %.5f" % (val_rmse, val_mae, val_mape)
        print_log(out_str, log=self.log)

        if plot:
            plt.plot(range(0, epoch + 1), train_loss_list, "-", label="Train Loss")
            plt.plot(range(0, epoch + 1), val_loss_list, "-", label="Val Loss")
            plt.title("Epoch-Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.show()

        if save_path:
            torch.save(best_state_dict, save_path)

    def test_model(self):
        print_log("--------- Test ---------", log=self.log)

        start = time.time()
        y_true, y_pred = self._predict(self.test_loader)
        end = time.time()

        rmse_all, mae_all, mape_all = RMSE_MAE_MAPE(y_true, y_pred)
        out_str = "All Steps RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (rmse_all, mae_all, mape_all)

        out_steps = y_pred.shape[1]
        for i in range(out_steps):
            rmse, mae, mape = RMSE_MAE_MAPE(y_true[:, i, :], y_pred[:, i, :])
            out_str += "Step %d RMSE = %.5f, MAE = %.5f, MAPE = %.5f\n" % (i + 1, rmse, mae, mape)

        print_log(out_str, log=self.log, end="")
        print_log("Inference time: %.2f s" % (end - start), log=self.log)