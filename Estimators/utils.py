import os
import sys

import numpy as np
import torch
import yaml
from torch.utils.data import Dataset, random_split


def load_configure(config_file, model_type):
    """
    Load configuration from yaml file
    """
    if not os.path.exists(config_file):
        print(f"{config_file} does not exists.")
        sys.exit()

    with open(config_file, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
            print(configs)
            print("Loading config file successful.")
        except Exception as e:
            print(e)

    configs_param = configs["Param"]
    configs_param["normal"] = configs["Param"]["Net"]["normal"]
    configs_param["device"] = configs["device"]
    param = model_type
    configs_param["type"] = model_type
    if param is not None:
        if param == "VAE":
            values = configs_param["Net"]["VAE_Net"]
        elif param in ["ODE_RNN"]:
            values = configs_param["Net"]["ODE_RNN_Net"]
        elif param in ["RNN"]:
            values = configs_param["Net"]["RNN_Net"]
    if param is None or param == "MLP":
        values = configs_param["Net"]["MLP_Net"]
    configs_param["Net"].update(values)
    del configs_param["Net"]["ODE_RNN_Net"]
    del configs_param["Net"]["RNN_Net"]
    del configs_param["Net"]["MLP_Net"]
    del configs_param["Net"]["VAE_Net"]
    return configs_param


def split_data(data, time, param, train_frac=0.6):
    """
    Split data into training and testing sets based on the specified training fraction.
    """
    n_train = int(len(data) * train_frac)

    def split(arr):
        return arr[:n_train], arr[n_train:]

    data_train, data_test = split(data)
    time_train, time_test = split(time)
    param_train, param_test = split(param)

    train = {"data": data_train, "params": param_train, "time": time_train}
    test = {"data": data_test, "params": param_test, "time": time_test}
    return train, test


class EarlyStopping:
    def __init__(self, save_path, patience=7, verbose=False, delta=0):
        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        path = os.path.join(self.save_path, "best_loss.pth")
        entire_path = os.path.join(self.save_path, "entire.pth")
        torch.save(model.state_dict(), path)
        torch.save(model, entire_path)
        self.val_loss_min = val_loss


class SimpleDataSet(Dataset):
    """
    Creates a data-loader for the wzave prop data
    """

    def __init__(self, dataset):
        indexes = list(range(0, dataset["data"].shape[0]))
        self.data = torch.DoubleTensor(dataset["data"])[indexes]
        self.time = torch.DoubleTensor(dataset["time"])[indexes]
        if "params" in dataset.keys() and dataset["params"] is not None:
            self.params = torch.DoubleTensor(dataset["params"])[indexes]
        else:
            self.params = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = [
            self.data[idx],
            self.params[idx]
            if self.params is not None
            else np.zeros(self.data[idx].shape),
            self.time[idx],
        ]
        return sample

    def get_splits(self, n_test=0.0):
        train_size = len(self.data) - n_test
        return random_split(self, [train_size, n_test])

    def get_labels_min_max(self, labels):
        return torch.min(labels, dim=0)[0].data, torch.max(labels, dim=0)[0].data

    def get_data_min_max(self, data):
        return torch.min(data, dim=0)[0].data, torch.max(data, dim=0)[0].data

    def preprocess_data(self):
        with torch.no_grad():
            data_min, data_max = self.get_data_min_max(self.data)
            data_min = data_min.unsqueeze(0)
            data_max = data_max.unsqueeze(0)
            self.data = (self.data - data_min) / (data_max - data_min)
            self.data = torch.where(
                torch.isnan(self.data), torch.full_like(self.data, 0.01), self.data
            )
            scale = {"shift": data_min, "mult": (data_max - data_min)}
            print("normal successfully")
        return scale

    def postprocess_data(self, data_predict, scale):
        data_predict = data_predict * scale["mult"] + scale["shift"]
        return data_predict

    def preprocess_labels(self):
        with torch.no_grad():
            labels_param_min, labels_param_max = self.get_labels_min_max(self.params)
            labels_param_min = labels_param_min.unsqueeze(0)
            labels_param_max = labels_param_max.unsqueeze(0)
            labels_min = labels_param_min
            labels_max = labels_param_max
            scale = {"shift": labels_min, "mult": (labels_max - labels_min)}
        print("scale", scale)
        return scale

    def postprocess_label(self, label_preict, scale):
        label_preict = (label_preict - scale["shift"]) / scale["mult"]
        return label_preict
