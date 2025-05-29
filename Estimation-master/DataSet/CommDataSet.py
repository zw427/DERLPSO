import numpy as np
import torch
from torch.utils.data import Dataset, random_split


class SimpleDataSet(Dataset):
    """
    Creates a data-loader for the wzave prop data
    """

    def __init__(self, dataset):


        indexes = list(range(0, dataset['data'].shape[0]))

        self.data = torch.DoubleTensor(dataset['data'])[indexes]
        self.time = torch.DoubleTensor(dataset['time'])[indexes]

        if 'params' in dataset.keys() and dataset['params'] is not None:
            self.params = torch.DoubleTensor(dataset['params'])[indexes]
        else:
            self.params=None
        if 'u_samples' in dataset.keys() and dataset['u_samples'] is not None:
            self.u_samples = torch.DoubleTensor(dataset['u_samples'])[indexes]
        else:
            self.u_samples = None





    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

         sample=  [self.data[idx],
                   self.params[idx] if self.params is not None else np.zeros(self.data[idx].shape),
                   self.u_samples[idx] if self.u_samples is not None else np.zeros(self.data[idx].shape),
                   self.time[idx]]
         return sample

    # get indexes for train and test rows
    def get_splits(self, n_test=0.):
        # determine sizes
        train_size = len(self.data) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])

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
            self.data=torch.where(torch.isnan(self.data), torch.full_like(self.data, 0.01), self.data)

            scale = {'shift': data_min, 'mult': (data_max - data_min)}
            print("normal successfully")
        return scale

    def postprocess_data(self, data_predict, scale):
        data_predict = data_predict * scale['mult'] + scale['shift']
        return data_predict

    def preprocess_labels(self):
            with torch.no_grad():
                labels_param_min, labels_param_max = self.get_labels_min_max(self.params)

                labels_param_min = labels_param_min.unsqueeze(0)
                labels_param_max = labels_param_max.unsqueeze(0)

                labels_min = labels_param_min
                labels_max = labels_param_max
                scale = {'shift': labels_min, 'mult': (labels_max - labels_min)}
            print("scale", scale)
            return scale
    def postprocess_label(self,label_preict,scale):
            label_preict = (label_preict - scale["shift"]) / scale["mult"]
            return label_preict





