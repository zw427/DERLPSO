import deepdish as dd
from torch.utils.data import Dataset


class SimpleDataset(Dataset):
    """
    Creates a data-loader for the wave prop data
    """

    def __init__(self, filename):
        self.filename = filename
        dataset = dd.io.load(filename)

        self.origin_data= dataset['origin_data']
        self.model_data = dataset['model_data']
        self.math_data= dataset['math_data']
        self.compare_data = dataset['compare_data']
        self.origin_param= dataset['origin_param']
        self.model_param=dataset['model_param']
        self.math_param=dataset['math_param']
        self.compare_param=dataset['compare_param']

    def __len__(self):
        return len(self.origin_data)

    def __getitem__(self, idx):

         sample=  [self.origin_data[idx],
        self.model_data[idx],
        self.math_data[idx],
        self.compare_data[idx],
        self.origin_param[idx],
        self.model_param[idx],
        self.math_param[idx],
        self.compare_param[idx]]
         return sample


