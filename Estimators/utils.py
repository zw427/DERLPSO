
import logging
import os
import sys

import yaml


import numpy as np
import torch

from torch.nn import functional as F


def load_configure(configure_file,type_param):
    '''

    :param configure_filw: 配置文件路径
    :param step: 'Dis' or 'Param' 指定可以进行单独的模型训练，未指定则进行综合训练
    :param type: 模型类型，只读取在配置文件中设置的模型，如需其他模型，自行添加
    :return:
    '''

    if not os.path.exists(configure_file):
        logging.error('config file not exists')
        sys.exit()
        # writer = None
    with open(configure_file, "r") as stream:
        try:
            configs = yaml.safe_load(stream)
            print(configs)
            logging.info("loading config file successfully!!")
        except yaml.YAMLError as exc:
            logging.error(exc)

    configs_param = configs['Param']
    # 加载Param配置
    configs_param['normal'] = configs['Param']['Net']['normal']
    configs_param['device'] = configs['device']
    param=type_param
    configs_param['type']=type_param
    if param is not None:
        # Create output folder
        if param == 'VAE':
            values = configs_param['Net']['VAE_Net']
        elif param in ['ODE_RNN']:
            values = configs_param['Net']['ODE_RNN_Net']
        elif param in ['RNN']:
            values = configs_param['Net']['RNN_Net']
    if param is None or param =='MLP':
        values=configs_param['Net']['MLP_Net']
    configs_param['Net'].update(values)
    del configs_param['Net']['ODE_RNN_Net']
    del configs_param['Net']['RNN_Net']
    del configs_param['Net']['MLP_Net']
    del configs_param['Net']['VAE_Net']

    return configs_param




class EarlyStopping:
    def __init__(self, save_path,patience=7, verbose=False, delta=0):

        self.save_path = save_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        path = os.path.join(self.save_path, 'best_loss.pth')
        entire_path = os.path.join(self.save_path, 'entire.pth')
        torch.save(model.state_dict(), path)	# 这里会存储迄今最优模型的参数
        torch.save(model, entire_path)
        self.val_loss_min = val_loss



def get_logger(logpath, filepath, package_files=[],
               displaying=True, saving=True, debug=False):
    logger = logging.getLogger()
    logger.handlers=[]
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logger.setLevel(level)
    if saving:
        info_file_handler = logging.FileHandler(logpath, mode='w')
        info_file_handler.setLevel(level)
        logger.addHandler(info_file_handler)
    if displaying:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        logger.addHandler(console_handler)
    logger.info(filepath)

    for f in package_files:
        logger.info(f)
        with open(f, 'r') as package_f:
            logger.info(package_f.read())

    return logger



def split_data(data,time,param,train_fraq=0.6):
    n_samples = data.shape[0]
    data_train = data[:int(n_samples * train_fraq)]
    data_test = data[int(n_samples * train_fraq):]


    train_time_steps = time[:int(n_samples * train_fraq)]
    test_time_steps = time[int(n_samples * train_fraq):]

    train_param=param[:int(n_samples * train_fraq)]
    test_param = param[int(n_samples * train_fraq):]

    train_data_dict = {
        'data': data_train ,
        'params': train_param,
        'time': train_time_steps}

    test_data_dict = {
        'data': data_test,
        'params': test_param,
        'time': test_time_steps}

    return train_data_dict,test_data_dict






def get_ckpt_model(ckpt_path, model, optimizer,device,train=True):
    if not os.path.exists(ckpt_path):
        raise Exception("Checkpoint " + ckpt_path + " does not exist.")
    # Load checkpoint.
    load_state = torch.load(ckpt_path,map_location=device)

    if train:
         optimizer.load_state_dict(load_state["optimizer"])
    current_epoch = load_state["epoch"]

    best_loss = load_state["loss"]

    state_dict = load_state['state_dict']
    #
    # print("Model's state_dict:")
    # for param_tensor in state_dict:
    #     print(param_tensor, "\t", state_dict[param_tensor].size())

    model_dict = model.state_dict()

    # 1. filter out unnecessary keys
    state_dict = {k: v for k, v in state_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(state_dict)
    # 3. load the new state dict
    #model.load_state_dict(torch.load(configs['VAE']['Net']['checkpoints'], map_location=device))
    model.load_state_dict(state_dict)
    model.to(device)
    return current_epoch,best_loss


def mse_loss(output, target):
    return F.mse_loss(output, target, reduction='mean')







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
        train_size = len(self.data) - n_test
        # calculate the split
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
    

