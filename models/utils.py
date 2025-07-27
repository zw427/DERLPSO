import functools
import logging
import os
import sys

import yaml
import os

import numpy as np
import torch

import os
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

class StringConcatinator(yaml.YAMLObject):
    yaml_loader = yaml.SafeLoader
    yaml_tag = '!join'
    @classmethod
    def from_yaml(cls, loader, node):
        return functools.reduce(lambda a, b: a.value + b.value, node.value)


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




import pandas as pd

def makedirs(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

def checkDataExists(*args):
    for i in args:
        if not os.path.exists(i):
            return False
    return True



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






def flatten(x, dim):
    return x.reshape(x.size()[:dim] + (-1,))



#resample
def sample_standard_gaussian(mu, sigma):
    device = get_device(mu)

    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(device), torch.Tensor([1.]).to(device))

    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma + mu


def split_train_test(data, train_fraq=0.8):
    n_samples = data.size(0)
    data_train = data[:int(n_samples * train_fraq)]
    data_test = data[int(n_samples * train_fraq):]
    return data_train, data_test

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

def split_train_test_data_and_time(data, time_steps, train_fraq=0.6):
    n_samples = data.size(0)
    data_train = data[:int(n_samples * train_fraq)]
    data_test = data[int(n_samples * train_fraq):]

    assert (len(time_steps.size()) == 2)
    train_time_steps = time_steps[:, :int(n_samples * train_fraq)]
    test_time_steps = time_steps[:, int(n_samples * train_fraq):]

    return data_train, data_test, train_time_steps, test_time_steps



def get_data_dict(observed_data,observed_tp,params):
    batch_dict = {
                "observed_data": None,
                "observed_tp": None,
                "param_to_predict": None
            }


    batch_dict["observed_data"] = observed_data
    batch_dict["observed_tp"] = observed_tp
    #param
    batch_dict["param_to_predict"]=params
    return batch_dict





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





def csv_file_read(filename):

    head_row = pd.read_csv(filename, nrows=0)
    head_row_list = list(head_row)

    csv_result = pd.read_csv(filename, usecols=head_row_list)
    row_list = csv_result.values.tolist()

    return row_list



def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert (start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points,dtype=torch.float64)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(start[i], end[i], n_points,dtype=torch.float64)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res





def kld_loss(mu, logvar):
    # computes the kld divergence for a VAE model between the normal prior
    # and the posterior q(z|x) of the VAE
    # The KLD is computed for each sample and then averaged over the batch
    loss_per_batch_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim = 1)
    loss_per_batch = torch.mean(loss_per_batch_sample, dim = 0)
    return loss_per_batch

def l1_loss(output, target):
    # The reconstruction loss, between the real values and the prediction.
    # Each output dimension is considered independent, conditioned
    # on the latent representation p(x_i|z). Where i traverses the output dimension.
    # Hence the likelihood is  L = Mult_i p(x_i|z)
    # And the log likelihood is log L = Sum_i log p(x_i|z)
    # We then take the average of those log likelihoods for the batch
    # Notice we use L1 for now not L2
    # https://stats.stackexchange.com/questions/323568/help-understanding-reconstruction-loss-in-variational-autoencoder
    # loss = torch.mean(torch.sum(torch.abs(output.flatten(start_dim=1) -
    #                             target.flatten(start_dim=1)), axis=1),axis=0)
    # mean per output dimension
    # loss = torch.mean(torch.abs(output - target))
    return F.l1_loss(output, target, reduction='mean')

def mse_loss(output, target):
    return F.mse_loss(output, target, reduction='mean')




def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x):
    return torch.from_numpy(x)


def save_checkpoint(state, save, epoch):
    if not os.path.exists(save):
        os.makedirs(save)
    filename = os.path.join(save, 'checkpt-%04d.pth' % epoch)
    torch.save(state, filename)




def build_modules( hidden_dims, nonlinearity,normal=False):
    nonlinearity = get_actFunc(nonlinearity)
    modules = []
    input_dim = hidden_dims[0]
    output_dim=hidden_dims[-1]

    for h_dim in hidden_dims[1:-1]:
        modules.append(
            nn.Sequential(
                nn.Linear(input_dim, h_dim,bias=False),
                # nn.BatchNorm1d(h_dim),
                nonlinearity,

            )
        )
        input_dim = h_dim
    modules.append(nn.Sequential(nn.Linear(input_dim,output_dim,bias=False)))
    if normal:
        modules.append(nn.Sigmoid())

    return nn.Sequential(*modules)


def get_actFunc(nonlinearity):
    if nonlinearity == 'relu':
        nonlinearity = nn.ReLU()
    elif nonlinearity == 'leaky':
        nonlinearity = nn.LeakyReLU()
    elif nonlinearity == 'tanh':
        nonlinearity = nn.Tanh()
    elif nonlinearity == 'elu':
        nonlinearity = nn.ELU()
    else:
        raise ('Unknown nonlinearity. Accepting relu or tanh only.')
    return nonlinearity


def create_net(n_inputs, n_outputs, n_layers=1,
               n_units=100, nonlinear=nn.Tanh,drop=False):
    layers = [nn.Linear(n_inputs, n_units)]
    for i in range(n_layers):
        layers.append(nonlinear())
        layers.append(nn.Linear(n_units, n_units))
        if (not i == n_layers) and drop:
            layers.append(nn.Dropout(0.8))

    layers.append(nonlinear())
    layers.append(nn.Linear(n_units, n_outputs))
    return nn.Sequential(*layers)

def split_last_dim(data,avg=True,pos=None):
    last_dim = data.size()[-1]

    if avg:
            last_dim = last_dim // 2

            if len(data.size()) == 3:
                res = data[:, :, :last_dim], data[:, :, last_dim:]

            if len(data.size()) == 2:
                res = data[:, :last_dim], data[:, last_dim:]
    else:
        if len(data.size()) == 3:
            res = data[:, :, :pos], data[:, :, pos:]

        if len(data.size()) == 2:
            res = data[:, :pos], data[:, pos:]
    return res


def init_network_weights(net, std=0.1):
    if isinstance(net,list):
        for i in range(len(net)):
          for m in net[i].modules():
             if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=std)
                nn.init.constant_(m.bias, val=0.)
    else:
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0., std=std)
                nn.init.constant_(m.bias, val=0.)



def get_device(tensor):
    device = torch.device("cpu")
    if tensor.is_cuda:

        device = torch.device("cuda:"+str(tensor.get_device()))
    return device



def linspace_vector(start, end, n_points):
    # start is either one value or a vector
    size = np.prod(start.size())

    assert (start.size() == end.size())
    if size == 1:
        # start and end are 1d-tensors
        res = torch.linspace(start, end, n_points,dtype=torch.float64)
    else:
        # start and end are vectors
        res = torch.Tensor()
        for i in range(0, start.size(0)):
            res = torch.cat((res,
                             torch.linspace(start[i], end[i], n_points,dtype=torch.float64)), 0)
        res = torch.t(res.reshape(start.size(0), n_points))
    return res

# #反归一化
def postprocess_labels(labels_train_predict,scale):
    labels_train_predict = labels_train_predict * scale['mult'] + scale['shift']
  #  labels_test_predict  = labels_test_predict * scale['mult'] + scale['shift']
    return labels_train_predict




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
    














import copy

from lmfit import minimize, Parameters
from scipy.integrate import ode


#装载数据

def prepare_data(dataset,b_train=1):

    if torch.cuda.is_available():
        train_dataset = torch.utils.data.DataLoader(dataset, batch_size=b_train, shuffle=False,pin_memory=False)

    else:
        train_dataset = torch.utils.data.DataLoader(dataset, batch_size=b_train, shuffle=False,pin_memory=False)


    return train_dataset


# -----------------------math method to estimate param-----------------------------------------#

def est_param(t, data_all, param_init,f):
    all_pred = None

    for i in range(data_all.shape[0]):
        # print('start  {0} ....'.format(i))
        data_true = data_all[i, :, :]
        y0 = data_true[0, :]
        params = Parameters()
        for v in range(len(y0)):
            params.add('x_{}'.format(v), value=y0[v], vary=False)
        for v in range(param_init.size):
            params.add(chr(v+65), value=param_init[v])

        # fit model
        result = minimize(residual, params, args=(t, data_true,f), method='powell', tol=1e-3,
                          options={'xtol': 0.001, 'ftol': 0.001, 'maxiter': 1000, 'disp': False})  # leastsq nelder
        if i%50==0:
            print('complete  {0} param ....'.format(i))
        # get pred param
        pred = []
        for name, item in result.params.items():
            if 'x_' not in name:
                pred.append(item.value)

        pred = np.array(pred)
        all_pred = pred.reshape((-1, param_init.size)) if all_pred is None else np.concatenate((all_pred, pred.reshape((-1, param_init.size))),
                                                                                 axis=0)

    return all_pred


def residual(paras, t, data,f):
    """
    compute the residual between actual data and fitted data
    """
    x0=[]
    param=[]
    for par in paras.items():
        if 'x_' in par[0]:
            x0.append(par[1].value)
        else:
            param.append(par[1].value)
    r = ode(f).set_integrator('dopri5', nsteps=5000)
    r.set_initial_value(x0, t[0]).set_f_params(param)

    x = np.array(x0).reshape((1, -1))

    t_index = 1
    while r.successful() and t_index < len(t):
        #  print('start:',r.t)
        timestep = t[t_index] - t[t_index - 1]
        tmp_data = np.array(r.integrate(r.t + timestep)).reshape((1, -1))
        #  print('after:', r.t)
        x = np.concatenate((x, tmp_data), axis=0)
        t_index = t_index + 1

    if x.shape[0] != len(t):
        x = np.concatenate((x, np.zeros((len(t) - x.shape[0], x.shape[1]))), axis=0)

    return x - data


def get_param(data, est_df):
    a = None
    b = None
    c = None
    assert (not torch.isinf(est_df).any())
    assert (not torch.isnan(est_df).any())
    # data[data ==0] = 0.01

    for i in range(data.shape[1]):
        df0 = est_df[:, i, 0]
        df1 = est_df[:, i, 1]
        df2 = est_df[:, i, 2]
        x0 = data[:, i, 0]
        x1 = data[:, i, 1]
        x2 = data[:, i, 2]

        e_a = df0 / (x1 - x0)
        e_b = (df1 + x1 + x0 * x2) / x0
        e_c = -(df2 - x0 * x1) / x2
        for i in range(data.shape[0]):
            num = e_b[i]
            if torch.isinf(num):
                print(x0[i])
                print(df1[i])
                print(x0[i] * x1[i])

        a = e_a.unsqueeze(1) if a is None else torch.cat((a, e_a.unsqueeze(1)), dim=-1)
        b = e_b.unsqueeze(1) if b is None else torch.cat((b, e_b.unsqueeze(1)), dim=-1)
        c = e_c.unsqueeze(1) if c is None else torch.cat((c, e_c.unsqueeze(1)), dim=-1)
    pred_params = torch.cat((a.unsqueeze(2), b.unsqueeze(2), c.unsqueeze(2)), dim=-1)
    assert (not torch.isnan(pred_params).any())
    assert (not torch.isinf(pred_params).any())
    return torch.mean(pred_params, dim=1)




