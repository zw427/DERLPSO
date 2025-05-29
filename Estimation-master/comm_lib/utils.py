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


import os
import logging

import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F

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
