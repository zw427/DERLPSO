###########################
# Latent ODEs for Irregularly-Sampled Time Series
# Author: Yulia Rubanova
###########################

import torch
import torch.nn as nn
# import gc
from torch.distributions import Normal, Independent

from comm_lib.utils import get_device


def get_gaussian_likelihood(truth, pred_y, std=0.01):
    batch_size = truth.shape[0]
    std = torch.tensor(std).to(get_device(truth))

    pred_flat = pred_y.reshape(batch_size, -1)
    data_flat = truth.reshape(batch_size, -1)

    res = gaussian_log_likelihood(pred_flat, data_flat, std)

    return res
def gaussian_log_likelihood(mu_2d, data_2d, obsrv_std, indices=None,sigma=None,rho=None):
    n_data_points = mu_2d.size()[-1]
    if n_data_points > 0:
        # loc,scale 均值 标准差
        gaussian = Independent(Normal(loc=mu_2d, scale=obsrv_std.repeat(n_data_points)), 1)
        # Returns the log of the probability density/mass function evaluated at`value`.
        log_prob = gaussian.log_prob(data_2d)
        log_prob = log_prob / n_data_points
    else:
        log_prob = torch.zeros([1]).to(get_device(data_2d)).squeeze()
    return log_prob



def compute_likelihood(mu, data, likelihood_func):
    # Compute the likelihood per patient and per attribute so that we don't priorize patients with more measurements
    n_traj, n_timepoints, n_dims = data.size()

    res = []
    for k in range(n_traj):
        #print('k',k)
        for j in range(n_dims):
        #    # print('j',j)

            log_prob = likelihood_func(mu[k,:,j], data[k,:,j], indices=(k, j))
            res.append(log_prob)

    res = torch.stack(res, 0).to(get_device(data))
    res = res.reshape(( n_traj, n_dims))
    res = torch.mean(res, -1)  # !!!!!!!!!!! changed from sum to mean
    return res


def gaussian_log_density(mu, data, obsrv_std,sigma=None,rho=None):

    n_traj, n_timepoints, n_dims = mu.size()

    assert (data.size()[-1] == n_dims)

    # Shape after permutation: [n_traj, n_traj_samples, n_timepoints, n_dims]
    if sigma is None:
         mu_flat = mu.reshape( n_traj, n_timepoints * n_dims)
         n_traj, n_timepoints, n_dims = data.size()
         data_flat = data.reshape(n_traj, n_timepoints * n_dims)

         res = gaussian_log_likelihood(mu_flat, data_flat, obsrv_std)

         return res
    else:
            # Compute the likelihood per patient so that we don't priorize patients with more measurements
            func = lambda mu, data, indices: gaussian_log_likelihood(mu, data, obsrv_std=obsrv_std, indices=indices,sigma=sigma,rho=rho)
            res = compute_likelihood(mu, data, func)
            return res


def mse(mu, data):
    n_data_points = mu.size()[-1]

    if n_data_points > 0:
        mse = nn.MSELoss()(mu, data)
    else:
        mse = torch.zeros([1]).to(get_device(data)).squeeze()
    return mse


def compute_mse(mu, data):

    n_traj, n_timepoints, n_dims = mu.size()
    assert (data.size()[-1] == n_dims)

    mu_flat = mu.reshape( n_traj, n_timepoints * n_dims)
    n_traj, n_timepoints, n_dims = data.size()
    data_flat = data.reshape(n_traj, n_timepoints * n_dims)
    res = mse(mu_flat, data_flat)
    return res





