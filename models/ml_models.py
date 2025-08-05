from .base_model import BaseModel
from typing import List
from equations import Equation
from torch.distributions import Normal, Independent
from .ml_model_components.create_model import create_model

class MLModel(BaseModel):
    def __init__(self):
        raise NotImplementedError("MLModel is an abstract base class and cannot be instantiated directly.")
    
    def train(self):
        pass

    def predict(self):
        pass 



class MLP(MLModel):
    def __init__(self, equation, init_data, param_mu, param_sigma):
        super().__init__(equation, init_data, param_mu, param_sigma)
        # Initialize MLP specific attributes here
        self.hidden_layers = []  # Placeholder for hidden layers
        self.output_layer = None  # Placeholder for output layer

class RNN(MLModel):
    def __init__(self, equation, init_data, param_mu, param_sigma):
        super().__init__(equation, init_data, param_mu, param_sigma)
        # Initialize RNN specific attributes here
        self.hidden_state = None  # Placeholder for hidden state

class ODE_RNN(MLModel):
    def __init__(self, equation, init_data, param_mu, param_sigma):
        super().__init__(equation, init_data, param_mu, param_sigma)
        # Initialize ODE_RNN specific attributes here
        self.ode_solver = None  # Placeholder for ODE solver

class VAE(MLModel):
    def __init__(self, equation, init_data, param_mu, param_sigma):
        super().__init__(equation, init_data, param_mu, param_sigma)
        # Initialize VAE specific attributes here
        self.encoder = None  # Placeholder for encoder
        self.decoder = None  # Placeholder for decoder
        self.latent_dim = None  # Placeholder for latent dimension size
    

import sys
from pathlib import Path

import time as timer
from torch.distributions import kl_divergence
from .utils import *

import pandas as pd
import  matplotlib.pyplot as plt
import torch.optim as optim

from .ml_model_components import *

import torch
from torch.utils.tensorboard import SummaryWriter




def train_model(configure_file,base_dir, dim_of_data, num_of_param, data_filename, time_filename, param_filename, ode_func,model_type):
    if isinstance(ode_func, str):
        ode_func = __import__(ode_func, globals(), locals(), ['get_data', 'f'])

    configs_param = load_configure(configure_file,model_type)
    log_path = '{}/{}/{}'.format(base_dir,configs_param['type'], "train_model.log")
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

    data = read_data(data_filename, time_filename, param_filename)
    train_with_data(configs_param, base_dir,num_of_param, dim_of_data, data)


def read_data(data_filename,time_filename,param_filename=None):
    time = pd.read_csv(time_filename, delimiter=',').values
    time_points = time.shape[-1]
    data = pd.read_csv(data_filename, delimiter=',').values
    data = data.reshape(data.shape[0], time_points, -1)
    data_dict = {
        'data': data,
        # 'u_samples': None,
        # 'params': params,
        'time': time}
    if param_filename is not None:
        params = pd.read_csv(param_filename, delimiter=',').values
        params=params.reshape(data.shape[0],-1)
        data_dict['params']=params

    return data_dict


def train_with_data(configs,base_dir,num_of_param,dim_of_data,data):

    num_train_batches = configs['Net']["train_batch_size"]
    num_test_batches = configs['Net']["eval_batch_size"]

    time_points=data["time"].shape[-1]
    all_data=data['data']
    all_time = data['time']
    all_param= data['params']
    train_dict,test_dict=split_data(all_data,all_time,all_param,train_fraq=0.6)
    train_data_comm = SimpleDataSet(train_dict)
    test_data_comm = SimpleDataSet(test_dict)

    # 归一化
    if configs['normal']:
        train_data_comm.preprocess_data()
        test_data_comm.preprocess_data()
    # 装载数据
    train_dataset = prepare_data(train_data_comm, b_train=num_train_batches)
    test_dataset = prepare_data(test_data_comm, b_train=num_test_batches)
    train(configs, base_dir,train_dataset, test_dataset,num_of_param,dim_of_data,time_points)


def train(configs,base_dir,train_dataset,val_dataset,num_of_param,dim_of_data,time_points):
    model== create_model(configs['Net'],num_of_param,dim_of_data,time_points,configs['type'], configs['device'])
    device = torch.device(configs['device'] if torch.cuda.is_available() else 'cpu')
    writer = SummaryWriter(log_dir='{}/{}'.format(base_dir,configs['type']))

    log_path = '{}/{}/{}'.format(base_dir, configs['type'], 'train.log')
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
    optimizer = optim.Adam(model.parameters(), lr=float(configs['Net']['learning_rate']), weight_decay=1e-5)
    CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    if configs["Net"]["load"]:
        current_epoch, best_loss = get_ckpt_model(
            '{}/{}/checkpoints/best_loss.ckpt'.format(base_dir, configs['type']), model, optimizer, device)
        logger.info(
            'continue training in {} epoch,current loss  is {}'.format(current_epoch, best_loss))

    current_epoch = 0
    best_loss = float("inf")

    model.double()
    model = model.to(device)

    save_path = '{}/{}/checkpoints/'.format(base_dir, configs['type'])
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    early_stopping = EarlyStopping(save_path,500,True,delta=1e-3)

    ##### 训练vae模型
    if configs['type']=='VAE':
                    latent_dims=configs['Net']['latent_dim']

                    prior_mu = torch.zeros((latent_dims), dtype=torch.float64).to(device)
                    prior_sigma = torch.ones((latent_dims), dtype=torch.float64).to(device)

                    prior = Normal(prior_mu, prior_sigma)

                    for epoch in range(current_epoch, configs['Net']['max_epochs']):

                        logger.info('\nEpoch: {}'.format(str(epoch)))
                        logger.info('\nLearning rate:{}'.format(str(optimizer.param_groups[0]["lr"])))
                        optimizer.zero_grad()

                        train_res = {}
                        train_res["train_param_loss"] = 0.
                        train_res["train_kl_loss"] = 0.

                        total_loss=0.

                        for step, (data, param,u_samples, time) in enumerate(train_dataset):

                            true_param=param.to(device)
                            data_encoder = data.detach().to(device)
                            enc_time = torch.tensor(time[0,:]).to(device)


                            pred_param,pred_mu,pred_std= model.compute(data_encoder,enc_time)

                            #loss--klv
                            fp_distr = Normal(pred_mu,pred_std)
                            kldiv_z0_gau = kl_divergence(fp_distr, prior)
                            train_kld = torch.mean(kldiv_z0_gau)


                            #loss--mse,likelihood
                            train_param_loss = mse_loss(true_param, pred_param)

                            total_loss=train_param_loss+total_loss+train_kld

                            result = {}
                            result["train_param_loss"] = train_param_loss.item()
                            result["train_kl_loss"] = train_kld.item()


                            #cal loss
                            for key in train_res.keys():
                                if key in result.keys():
                                    var = result[key]
                                    if isinstance(var, torch.Tensor):
                                        var = var.detach()
                                    train_res[key] += var

                        message = 'Epoch {:04d},[Train |  KLD_loss {:.4f}param_loss {:.4f} '.format(
                            epoch, train_res["train_kl_loss"], train_res["train_param_loss"])

                        logger.info(message)
                        writer.add_scalar('train/kld', train_res["train_kl_loss"], epoch)
                        writer.add_scalar('train/param_loss', train_res["train_param_loss"], epoch)

                        #backwards
                        total_loss.backward()
                        optimizer.step()

                        print('=====================train epoch {0} over=============================='.format(epoch))
                        test_total_loss=0.
                        if epoch % configs['Net']['eval_epoch'] == 0:
                            test_epoch=int(epoch/configs['Net']['eval_epoch'])
                            with torch.no_grad():
                                test_res = {}
                                test_res["test_kl_loss"] = 0.
                                test_res["test_param_loss"] = 0.

                                for step, (data, param,u_samples, time) in enumerate(val_dataset):
                                    #true_param = torch.cat((rho, sigma, param), dim=-1)
                                    true_param=param.to(device)

                                    data_encoder = data.to(device)
                                    enc_time = torch.tensor(time[0,:]).to(device)
                                    pred_param,pred_mu,pred_std = model.compute(data_encoder,enc_time)

                                    fp_distr = Normal(pred_mu, pred_std)

                                    kldiv_z0_gau = kl_divergence(fp_distr, prior)

                                    test_kld = torch.mean(kldiv_z0_gau)

                                    test_param_loss = mse_loss(true_param, pred_param)


                                    test_total_loss = test_total_loss + test_param_loss+test_kld

                                    result = {}
                                    result["test_kl_loss"] = test_kld.item()
                                    result["test_param_loss"] = test_param_loss.item()

                                    for key in test_res.keys():
                                        if key in result.keys():
                                            var = result[key]
                                            if isinstance(var, torch.Tensor):
                                                var = var.detach()
                                            test_res[key] += var

                                message = 'Epoch {:04d},[Test | KLD_loss {:.4f} |  param_loss {:.4f} '.format(
                                    test_epoch,  test_res["test_kl_loss"],test_res["test_param_loss"])

                                logger.info(message)
                                writer.add_scalar('test/kld', test_res["test_kl_loss"], test_epoch)
                                writer.add_scalar('test/param_loss', test_res["test_param_loss"], test_epoch)

                                early_stopping((test_res["test_param_loss"]+test_res["test_kl_loss"]), model)
                                # 达到早停止条件时，early_stop会被置为True
                                if early_stopping.early_stop:
                                    print("Early stopping")
                                    writer.close()
                                    return  # 跳出迭代，结束训练

                        CosineLR.step()
                    writer.close()
    else:
        for epoch in range(current_epoch, configs['Net']['max_epochs']):
            logger.info('\nEpoch: {}'.format(str(epoch)))
            logger.info('\nLearning rate:{}'.format(str(optimizer.param_groups[0]["lr"])))
            optimizer.zero_grad()

            train_res = {}
            # train_res["loss"] = 0.
            # train_res["train_param_likelihood"] = 0.
            train_res["train_param_loss"] = 0.

            total_loss = 0.


            for step, (data, param,u_samples, time) in enumerate(train_dataset):

                    true_param=param.to(device)

                    data_encoder = data.detach().to(device)
                    enc_time = torch.tensor(time[0,:]).to(device)


                    pred_param = model.compute(data_encoder,enc_time)

                    train_param_loss = mse_loss(true_param, pred_param)
                  
                    total_loss = train_param_loss + total_loss

                    result = {}

                    result["train_param_loss"] = train_param_loss.item()

                    # cal loss
                    for key in train_res.keys():
                        if key in result.keys():
                            var = result[key]
                            if isinstance(var, torch.Tensor):
                                var = var.detach()
                            train_res[key] += var

            message = 'Epoch {:04d},[Train | mse_loss {:.6f} '.format(epoch,train_res["train_param_loss"])

            logger.info(message)
            writer.add_scalar('train/param_loss', train_res["train_param_loss"], epoch)

            # backwards
            total_loss.backward()
            optimizer.step()

            print('=====================train epoch {0} over=============================='.format(epoch))
            test_total_loss = 0.
            if epoch % configs['Net']['eval_epoch'] == 0:
                model.eval()
                test_epoch = int(epoch / configs['Net']['eval_epoch'])
                with torch.no_grad():
                    test_res = {}
                    test_res["test_param_loss"] = 0.
                    for step, (data,  param, u_samples,time) in enumerate(val_dataset):


                     
                        true_param=param.to(device)

                        data_encoder =data.detach().to(device)
                        enc_time = torch.tensor(time[0,:]).to(device)


                        pred_param= model.compute(data_encoder,enc_time)

                        test_param_loss = mse_loss(true_param, pred_param)
                        test_total_loss = test_total_loss + test_param_loss

                        result = {}
                        result["test_param_loss"] = test_param_loss.item()

                        for key in test_res.keys():
                            if key in result.keys():
                                var = result[key]
                                if isinstance(var, torch.Tensor):
                                    var = var.detach()
                                test_res[key] += var

                    message = 'Epoch {:04d},[Test | param_loss {:.6f}'.format(test_epoch, test_res["test_param_loss"])

                    logger.info(message)

                  
                    writer.add_scalar('test/param_loss', test_res["test_param_loss"], test_epoch)


                    # if (test_res["test_param_loss"]) < best_loss:
                    #     logger.info('current eval loss:{}'.format(test_res["test_param_loss"]))
                    #     logger.info('best eval loss:{}'.format(best_loss))
                    #     best_loss = test_res["test_param_loss"]
                    #
                    #     torch.save({
                    #         'epoch': epoch,
                    #         'loss': best_loss,
                    #         'state_dict': model.state_dict(),
                    #         'optimizer': optimizer.state_dict(),
                    #     },save_path)
                    #     torch.save(model,model_path)
                    #     #----------------------------------------------------------------------#
                    early_stopping((test_res["test_param_loss"]), model)
                    # 达到早停止条件时，early_stop会被置为True
                    if early_stopping.early_stop:
                        print("Early stopping")
                        writer.close()
                        return  # 跳出迭代，结束训练
            CosineLR.step()
        writer.close()


# inference
def predict(model_path,data_filename,time_filename,normal=True,save_dir=None):
    test_dataset=read_data(data_filename,time_filename) 
    if save_dir is None:
        save_dir=os.path.dirname(data_filename)
    model = torch.load(model_path)
    num_val_batches = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    test_data = SimpleDataSet(dataset=test_dataset)

    # 归一化
    if normal:
        test_data.preprocess_data()
    dataset = prepare_data(test_data, b_train=num_val_batches)
    all_param_model = None

    with torch.no_grad():


            for step, (data,params,u_samples, time) in enumerate(dataset):

                result = {}
                data_encoder = data.to(device)
                enc_time = time[0,:].to(device)


                start = timer.time()
                pred_param= model.compute(data_encoder, enc_time)
                if isinstance(pred_param,tuple):
                    pred_param=pred_param[0]
                end = timer.time()
                result['test_model_time'] = (end - start)



                all_param_model = pred_param if all_param_model is None else torch.cat(
                    (all_param_model, pred_param), dim=0)

            data_dir = save_dir + '/pred_param.xlsx'
            df = pd.DataFrame(to_np(all_param_model))
            write = pd.ExcelWriter(data_dir)
            df.to_excel(write, 'sheet_1', float_format='%.3f')
            write.save()
            print("use model predict successfully")


# inference
def infer_batches(model_paths,data_filename,time_filename,param_filename,ode_func,normal):
    if isinstance(ode_func, str):
        ode_func = __import__(ode_func, globals(), locals(), ['get_data', 'f'])

    test_dataset = read_data(data_filename, time_filename, param_filename)

    test_data = SimpleDataSet(dataset=test_dataset)
    # 归一化
    if normal:
        scale=test_data.preprocess_data()
    all_param_math = None

    plt.figure(1)
    color=['r','b','g','y']


    num_val_batches = 1024
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    dataset = prepare_data(test_data, b_train=num_val_batches)

    models=[]

    model_mlp = torch.load(model_paths, map_location=device)
    # model_rnn = torch.load(model_paths[1], map_location=device)
    # model_ode = torch.load(model_paths[2], map_location=device)
    # model_vae = torch.load(model_paths[3], map_location=device)

    all_param_mlp = None
    all_param_rnn= None
    all_param_ode = None
    all_param_vae = None
    all_param = None
    mlp_time=0.
    rnn_time=0.
    ode_time=0.
    vae_time=0.
    math_time=0.
    with torch.no_grad():

            for step, (data,params,u_samples,time) in enumerate(dataset):

                result = {}
                data_encoder = data.to(device)
                enc_time = time[0, :].to(device)


                s_time=timer.time()
                pred_param_mlp = model_mlp.compute(data_encoder, enc_time)
                e_time = timer.time()
                mlp_time=mlp_time+(e_time-s_time)
                s_time = timer.time()
                # pred_param_rnn = model_rnn.compute(data_encoder, enc_time)
                # e_time = timer.time()
                # rnn_time=rnn_time+(e_time-s_time)
                # s_time = timer.time()
                # pred_param_ode = model_ode.compute(data_encoder, enc_time)
                # e_time = timer.time()
                # ode_time =ode_time+(e_time - s_time)
                # s_time = timer.time()
                # pred_param_vae = model_vae.compute(data_encoder, enc_time)
                # e_time = timer.time()
                # vae_time = vae_time + (e_time - s_time)

                if isinstance(pred_param_mlp, tuple):
                    pred_param_mlp = pred_param_mlp[0]
                # if isinstance(pred_param_rnn, tuple):
                #     pred_param_rnn = pred_param_rnn[0]
                # if isinstance(pred_param_ode, tuple):
                #     pred_param_ode = pred_param_ode[0]
                # if isinstance(pred_param_vae, tuple):
                #     pred_param_vae = pred_param_vae[0]


                # param_init = np.ones((params.shape[-1]))
                # s_time=timer.time()
                # if normal:
                #     data = data * scale['mult'] + scale['shift']
                # pred_param_math = ode_func.est_param(to_np(time[0, :]), to_np(data), param_init)
                # e_time=timer.time()
                # math_time=math_time+(e_time-s_time)

                all_param_mlp = pred_param_mlp if all_param_mlp is None else torch.cat(
                    (all_param_mlp, pred_param_mlp), dim=0)
                # all_param_rnn = pred_param_rnn if all_param_rnn is None else torch.cat(
                #     (all_param_rnn, pred_param_rnn), dim=0)
                # all_param_ode = pred_param_ode if all_param_ode is None else torch.cat(
                #     (all_param_ode, pred_param_ode), dim=0)
                # all_param_vae = pred_param_vae if all_param_vae is None else torch.cat(
                #     (all_param_vae, pred_param_vae), dim=0)
                all_param=params if all_param is None else torch.cat(
                    (all_param, params), dim=0)
                # all_param_math = pred_param_math if all_param_math is None else torch.cat(
                #     (all_param_math,pred_param_math), dim=0)


    
    print('MLP,mse loss:{:.3f},time:{:.3f}'.format( mse_loss(all_param_mlp,all_param),mlp_time))
    # print('RNN,mse loss:{:.3f},time:{:.3f}'.format(mse_loss(all_param_rnn, all_param),rnn_time))
    # print('ODE,mse loss:{:.3f},time:{:.3f}'.format(mse_loss(all_param_ode, all_param),ode_time))
    # print('VAE,mse loss :{:.3f},time:{:.3f}'.format(mse_loss(all_param_vae, all_param),vae_time))
    # print('MATH,mse loss:{:.3f},time:{:.3f}'.format(mse_loss(all_param,to_tensor(all_param_math)).to(device),math_time))

    all_param = to_np(all_param)
    # all_param_mlp = to_np(all_param_mlp)
    # all_param_rnn = to_np(all_param_rnn)
    # all_param_ode = to_np(all_param_ode)
    # all_param_vae = to_np(all_param_vae)

    whole_data=None

    for i in range(all_param.shape[0]):
        a=all_param[i,:]
        b=all_param_mlp[i,:]
        # c = all_param_rnn[i, :]
        # d = all_param_ode[i, :]
        # e = all_param_vae[i, :]
        # f = all_param_math[i, :]
        tmp=a
        tmp=np.vstack((tmp,b))
        # tmp = np.vstack((tmp, c))
        # tmp = np.vstack((tmp, d))
        # tmp = np.vstack((tmp, e))
        # tmp = np.vstack((tmp, f))

        whole_data=tmp if whole_data is None else np.vstack((whole_data,tmp))

    
    data_dir = os.path.join(os.path.dirname(data_filename),'compare_param.xlsx')
    index = ['true', 'mlp'] * all_param.shape[0]

    df = pd.DataFrame(whole_data,index=index)
    write = pd.ExcelWriter(data_dir)
    df.to_excel(write, 'sheet_1', float_format='%.3f')
    write.close()
    




# write output and summary
def write_excel(save_path,all_param,all_param_model):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # export to excel
    four_data = None
    param_pic_dir = save_path + 'param_pic'
    if not os.path.exists(param_pic_dir):
        os.makedirs(param_pic_dir)


    for i in range(0, all_param.shape[0]):
        four_data = all_param[i, :] if four_data is None else np.row_stack(
            (four_data, all_param[i, :]))
        four_data = np.row_stack((four_data, all_param_model[i, :]))

    data_dir = param_pic_dir + '/param.xlsx'
    index = ['true', 'ode'] * all_param.shape[0]
    df = pd.DataFrame(four_data, index=index)
    write = pd.ExcelWriter(data_dir)
    df.to_excel(write, 'sheet_1', float_format='%.3f')
    write.save()

    data_dir = param_pic_dir + '/param_ode.xlsx'
    # index = ['true', 'ode'] * all_param.shape[0]
    df = pd.DataFrame(to_np(all_param_model))
    write = pd.ExcelWriter(data_dir)
    df.to_excel(write, 'sheet_1', float_format='%.3f')
    write.save()

    data_dir = param_pic_dir + '/param_ori.xlsx'
    # index = ['true', 'ode'] * all_param.shape[0]
    df = pd.DataFrame(to_np(all_param))
    write = pd.ExcelWriter(data_dir)
    df.to_excel(write, 'sheet_1', float_format='%.3f')
    write.save()
    write.close()

    # plot true testing values vs. predictions
    n_plot_cols = all_param_model.shape[1]
    #  fig, ax = plt.subplots(n_plot_cols, 1)
    for i in range(n_plot_cols):
        ymin = torch.amin(all_param[:, i])
        ymax = torch.amax(all_param[:, i])
        plt.scatter(all_param[:, i], all_param_model[:, i])
        plt.plot([ymin, ymax], [ymin, ymax], linewidth=3, linestyle='--', color='orange')
        plt.xlabel('true')
        plt.ylabel('predict')
        name = 'param_{}'.format(i)
        # plt.legend(loc='lower right')
        plt.savefig('{}/{}'.format(param_pic_dir, name), dpi=500, bbox_inches='tight')
        plt.clf()





