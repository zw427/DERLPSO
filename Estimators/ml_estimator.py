
# Standard library
import os
import time as timer
from typing import Optional

# Third-party
import torch
import torch.optim as optim
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torch.distributions import Normal, Independent, kl_divergence

# Local imports
from .estimator import Estimator
from .ml_model_components import *
from .ml_model_components.create_model import create_model
from .utils import *
from equation import Equation



class MLEstimator(Estimator):
    def __init__(self, model_type: str, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path: Optional[str] = None):
        """
        Initialize ML Model with configuration and model creation.
        
        Args:
            model_type: Type of model ('MLP', 'RNN', 'ODE_RNN', 'VAE', etc)
            num_of_param: Number of parameters to predict
            dim_of_data: Dimensionality of input data
            time_points: Number of time points in sequences
            config: Path to configuration file
            model_path: Path to pre-trained model (optional)
        """
        # torch.set_default_dtype(torch.float64)

        # Store basic attributes
        self.model_type = model_type
        self.num_of_param = num_of_param
        self.dim_of_data = dim_of_data
        self.time_points = time_points
        
        # Load configuration
        self.config = load_configure(config, model_type)
        
        # Set device
        self.device = torch.device(self.config['device'])
        
        # Extract configuration parameters
        net_config = self.config['Net']
        self.normal = net_config['normal']
        self.max_epoch = net_config['max_epochs']

        self.learning_rate = float(net_config['learning_rate'])
        self.eval_epoch = net_config['eval_epoch']
        self.train_batch_size = net_config['train_batch_size']
        self.val_batch_size = net_config['eval_batch_size']
        
        # Initialize model
        if model_path:
            self.model = torch.load(model_path, map_location=self.device)
        else:
            self.model = create_model(
                net_config, num_of_param, dim_of_data, 
                time_points, model_type, self.device
            ).float().to(self.device)
    
    def train(self):
        pass

    # log_path = "."
    # if not os.path.exists(os.path.dirname(log_path)):
    #     os.makedirs(os.path.dirname(log_path))
    # logger = get_logger(logpath=log_path)

#     data = read_data(data_filename, time_filename, param_filename)
#     train_with_data(configs_param, base_dir,num_of_param, dim_of_data, data)

# def train_with_data(configs,base_dir,num_of_param,dim_of_data,data):

#     num_train_batches = configs['Net']["train_batch_size"]
#     num_test_batches = configs['Net']["eval_batch_size"]

#     time_points=data["time"].shape[-1]
#     all_data=data['data']
#     all_time = data['time']
#     all_param= data['params']

#     train_dict,test_dict=split_data(all_data,all_time,all_param,train_fraq=0.6)

    
#     train_data_comm = SimpleDataSet(train_dict)
#     test_data_comm = SimpleDataSet(test_dict)

#     # 归一化
#     if configs['normal']:
#         train_data_comm.preprocess_data()
#         test_data_comm.preprocess_data()
#     # 装载数据

#     train_dataset = prepare_data(train_data_comm, b_train=num_train_batches)
#     test_dataset = prepare_data(test_data_comm, b_train=num_test_batches)
#     train(configs, base_dir,train_dataset, test_dataset,num_of_param,dim_of_data,time_points)


# def train(configs,base_dir,train_dataset,val_dataset,num_of_param,dim_of_data,time_points):
#     model== create_model(configs['Net'],num_of_param,dim_of_data,time_points,configs['type'], configs['device'])
#     device = torch.device(configs['device'] if torch.cuda.is_available() else 'cpu')
#     writer = SummaryWriter(log_dir='{}/{}'.format(base_dir,configs['type']))

#     log_path = '{}/{}/{}'.format(base_dir, configs['type'], 'train.log')
#     logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
#     optimizer = optim.Adam(model.parameters(), lr=float(configs['Net']['learning_rate']), weight_decay=1e-5)
#     CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
#     if configs["Net"]["load"]:
#         current_epoch, best_loss = get_ckpt_model(
#             '{}/{}/checkpoints/best_loss.ckpt'.format(base_dir, configs['type']), model, optimizer, device)
#         logger.info(
#             'continue training in {} epoch,current loss  is {}'.format(current_epoch, best_loss))

#     current_epoch = 0
#     best_loss = float("inf")

#     model.double()
#     model = model.to(device)

#     save_path = '{}/{}/checkpoints/'.format(base_dir, configs['type'])
#     if not os.path.exists(os.path.dirname(save_path)):
#         os.makedirs(os.path.dirname(save_path))
#     early_stopping = EarlyStopping(save_path,500,True,delta=1e-3)

#     ##### 训练vae模型
#     if configs['type']=='VAE':
#                     latent_dims=configs['Net']['latent_dim']

#                     prior_mu = torch.zeros((latent_dims), dtype=torch.float64).to(device)
#                     prior_sigma = torch.ones((latent_dims), dtype=torch.float64).to(device)

#                     prior = Normal(prior_mu, prior_sigma)

#                     for epoch in range(current_epoch, configs['Net']['max_epochs']):

#                         logger.info('\nEpoch: {}'.format(str(epoch)))
#                         logger.info('\nLearning rate:{}'.format(str(optimizer.param_groups[0]["lr"])))
#                         optimizer.zero_grad()

#                         train_res = {}
#                         train_res["train_param_loss"] = 0.
#                         train_res["train_kl_loss"] = 0.

#                         total_loss=0.

#                         for step, (data, param,u_samples, time) in enumerate(train_dataset):

#                             true_param=param.to(device)
#                             data_encoder = data.detach().to(device)
#                             enc_time = torch.tensor(time[0,:]).to(device)


#                             pred_param,pred_mu,pred_std= model.compute(data_encoder,enc_time)

#                             #loss--klv
#                             fp_distr = Normal(pred_mu,pred_std)
#                             kldiv_z0_gau = kl_divergence(fp_distr, prior)
#                             train_kld = torch.mean(kldiv_z0_gau)


#                             #loss--mse,likelihood
#                             train_param_loss = mse_loss(true_param, pred_param)

#                             total_loss=train_param_loss+total_loss+train_kld

#                             result = {}
#                             result["train_param_loss"] = train_param_loss.item()
#                             result["train_kl_loss"] = train_kld.item()


#                             #cal loss
#                             for key in train_res.keys():
#                                 if key in result.keys():
#                                     var = result[key]
#                                     if isinstance(var, torch.Tensor):
#                                         var = var.detach()
#                                     train_res[key] += var

#                         message = 'Epoch {:04d},[Train |  KLD_loss {:.4f}param_loss {:.4f} '.format(
#                             epoch, train_res["train_kl_loss"], train_res["train_param_loss"])

#                         logger.info(message)
#                         writer.add_scalar('train/kld', train_res["train_kl_loss"], epoch)
#                         writer.add_scalar('train/param_loss', train_res["train_param_loss"], epoch)

#                         #backwards
#                         total_loss.backward()
#                         optimizer.step()

#                         print('=====================train epoch {0} over=============================='.format(epoch))
#                         test_total_loss=0.
#                         if epoch % configs['Net']['eval_epoch'] == 0:
#                             test_epoch=int(epoch/configs['Net']['eval_epoch'])
#                             with torch.no_grad():
#                                 test_res = {}
#                                 test_res["test_kl_loss"] = 0.
#                                 test_res["test_param_loss"] = 0.

#                                 for step, (data, param,u_samples, time) in enumerate(val_dataset):
#                                     #true_param = torch.cat((rho, sigma, param), dim=-1)
#                                     true_param=param.to(device)

#                                     data_encoder = data.to(device)
#                                     enc_time = torch.tensor(time[0,:]).to(device)
#                                     pred_param,pred_mu,pred_std = model.compute(data_encoder,enc_time)

#                                     fp_distr = Normal(pred_mu, pred_std)

#                                     kldiv_z0_gau = kl_divergence(fp_distr, prior)

#                                     test_kld = torch.mean(kldiv_z0_gau)

#                                     test_param_loss = mse_loss(true_param, pred_param)


#                                     test_total_loss = test_total_loss + test_param_loss+test_kld

#                                     result = {}
#                                     result["test_kl_loss"] = test_kld.item()
#                                     result["test_param_loss"] = test_param_loss.item()

#                                     for key in test_res.keys():
#                                         if key in result.keys():
#                                             var = result[key]
#                                             if isinstance(var, torch.Tensor):
#                                                 var = var.detach()
#                                             test_res[key] += var

#                                 message = 'Epoch {:04d},[Test | KLD_loss {:.4f} |  param_loss {:.4f} '.format(
#                                     test_epoch,  test_res["test_kl_loss"],test_res["test_param_loss"])

#                                 logger.info(message)
#                                 writer.add_scalar('test/kld', test_res["test_kl_loss"], test_epoch)
#                                 writer.add_scalar('test/param_loss', test_res["test_param_loss"], test_epoch)

#                                 early_stopping((test_res["test_param_loss"]+test_res["test_kl_loss"]), model)
#                                 # 达到早停止条件时，early_stop会被置为True
#                                 if early_stopping.early_stop:
#                                     print("Early stopping")
#                                     writer.close()
#                                     return  # 跳出迭代，结束训练

#                         CosineLR.step()
#                     writer.close()
#     else:
#         for epoch in range(current_epoch, configs['Net']['max_epochs']):
#             logger.info('\nEpoch: {}'.format(str(epoch)))
#             logger.info('\nLearning rate:{}'.format(str(optimizer.param_groups[0]["lr"])))
#             optimizer.zero_grad()

#             train_res = {}
#             # train_res["loss"] = 0.
#             # train_res["train_param_likelihood"] = 0.
#             train_res["train_param_loss"] = 0.

#             total_loss = 0.


#             for step, (data, param,u_samples, time) in enumerate(train_dataset):

#                     true_param=param.to(device)

#                     data_encoder = data.detach().to(device)
#                     enc_time = torch.tensor(time[0,:]).to(device)


#                     pred_param = model.compute(data_encoder,enc_time)

#                     train_param_loss = mse_loss(true_param, pred_param)
                  
#                     total_loss = train_param_loss + total_loss

#                     result = {}

#                     result["train_param_loss"] = train_param_loss.item()

#                     # cal loss
#                     for key in train_res.keys():
#                         if key in result.keys():
#                             var = result[key]
#                             if isinstance(var, torch.Tensor):
#                                 var = var.detach()
#                             train_res[key] += var

#             message = 'Epoch {:04d},[Train | mse_loss {:.6f} '.format(epoch,train_res["train_param_loss"])

#             logger.info(message)
#             writer.add_scalar('train/param_loss', train_res["train_param_loss"], epoch)

#             # backwards
#             total_loss.backward()
#             optimizer.step()

#             print('=====================train epoch {0} over=============================='.format(epoch))
#             test_total_loss = 0.
#             if epoch % configs['Net']['eval_epoch'] == 0:
#                 model.eval()
#                 test_epoch = int(epoch / configs['Net']['eval_epoch'])
#                 with torch.no_grad():
#                     test_res = {}
#                     test_res["test_param_loss"] = 0.
#                     for step, (data,  param, u_samples,time) in enumerate(val_dataset):


                     
#                         true_param=param.to(device)

#                         data_encoder =data.detach().to(device)
#                         enc_time = torch.tensor(time[0,:]).to(device)


#                         pred_param= model.compute(data_encoder,enc_time)

#                         test_param_loss = mse_loss(true_param, pred_param)
#                         test_total_loss = test_total_loss + test_param_loss

#                         result = {}
#                         result["test_param_loss"] = test_param_loss.item()

#                         for key in test_res.keys():
#                             if key in result.keys():
#                                 var = result[key]
#                                 if isinstance(var, torch.Tensor):
#                                     var = var.detach()
#                                 test_res[key] += var

#                     message = 'Epoch {:04d},[Test | param_loss {:.6f}'.format(test_epoch, test_res["test_param_loss"])

#                     logger.info(message)

                  
#                     writer.add_scalar('test/param_loss', test_res["test_param_loss"], test_epoch)


#                     # if (test_res["test_param_loss"]) < best_loss:
#                     #     logger.info('current eval loss:{}'.format(test_res["test_param_loss"]))
#                     #     logger.info('best eval loss:{}'.format(best_loss))
#                     #     best_loss = test_res["test_param_loss"]
#                     #
#                     #     torch.save({
#                     #         'epoch': epoch,
#                     #         'loss': best_loss,
#                     #         'state_dict': model.state_dict(),
#                     #         'optimizer': optimizer.state_dict(),
#                     #     },save_path)
#                     #     torch.save(model,model_path)
#                     #     #----------------------------------------------------------------------#
#                     early_stopping((test_res["test_param_loss"]), model)
#                     # 达到早停止条件时，early_stop会被置为True
#                     if early_stopping.early_stop:
#                         print("Early stopping")
#                         writer.close()
#                         return  # 跳出迭代，结束训练
#             CosineLR.step()
#         writer.close()






    def predict_one(self, data: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        Predict parameters for a single data sample.
        
        Args:
            data: Input data tensor of shape (batch_size, sequence_length, features)
            time: Time points tensor of shape (sequence_length,) or (batch_size, sequence_length)

        Returns:
            Predicted parameters tensor
        """
        self.model.eval()
        return self.model.compute(data.to(self.device), time.to(self.device))

    def predict(self, data, time):
        """
        Predict parameters for test data and optionally calculate metrics.
        
        Args:
            data_file: Path to data file
            time_file: Path to time file
            param_file: Path to parameter file (optional, for evaluation)
            save_dir: Directory to save results (optional)
            
        Returns:
            If param_file provided: (predictions, truth, mse_losses)
            If param_file not provided: predictions
        """
        # Load and prepare data
        dataset = SimpleDataSet({'data': data, 'time': time})

        if self.normal:
            dataset.preprocess_data()
        
        dataset = torch.utils.data.DataLoader(dataset, batch_size=self.val_batch_size)

        self.model.eval()
        with torch.no_grad():
            prediction = []
            for data, _, _, time in dataset:

                predicted_param = self.predict_one(data, time[0, :])

                # VAE for some reason
                if isinstance(predicted_param, tuple):
                    predicted_param = predicted_param[0]
                
                prediction.append(predicted_param)

        # Concatenate results
        return torch.cat(prediction, dim=0).cpu().numpy()
        

class MLP(MLEstimator):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('MLP', num_of_param, dim_of_data, 
                         time_points, config, model_path)


class RNN(MLEstimator):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('RNN', num_of_param, dim_of_data, 
                         time_points, config, model_path)


class ODE_RNN(MLEstimator):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('ODE_RNN', num_of_param, dim_of_data, 
                         time_points, config, model_path)


class VAE(MLEstimator):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('VAE', num_of_param, dim_of_data, 
                         time_points, config, model_path)
