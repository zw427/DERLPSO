from .estimator import Estimator
from typing import List, Optional
from equations import Equation
from torch.distributions import Normal, Independent
from .ml_model_components.create_model import create_model
import pandas as pd

import os
import time as timer
from torch.distributions import kl_divergence
from .utils import *

import torch.optim as optim
from .ml_model_components import *
import torch
from torch.utils.tensorboard import SummaryWriter


class MLModel(Estimator):
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
        super().__init__()
        
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
        self.load = net_config['load']
        self.learning_rate = float(net_config['learning_rate'])
        self.train_ite = net_config['train_ite']
        self.save_epoch = net_config['save_epoch']
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
            ).to(self.device)
    
    def train(self, ):
        pass




def train_model(configure_file,base_dir, data_filename, time_filename, param_filename):


    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

    data = read_data(data_filename, time_filename, param_filename)
    train_with_data(configs_param, base_dir,num_of_param, dim_of_data, data)

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

    def predict(self, data_file: str, time_file: str, 
                param_file: str = None, save_dir: str = None):
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
        test_data = read_data(data_file, time_file, param_file)
        test_dataset = SimpleDataSet(test_data)

        if self.normal:
            test_dataset.preprocess_data()
        
        dataset = torch.utils.data.DataLoader(test_dataset, batch_size=self.val_batch_size)

        self.model.eval()
        with torch.no_grad():
            start_time = timer.time()
            predictions = []
            ground_truth = []
            mse_losses = []

            for data, params, time in dataset:
                # Get predictions
                predicted_param = self.predict_one(data, time[0, :])

                # Why can this be a tuple...
                if isinstance(predicted_param, tuple):
                    import pdb; pdb.set_trace()
                    predicted_param = predicted_param[0]
                
                predictions.append(predicted_param)

                # Calculate metrics if ground truth is available
                if param_file:
                    ground_truth.append(params)
                    mse_losses.append(mse_loss(predicted_param, params.to(self.device)))

            duration = timer.time() - start_time

        # Concatenate results
        all_predictions = torch.cat(predictions, dim=0)
        
        results = {"predictions": all_predictions}
        
        if param_file:
            all_truth = torch.cat(ground_truth, dim=0)
            all_mse = torch.stack(mse_losses)
            results.update({"truth": all_truth, "mse": all_mse})

        # Save results if requested
        if save_dir:
            self.save_predictions(save_dir, results)
        
        print(f"Prediction completed in {duration:.2f} seconds.")
        print(f"Average MSE is {all_mse.mean().item():.4f}" if param_file else "")
        
        # Return appropriate results
        if param_file:
            return results["predictions"], results["truth"], results["mse"]
        return results["predictions"]

    def save_predictions(self, save_dir: str, results: dict):
        """Helper method to save prediction results to CSV."""

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        predictions = results["predictions"].cpu().numpy()
        
        # Create DataFrame with predictions
        df_data = {"prediction": predictions.tolist()}
        
        # Add truth and MSE if available
        if "truth" in results:
            truth = results["truth"].cpu().numpy()
            mse = results["mse"].cpu().numpy()
            df_data.update({
                "truth": truth.tolist(),
                "mse": mse.tolist()
            })
        
        df = pd.DataFrame(df_data)
        output_path = os.path.join(save_dir, f'{self.model_type}_predictions.csv')
        df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")


class MLP(MLModel):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('MLP', num_of_param, dim_of_data, 
                         time_points, config, model_path)


class RNN(MLModel):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('RNN', num_of_param, dim_of_data, 
                         time_points, config, model_path)


class ODE_RNN(MLModel):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('ODE_RNN', num_of_param, dim_of_data, 
                         time_points, config, model_path)


class VAE(MLModel):
    def __init__(self, num_of_param: int, dim_of_data: int,
                 time_points: int, config: str, model_path=None):
        super().__init__('VAE', num_of_param, dim_of_data, 
                         time_points, config, model_path)
