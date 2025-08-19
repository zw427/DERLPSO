
# Standard library
import os
import time as timer
from typing import Optional

# Third-party
import torch
import torch.optim as optim
import pandas as pd
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
    
    def train(self, train: dict, seed: int = None):

        if seed:
            # Set random seed for reproducibility
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

        num_train_batches = self.train_batch_size
        num_test_batches = self.train_batch_size 

        time_points = train["time"].shape[-1]
        data = train['data']
        time = train['time']
        param = train['param']

        train_dict, test_dict = split_data(data, time, param, train_fraq = 0.6)

        train_dataset = SimpleDataSet(train_dict)
        test_dataset = SimpleDataSet(test_dict)

        if self.normal:
            train_dataset.preprocess_data()
            test_dataset.preprocess_data()

        train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=num_train_batches, shuffle=False,pin_memory=False)
        test_dataset = torch.utils.data.DataLoader(test_dataset, batch_size=num_test_batches, shuffle=False,pin_memory=False)

        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=1e-5)
        
        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    

        current_epoch = 0
        best_loss = float("inf")

        self.model.to(self.device)

        save_path = f'models/{self.model_type}/checkpoints/'
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        early_stopping = EarlyStopping(save_path, 500, True, delta=1e-3)

        if self.model_type == 'VAE':
            latent_dims = self.config['Net']['latent_dim']

            prior_mu = torch.zeros((latent_dims), dtype=torch.float64).to(self.device)
            prior_sigma = torch.ones((latent_dims), dtype=torch.float64).to(self.device)

            prior = Normal(prior_mu, prior_sigma)

            for epoch in range(current_epoch, self.max_epoch):
                    
                optimizer.zero_grad()

                train_res = {}
                train_res["train_param_loss"] = 0.
                train_res["train_kl_loss"] = 0.

                total_loss=0.

                for step, (data, param, u_samples, time) in enumerate(train_dataset):

                    data = data.type(torch.float)
                    param = param.type(torch.float)
                    time = time.type(torch.float)

                    true_param=param.to(self.device)
                    data_encoder = data.detach().to(self.device)
                    enc_time = torch.tensor(time[0,:]).to(self.device)

                    pred_param,pred_mu,pred_std= self.model.compute(data_encoder,enc_time)

                    #loss--klv
                    fp_distr = Normal(pred_mu,pred_std)
                    kldiv_z0_gau = kl_divergence(fp_distr, prior)
                    train_kld = torch.mean(kldiv_z0_gau)

                    #loss--mse,likelihood
                    train_param_loss = mse_loss(true_param, pred_param)

                    total_loss = train_param_loss+total_loss+train_kld

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
                
                #backwards
                total_loss.backward()
                optimizer.step()
                
                test_total_loss = 0.
                if epoch % self.eval_epoch == 0:
                    test_epoch = int(epoch/self.eval_epoch)
                    with torch.no_grad():
                        test_res = {}
                        test_res["test_kl_loss"] = 0.
                        test_res["test_param_loss"] = 0.

                        for step, (data, param,u_samples, time) in enumerate(test_dataset):
                            #true_param = torch.cat((rho, sigma, param), dim=-1)

                            data = data.type(torch.float)
                            param = param.type(torch.float)
                            time = time.type(torch.float)

                    
                            true_param=param.to(self.device)

                            data_encoder = data.to(self.device)
                            enc_time = torch.tensor(time[0,:]).to(self.device)
                            pred_param,pred_mu,pred_std = self.model.compute(data_encoder,enc_time)

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


                        early_stopping((test_res["test_param_loss"]+test_res["test_kl_loss"]), self.model)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            return

                CosineLR.step()

        else:
            for epoch in range(current_epoch, self.max_epoch):
                optimizer.zero_grad()

                train_res = {}
                # train_res["loss"] = 0.
                # train_res["train_param_likelihood"] = 0.
                train_res["train_param_loss"] = 0.

                total_loss = 0.

                for step, (data, param, u_samples, time) in enumerate(train_dataset):

                        data = data.type(torch.float)
                        param = param.type(torch.float)
                        time = time.type(torch.float)

                        true_param = param.to(self.device)

                        data_encoder = data.detach().to(self.device)
                        enc_time = torch.tensor(time[0,:]).to(self.device)

                        pred_param = self.model.compute(data_encoder,enc_time)

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

                # backwards
                total_loss.backward()
                optimizer.step()

                test_total_loss = 0.
                if epoch % self.eval_epoch == 0:
                    self.model.eval()
                    test_epoch = int(epoch / self.eval_epoch)
                    with torch.no_grad():
                        test_res = {}
                        test_res["test_param_loss"] = 0.
                        for step, (data,  param, u_samples,time) in enumerate(train_dataset):

                            data = data.type(torch.float)
                            param = param.type(torch.float)
                            time = time.type(torch.float)

                            true_param=param.to(self.device)

                            data_encoder =data.detach().to(self.device)
                            enc_time = torch.tensor(time[0,:]).to(self.device)


                            pred_param= self.model.compute(data_encoder,enc_time)

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


                        early_stopping((test_res["test_param_loss"]), self.model)

                        if early_stopping.early_stop:
                            return
                CosineLR.step()


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
