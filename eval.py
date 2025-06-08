from Estimation.interface.train_model import simulation, train_model
from Estimation.interface.run_param import infer_batches
from scipy.integrate import odeint
from DERLSPO.ODE import Model
from sklearn.metrics import mean_squared_error

import pandas as pd
import numpy as np
import torch
import csv
import os


class Noise:
    def __init__(self, conv_rho, conv_sigma):
        assert len(conv_rho) == len(conv_sigma)== 2
        self.conv_rho = conv_rho
        self.conv_sigma = conv_sigma


class ODE:
    def __init__(self, name, func, init_data, param_mu, param_sigma):
        
        assert(len(param_mu) == len(param_sigma))

        self.name = name
        self.func = func
        self.init_data = init_data
        self.param_mu = param_mu
        self.param_sigma = param_sigma


class ODE_Evaluate:
    def __init__(self, ode: ODE, models, points, interval, seed: int = 100, noise: Noise = None):

        self.ode = ode 

        supported_models = ['DERLPSO', 'RLLPSO', 'MLP', 'RNN', 'ODE_RNN', 'VAE']
        self.models = list(set(supported_models) & set(models)) 

        self.points = points

        assert(len(interval) == 2)
        self.interval = interval

        self.noise = 'no_noise' if noise is None else 'noise'
        self.noise_config = noise

        np.random.seed(seed)
        torch.manual_seed(seed)

    def get_filenames(self, point: int, train: bool):
        train_or_test = "train" if train else "test"
        
        data_file =  'data/{}/{}/{}/{}/{}.csv'.format(self.ode.name, point, self.noise, train_or_test, 'data')
        time_file =  'data/{}/{}/{}/{}/{}.csv'.format(self.ode.name, point, self.noise, train_or_test, 'time')
        param_file = 'data/{}/{}/{}/{}/{}.csv'.format(self.ode.name, point, self.noise, train_or_test, 'param')
        model_dir = 'models/{}/{}/{}'.format(self.ode.name, point, self.noise)

        return data_file, time_file, param_file, model_dir

    def _simulate_data(self, train: bool, num_data: int):
        # dummy noise conv if needed
        with_noise = False if self.noise_config is None else True
        conv_rho = [0, 0] if self.noise_config is None else self.noise_config.conv_rho
        conv_sigma = [0, 0] if self.noise_config is None else self.noise_config.conv_sigma

        for point in self.points:
            data_file, time_file, param_file, _ = self.get_filenames(point, train)

            if not (os.path.exists(data_file) and os.path.exists(time_file) and os.path.exists(param_file)):
                simulation(None, len(self.ode.init_data), len(self.ode.param_mu), self.ode.param_mu, self.ode.param_sigma, with_noise, conv_rho, conv_sigma, self.interval + [point], self.ode.init_data, self.ode.func, num_data, data_file, time_file, param_file)


    def simulate_train_data(self, num_data: int):
        self._simulate_data(True, num_data)


    def simulate_test_data(self, num_data: int):
        self._simulate_data(False, num_data)


    def train(self, configs):
        for point in self.points:
            data_file, time_file, param_file, model_dir = self.get_filenames(point, True)

            for model in self.models:

                if model == "DERLPSO":
                    continue

                if model == "RLLPSO":
                    continue

                if os.path.exists(data_file) and os.path.exists(time_file) and os.path.exists(param_file):
                    train_model(configs, model_dir, len(self.ode.init_data), len(self.ode.param_mu), data_file, time_file, param_file, self.ode.func, model)

    def _test_derlpso(self, point: int, data, time, params):

        output = []

        for i in range(len(data)):
            t = time[i]

            ode_func = __import__(self.ode.func, globals(), locals(), ['f_alt'])
            f = ode_func.f_alt

            data_  = np.array(data[i])

            model = Model(f, paramNum=len(params[i]), data=data_, time=t, threshold=1e-4)

            model.initParticles()
            model.iterator()
        
            est_params = np.asarray(model.getGBest())
            err = est_params - params[i]
            fit  = odeint(f, self.ode.init_data, t, args=tuple(est_params))

            mse = mean_squared_error(data_, fit)

            output.append({
                "rep": i,
                "tp": point,
                "true_params": params[i],
                "est_params": est_params.tolist(),
                "err": err.tolist(),
                "mse0": mse
            })

        return output
    

    def test(self):
        for point in self.points:
            data_file, time_file, param_file, model_dir = self.get_filenames(point, False)

            data = []
            with open(data_file, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                _ = next(csv_reader)
                for row in csv_reader:
                    row_data = [float(x) for x in row]
                    data.append([row_data[i:i + 2] for i in range(0, len(row_data), 2)])

            time = []
            with open(time_file, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                _ = next(csv_reader)
                for row in csv_reader:
                    row_data = [float(x) for x in row]
                    time.append(row_data)

            param = []
            with open(param_file, 'r') as csvfile:
                csv_reader = csv.reader(csvfile)
                _ = next(csv_reader)
                for row in csv_reader:
                    row_data = [float(x) for x in row]
                    param.append(row_data)

            for model in self.models:

                result = 'results/{}/{}/{}/{}.csv'.format(self.ode.name, point, self.noise, model)
                summary = 'results/{}/{}/{}/{}.txt'.format(self.ode.name, point, self.noise, model)
                os.makedirs(os.path.dirname(result), exist_ok=True)

                if os.path.exists(data_file) and os.path.exists(time_file) and os.path.exists(param_file):
                    if model == "DERLPSO":
                        output = self._test_derlpso(point, data, time, param)
                    elif model == "RLLPSO":
                        continue
                    else:
                        continue

                    df = pd.DataFrame(output)
                    df.to_csv(result, index=False)

                    err_array = np.vstack(df["err"].to_numpy())
                    mse_array = df["mse0"].to_numpy()

                    mean_param_err = err_array.mean(axis=0)
                    std_param_err = err_array.std(axis=0)
                    mean_mse = mse_array.mean()
                    std_mse = mse_array.std()

                    with open(summary, "w") as f:
                        f.write("==========  SUMMARY  ==========\n")
                        f.write(f"Mean parameter error : {mean_param_err}\n")
                        f.write(f"Std. parameter error : {std_param_err}\n")
                        f.write(f"Mean MSE             : {mean_mse}\n")
                        f.write(f"Std. MSE             : {std_mse}\n")

            if all(m in self.models for m in ['MLP', 'RNN', 'ODE_RNN', 'VAE']):
                model_paths = []
                for i in ['MLP', 'RNN', 'ODE_RNN', 'VAE']:
                    model_paths.append('{}/{}/checkpoints/entire.pth'.format(model_dir, i))
                infer_batches(model_paths, data_file, time_file, param_file, self.ode.func, False)

            

                    