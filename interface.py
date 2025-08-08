import os
from typing import List

import numpy as np
import torch
import pandas as pd

from equations import ODE_Equation
from Estimators.estimator import BaseModel
from Estimators.ml_models import MLModel, infer_batches, train_model

from Estimators.DERLPSO import DERLPSO


class DE_Models: 
    def __init__():
        pass 


class ODE_Models(DE_Models):
    def __init__(self, equation: ODE_Equation, init_data: List[float], 
                 param_mu: List[float], param_sigma: List[float],
                 config = "models/configs/fn.yaml"):
        assert len(param_mu) == len(param_sigma), "param_mu and param_sigma must have the same length"
        self.equation = equation
        self.init_data = init_data
        self.param_mu = param_mu
        self.param_sigma = param_sigma


    def get_filenames(self, interval: List[float], point: int, train: bool) -> List[str]:
        mode = 'train' if train else 'test'
        data_file =  f'data/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}/{mode}/data.csv'
        time_file =  f'data/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}/{mode}/time.csv'
        param_file = f'data/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}/{mode}/param.csv'
        model_dir = f'saved_models/{self.equation.name}/{str(interval[0]) + "-" + str(interval[-1])}/{point}'
        return data_file, time_file, param_file, model_dir
    

    def simulate_data(self, num_data: int, interval: List[float], points: List[int], seed: int = None, train: bool = True, over_write: bool = False) -> None:    
        
        assert len(interval) == 2, "interval must have exactly two floats [start, end]"
        
        if not seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print("Seed set to:", seed)

        for point in points:
            data_file, time_file, param_file, _ = self.get_filenames(interval, point, train)
            if not (os.path.exists(data_file) and os.path.exists(time_file) and os.path.exists(param_file)) or over_write:
                generate_data(
                    equation=self.equation,
                    param_mu=self.param_mu,
                    param_sigma=self.param_sigma,
                    init_data=self.init_data,
                    point=point,
                    interval=interval,
                    num_data=num_data,
                    data_filename=data_file,
                    time_filename=time_file,
                    param_filename=param_file
                )


    def evaluate(self, models: List[BaseModel], interval: List[float], points: List[int], seed: int = None) -> None:
        
        assert len(interval) == 2, "interval must have exactly two floats [start, end]"
        
        if not seed:
            np.random.seed(seed)
            torch.manual_seed(seed)
            print("Seed set to:", seed)

        if any([isinstance(model, MLModel) for model in models]):
            for point in points:
                data_file, time_file, param_file, _ = self.get_filenames(interval, point, train=True)
                if not (os.path.isfile(data_file) and os.path.isfile(time_file) and os.path.isfile(param_file)):
                    print("Please simulate the training data first.")
                    return 

        for point in points:
            data_file, time_file, param_file, _ = self.get_filenames(interval, point, train=False)
            if not (os.path.isfile(data_file) and os.path.isfile(time_file) and os.path.isfile(param_file)):
                print("Please simulate the testing data first.")
                return 

        ### evaluate here
        for m in models:
            for point in points:
                if issubclass(m, MLModel):
                    data_file, time_file, param_file, model_dir = self.get_filenames(interval, point, train=True)
                
                    configs = self.equation.configs
                    train_model(configs, model_dir, len(self.init_data), len(self.param_mu), data_file, time_file, param_file, self.equation.f(), m.__name__)

                    data_file, time_file, param_file, model_dir = self.get_filenames(interval, point, train=False)

                    model_paths = model_dir + "/" + m.__name__+ "/" + "checkpoints/entire.pth"
                    infer_batches(model_paths, data_file, time_file, param_file, self.equation.f(), False)

                elif m == DERLPSO:
                    data_file, time_file, param_file, _ = self.get_filenames(interval, point, train=False)
                    output = read_data(data_file, time_file, param_file)
                    data = output['data']
                    param = output['param']
                    time = output['time']

                    
                    results_dir = f'results/{self.equation.name}/{str(interval[0])}-{str(interval[-1])}/{point}'
                    os.makedirs(results_dir, exist_ok=True)
                        
                    # Store all results
                    results_list = []
                    
                    print('\n')
                    for i in range(data.shape[0]):
                        d = np.array(data[i])
                        p = param[i]
                        t = time[i]
                        model = DERLPSO(self.equation, p, d, t)
                        result = model.test()
                        
                        # Add sample index to result
                        result['sample_id'] = i
                        results_list.append(result)
                        
                        print(f"Sample {i:2d} | True: [{', '.join([f'{p:.4f}' for p in result['true_params']])}] | "
                            f"Est: [{', '.join([f'{p:.4f}' for p in result['est_params']])}] | "
                            f"Error: [{', '.join([f'{e:+.4f}' for e in result['err']])}] | "
                            f"MSE: {result['mse0']:.2e}")
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(results_list)
                    
                    # Save to CSV
                    csv_file = f'{results_dir}/DERLPSO_results.csv'
                    df.to_csv(csv_file, index=False)
                    
                    # Print summary statistics (no file saving)
                    print(f"Model: {m.__name__}")
                    print("=" * 30)
                    print(f"Interval: {interval}")
                    print(f"Points: {point}")
                    print(f"Number of samples: {len(results_list)}")
                    print(f"Mean MSE: {df['mse0'].mean():.2e}")
                    print(f"Std MSE: {df['mse0'].std():.2e}")
                    print(f"Min MSE: {df['mse0'].min():.2e}")
                    print(f"Max MSE: {df['mse0'].max():.2e}")
                    
                    # Parameter error statistics
                    err_array = np.array(df['err'].tolist())
                    print(f"Mean parameter errors: {err_array.mean(axis=0)}")
                    print(f"Std parameter errors: {err_array.std(axis=0)}")
                    
                    print(f"Results saved to {csv_file}")
          










def generate_data(equation: Equation, param_mu: List[float], param_sigma: List[float],
                init_data: List[float], point: int, interval: List[float], num_data: int,
                data_filename: str, time_filename: str, param_filename: str) -> Dict[str, np.ndarray]:
    
    if data_filename is not None and not os.path.exists(data_filename):
        os.makedirs(os.path.dirname(data_filename), exist_ok=True)
    
    time_points = np.linspace(interval[0], interval[-1], point)
    
    data_list  = []
    param_list = []
    time_list  = []

    while len(data_list) < num_data:
        success, output = get_one_data(param_mu, param_sigma, len(init_data), time_points, init_data, f=equation.f())
        if success:
            data_list.append(output['data'])
            param_list.append(output['param'])
            time_list.append(output['time'])
        else:
            continue

    data  = np.concatenate(data_list,  axis=0)
    param = np.concatenate(param_list, axis=0)
    time  = np.concatenate(time_list,  axis=0)

    data  = data.reshape(data.shape[0], -1)
    param = param.reshape(param.shape[0], -1)
    time  = time.reshape(time.shape[0], -1)
    
    header_data = ",".join(['d_{}'.format(i) for i in range(data.shape[-1])])
    header_param = ",".join(['p_{}'.format(i) for i in range(param.shape[-1])])
    header_time = ",".join(['t_{}'.format(i) for i in range(time.shape[-1])])
    
    np.savetxt(data_filename, data, header=header_data, delimiter=',')
    np.savetxt(param_filename, param, header=header_param, delimiter=',')
    np.savetxt(time_filename, time, header=header_time, delimiter=',')
    
    print()
    print(f"Data generated and saved to {data_filename}, {time_filename}, {param_filename} for point {point}.")
    
    return {'data': data, 'params': param, 'time': time}


def get_one_data(param_mu,param_sigma,dim_of_data, t,init_data,f):

    all_data = None
    all_param = None

    retry = 0

    while True :

        param_set=None
        for mu, sigma in zip(param_mu, param_sigma):
            param = np.random.normal(loc=mu, scale=sigma, size=1)
            param_set = param.reshape((1, 1)) if param_set is None else np.concatenate(
                (param_set, param.reshape(1, 1)), axis=1)

        param=np.abs(param_set)

        # generate data
        # init value

        x0 = np.array(init_data)
        res = solve_ivp(f, [t[0], t[-1]], x0, t_eval=t, args=(param))

        data = np.transpose(res.y, (1, 0))

        if not data.shape[0] ==len(t):
           retry=retry+1
           if retry>50:
                return False,None
           else:
               continue

        data = data.reshape((1, len(t),dim_of_data))

        all_data = copy.deepcopy(data) if all_data is None else np.concatenate((all_data, copy.deepcopy(data)),
                                                                               axis=0)
        all_param=copy.deepcopy(param) if all_param is None else np.concatenate((all_param,copy.deepcopy(param)),axis=0)
        
        break

    # print('==========================Exporting===========================')
    all_t = t.reshape((1, len(t)))

    return True,{'data': all_data,'param': all_param,'time': all_t}











