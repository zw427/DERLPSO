import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.integrate import ode, solve_ivp

from equations import Equation


def read_data(data_filename: str, time_filename: str, 
              param_filename: Optional[str] = None) -> Dict[str, np.ndarray]:
    data_dict = {}

    time = pd.read_csv(time_filename, delimiter=',').to_numpy()
    time_points = time.shape[-1]
    data_dict['time'] = time

    data = pd.read_csv(data_filename, delimiter=',').to_numpy()
    data = data.reshape(data.shape[0], time_points, -1)
    data_dict['data'] = data

    if param_filename is not None:
        param = pd.read_csv(param_filename, delimiter=',').to_numpy()
        data_dict['param'] = param

    return data_dict


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
