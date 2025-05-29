import sys
from pathlib import Path
current_folder = Path(__file__).absolute().parent
father_folder = str(current_folder.parent)
sys.path.append(father_folder)
from interface import run_param
from comm_lib.load_configs import load_configure
from Ode_Equation.comm_tools import *
import numpy as np
import os


def simulation(configure_file, dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
               time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename):
    ode_func = __import__(ode_func, globals(), locals(), ['get_data', 'f'])
    t_start, t_end, time_points = time_interval[0], time_interval[1], time_interval[2]
    time = np.arange(t_start, t_end, step=(t_end - t_start) / time_points)
    run_param.generate_param_data(data_nums, param_mu, param_sigma, dim_of_data,time, with_noise, conv_rho,
                                  conv_sigma, init_data, data_filename, time_filename, param_filename, ode_func)


def train_model(configure_file,base_dir, dim_of_data, num_of_param, data_filename, time_filename, param_filename, ode_func,model_type):
    if isinstance(ode_func, str):
        ode_func = __import__(ode_func, globals(), locals(), ['get_data', 'f'])

    configs_param = load_configure(configure_file,model_type)
    log_path = '{}/{}/{}'.format(base_dir,configs_param['type'], "train_model.log")
    if not os.path.exists(os.path.dirname(log_path)):
        os.makedirs(os.path.dirname(log_path))
    logger = get_logger(logpath=log_path, filepath=os.path.abspath(__file__))

    data = run_param.read_data(data_filename, time_filename, param_filename)
    run_param.train_with_data(configs_param, base_dir,num_of_param, dim_of_data, data)



def predict_param(model_path, normal,data_filename, time_filename):
    '''
    :param configure_file: 配置文件路径
    :param data_filename:  预测数据路径
    :param time_filename
    :param output_path
    :return:
    '''

    run_param.predict(model_path, data_filename, time_filename)
