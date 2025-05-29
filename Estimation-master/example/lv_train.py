from interface.train_model import *


## common
dim_of_data=3
num_of_param=3
param_mu=[2, 4, 1]
param_sigma=[0.5, 0.5, 0.5]
init_data=[0, 1, 1.25]
ode_func= 'Ode_Equation.Lorenz_equation'
data_nums=16000
## noise
with_noise=True
conv_rho= [0.2,0.1]
conv_sigma=[0.2,0.1]
noise='noise' if with_noise else 'no_noise'

for type in ['MLP','RNN','ODE_RNN','VAE']:
    for point in [5,8,10]:
        time_interval=[0,4,point]

        with_noise = True
        noise = 'noise' if with_noise else 'no_noise'
        data_filename='data/lv/{}/{}/train/{}.csv'.format(point,noise,'data')
        time_filename='data/lv/{}/{}/train/{}.csv'.format(point,noise,'time')
        param_filename='data/lv/{}/{}/train/{}.csv'.format(point,noise,'param')
        base_dir='models/model_lv/{}/{}'.format(point,noise)
        if not os.path.exists(data_filename):
            simulation('../configs/lv.yaml',dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
                       time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
        train_model('../configs/lv.yaml',base_dir, dim_of_data, num_of_param, data_filename, time_filename, param_filename, ode_func,type)

        with_noise = False
        noise = 'noise' if with_noise else 'no_noise'

        data_filename = 'data/lv/{}/{}/train/{}.csv'.format(point, noise, 'data')
        time_filename = 'data/lv/{}/{}/train/{}.csv'.format(point, noise, 'time')
        param_filename = 'data/lv/{}/{}/train/{}.csv'.format(point, noise, 'param')
        base_dir = 'models/model_lv/{}/{}'.format(point, noise)
        if not os.path.exists(data_filename):
            simulation('../configs/lv.yaml', dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
                       time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
        train_model('../configs/lv.yaml', base_dir, dim_of_data, num_of_param, data_filename, time_filename, param_filename,
                    ode_func, type)
