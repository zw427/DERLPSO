import sys
from pathlib import Path
current_folder = Path(__file__).absolute().parent
father_folder = str(current_folder.parent)
sys.path.append(father_folder)
import interface.run_param as run_param
from interface.train_model import *




# # common
# dim_of_data=2
# num_of_param=4
# param_mu=[0.4,1.3,1,1]
# param_sigma=[1,1,1,1]
# init_data=[0.9,0.9]
# ode_func= 'Ode_Equation.Lo_Vo_equation'
# data_nums=100
# ## noise
# with_noise=False
# conv_rho= [0.4,0.1]
# conv_sigma=[0.2,0.1]
#
#
# for point in [5,8,10]:
#     time_interval=[0,4,point]
#
#     # with_noise = True
#     # noise = 'noise' if with_noise else 'no_noise'
#     # data_filename='data/lovo/{}/{}/test/{}.csv'.format(point,noise,'data')
#     # time_filename='data/lovo/{}/{}/test/{}.csv'.format(point,noise,'time')
#     # param_filename='data/lovo/{}/{}/test/{}.csv'.format(point,noise,'param')
#     # base_dir='models/model_lovo/{}/{}'.format(point,noise)
#     # # simulation('../configs/lovo.yaml',dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
#     # #            time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
#     # model_paths= []
#     # for i in ['MLP', 'RNN', 'ODE_RNN', 'VAE']:
#     #     model_paths.append('{}/{}/checkpoints/entire.pth'.format(base_dir, i) )
#     #
#     # run_param.infer_batches(model_paths,data_filename,time_filename,param_filename,ode_func,False)
#     # print('---lovo---{} point-- {} over------'.format(point, noise))
#     with_noise = False
#     noise = 'noise' if with_noise else 'no_noise'
#
#     data_filename = 'data/lovo/{}/{}/test1/{}.csv'.format(point, noise, 'data')
#     time_filename = 'data/lovo/{}/{}/test1/{}.csv'.format(point, noise, 'time')
#     param_filename = 'data/lovo/{}/{}/test1/{}.csv'.format(point, noise, 'param')
#     # base_dir = 'models/model_lovo/{}/{}'.format(point, noise)
#     simulation('../configs/lovo.yaml',dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
#                time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
#     # model_paths = []
#     # for i in ['MLP', 'RNN', 'ODE_RNN', 'VAE']:
#     #     model_paths.append('{}/{}/checkpoints/entire.pth'.format(base_dir, i))
#     # run_param.infer_batches(model_paths, data_filename, time_filename, param_filename, ode_func,False)
#     # print('---lovo---{} point-- {} over------'.format(point, noise))

# ## common
# dim_of_data=3
# num_of_param=3
# param_mu=[2, 4, 1]
# param_sigma=[0.5, 0.5, 0.5]
# init_data=[0, 1, 1.25]
# ode_func= 'Ode_Equation.Lorenz_equation'
# data_nums=100
# conv_rho= [0.2,0.1]
# conv_sigma=[0.2,0.1]
#
# for point in [5,8,10]:
#     time_interval=[0,4,point]
#
#     with_noise = True
#     noise = 'noise' if with_noise else 'no_noise'
#     data_filename='data/lv/{}/{}/test/{}.csv'.format(point,noise,'data')
#     time_filename='data/lv/{}/{}/test/{}.csv'.format(point,noise,'time')
#     param_filename='data/lv/{}/{}/test/{}.csv'.format(point,noise,'param')
#     base_dir='models/model_lv/{}/{}'.format(point,noise)
#     # simulation('../configs/lv.yaml',dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
#     #            time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
#     model_paths= []
#     for i in ['MLP', 'RNN', 'ODE_RNN', 'VAE']:
#         model_paths.append('{}/{}/checkpoints/entire.pth'.format(base_dir, i) )
#
#     run_param.infer_batches(model_paths,data_filename,time_filename,param_filename,ode_func,False)
#     print('---lv---{} point-- {} over------'.format(point, noise))
#     with_noise = False
#     noise = 'noise' if with_noise else 'no_noise'
#
#     data_filename = 'data/lv/{}/{}/test/{}.csv'.format(point, noise, 'data')
#     time_filename = 'data/lv/{}/{}/test/{}.csv'.format(point, noise, 'time')
#     param_filename = 'data/lv/{}/{}/test/{}.csv'.format(point, noise, 'param')
#     base_dir = 'models/model_lv/{}/{}'.format(point, noise)
#     # simulation('../configs/lv.yaml',dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
#     #            time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
#     model_paths = []
#     for i in ['MLP', 'RNN', 'ODE_RNN', 'VAE']:
#         model_paths.append('{}/{}/checkpoints/entire.pth'.format(base_dir, i))
#     run_param.infer_batches(model_paths, data_filename, time_filename, param_filename, ode_func,False)
#     print('---lv---{} point-- {} over------'.format(point, noise))
#
# common
dim_of_data=2
num_of_param=2
param_mu=[0.7,0.8]
param_sigma=[2,2]
init_data=[0,0]
ode_func= 'Ode_Equation.FN_equation'
data_nums=100
## noise
with_noise=False
conv_rho= [0.4,0.1]
conv_sigma=[0.2,0.1]
noise='noise' if with_noise else 'no_noise'

for point in [5,8,10]:
    time_interval=[0,20,point]

    # with_noise = True
    # noise = 'noise' if with_noise else 'no_noise'
    # data_filename='data/fn/{}/{}/test1/{}.csv'.format(point,noise,'data')
    # time_filename='data/fn/{}/{}/test1/{}.csv'.format(point,noise,'time')
    # param_filename='data/fn/{}/{}/test1/{}.csv'.format(point,noise,'param')
    # # base_dir='models/model_fn/{}/{}'.format(point,noise)
    # simulation('../configs/fn.yaml',dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
    #            time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
    # model_paths= []
    # for i in ['MLP', 'RNN', 'ODE_RNN', 'VAE']:
    #     model_paths.append('{}/{}/checkpoints/entire.pth'.format(base_dir, i) )
    #
    # run_param.infer_batches(model_paths,data_filename,time_filename,param_filename,ode_func,False)
    #
    # print('---fn---{} point-- {} over------'.format(point, noise))

    with_noise = False
    noise = 'noise' if with_noise else 'no_noise'

    data_filename = 'data/fn/{}/{}/test2/{}.csv'.format(point, noise, 'data')
    time_filename = 'data/fn/{}/{}/test2/{}.csv'.format(point, noise, 'time')
    param_filename = 'data/fn/{}/{}/test2/{}.csv'.format(point, noise, 'param')
    base_dir = 'models/model_fn/{}/{}'.format(point, noise)
    # simulation('../configs/fn.yaml',dim_of_data, num_of_param, param_mu, param_sigma, with_noise, conv_rho, conv_sigma,
    #            time_interval, init_data, ode_func, data_nums, data_filename, time_filename, param_filename)
    model_paths = []
    for i in ['MLP', 'RNN', 'ODE_RNN', 'VAE']:
        model_paths.append('{}/{}/checkpoints/entire.pth'.format(base_dir, i))
    run_param.infer_batches(model_paths, data_filename, time_filename, param_filename, ode_func,False)
    print('---fn---{} point-- {} over------'.format(point,noise))

