
import os
from os import makedirs

import deepdish as dd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from comm_lib import utils, tools


def visualize(dir,true_y, pred_y,name):
    if not os.path.exists(dir):
         makedirs(dir)
    for i in range(0,pred_y.shape[1]):
        plt.subplot(2, 2, i+1)
        plt.scatter(np.arange(0,true_y.shape[0],step=1), (true_y[:,i]), color='blue')
        plt.scatter(np.arange(0,pred_y.shape[0],step=1), (pred_y[:,i]), color='red')
        plt.xlabel('time')
        plt.ylabel('value')
    plt.savefig('{}/{}'.format(dir, name))
    plt.clf()




def draw_pred_data(file,save_dir):
    pred_data,true_data=prepare_data_from_file(file)
    num=0
    #viz
    for i in range(pred_data.shape[0]):
        tmp_true=true_data[i,:,:]
        tmp_pred=pred_data[i,:,:]
        visualize(save_dir, tmp_true, tmp_pred, 'compare_{0}'.format(num))
        num=num+1




def prepare_data_from_file(file):
    data=np.load(file)
    pred_data=data['pred']
    true_data=data['true']
    return  pred_data,true_data

def getdata(file):
    data = dd.io.load(file)
    data_x=data["origin_data"]
 #   data_noise=data['data_with_noise']

    for i in range(data_x.shape[-1]):
        plt.subplot(data_x.shape[-1], 1, i + 1)
        plt.plot(np.arange(0, data_x.shape[1], step=1), (data_x[0,:,i]), color='blue')
        # plt.scatter(np.arange(0, pred_y.shape[0], step=1), (pred_y[:, i]), color='red')
        plt.xlabel('time')
        plt.ylabel('x_{0}'.format(i))
    plt.savefig('{}'.format('a.png'))
    plt.clf()
#
# getdata('..\Encoder\models\models_lv\lv_step_1_points_5_sigma_0.2_size_1000\infer\infer.hd5')



def draw_infer():
    config_file_list = os.listdir('../Encoder/models/models_lv')

    for configure_file in config_file_list:
        if 'size_1000' in configure_file:
            log_path = '../Encoder/logs/logs_lv/'+configure_file+'/infer.log'
            logger = utils.get_logger(logpath=log_path, filepath=os.path.abspath(__file__))
            print('start drawing config file {}'.format(configure_file))
            dir = '../Encoder/models/models_lv/'+configure_file+'/infer'
            filename=dir+'/infer.hd5'
            dataset=dd.io.load(filename)


            if not os.path.exists(dir):
                 makedirs(dir)
            # origin_data = dataset['origin_data']
            # model_data = dataset['model_data']
            # math_data = dataset['math_data']
            # compare_data = dataset['compare_data']
            origin_param = dataset['origin_param']
            model_param = dataset['model_param']
            math_param = dataset['math_param']
            compare_param = dataset['compare_param']
            #draw each pred data
            data_pic_dir=dir+'/data_pic'
            if not os.path.exists(data_pic_dir):
                 makedirs(data_pic_dir)

            # list loss
            model_rec_time=dataset['model_rec_time']
           # model_rec_loss=dataset['model_rec_loss']
            model_param_loss=dataset['model_param_loss']
            model_each_param_loss=dataset['model_each_param_loss']
            math_rec_time=dataset['math_rec_time']
           # math_rec_loss=dataset['math_rec_loss']
            math_param_loss=dataset['math_param_loss']
            math_each_param_loss=dataset['math_each_param_loss']
            compare_rec_time=dataset['compare_rec_time']
           # compare_rec_loss=dataset['compare_rec_loss']
            compare_param_loss=dataset['compare_param_loss']
            compare_each_param_loss=dataset['compare_each_param_loss']

            logger.info('model rec time:{:.3f}'.format(model_rec_time))
          #  logger.info('model rec loss:{:.3f}'.format(model_rec_loss))
            logger.info('model total param loss:{:.3f}'.format(model_param_loss))

            logger.info('model each param loss:' + str(['%.3f' % val for val in model_each_param_loss]))


            logger.info('---------------------------------------------------------------------')
            logger.info('math rec time:{:.3f}'.format(math_rec_time))
           # logger.info('math  rec loss:{:.3f}'.format(math_rec_loss))
            logger.info('math total param loss:{:.3f}'.format(math_param_loss))

            logger.info(
                'math each param loss:' + str(["%.3f" % val for val in math_each_param_loss]))

            logger.info('---------------------------------------------------------------------')
            logger.info('compare model rec time:{:.3f}'.format(compare_rec_time))
           # logger.info('compare model rec loss:{:.3f}'.format(compare_rec_loss))
            logger.info('compare model total param loss:{:.3f}'.format(compare_param_loss))
            logger.info(
                'compare model  each param loss:' + str(['%.3f' % val for val in compare_each_param_loss]))





            # for num in range(0,10):
            #     for i in range(0,origin_data.shape[2]):
            #         plt.subplot(origin_data.shape[2], 1, i+1)
            #         plt.ylim(min(origin_data[num,:,i]),max(origin_data[num,:,i]))
            #         plt.scatter(np.arange(0,origin_data.shape[1],step=1), (origin_data[num,:,i]), color='blue',label='true')
            #         plt.scatter(np.arange(0,origin_data.shape[1],step=1), (model_data[num,:,i]), color='red',label='model')
            #         # plt.scatter(np.arange(0, origin_data.shape[1], step=1), (math_data[num, :,i]), color='black',label='math')
            #         # plt.scatter(np.arange(0, origin_data.shape[1], step=1), (compare_data[num,:,i]), color='yellow',label='compare')
            #         plt.xlabel('time')
            #         plt.ylabel('value')
            #     name='data_{}'.format(num)
            #     plt.legend()
            #     plt.savefig('{}/{}'.format(data_pic_dir,name))
            #     plt.clf()
            # print('draw data picture successfully')
            #draw param
            param_pic_dir = dir + '/param_pic'
            if not os.path.exists(param_pic_dir):
                 makedirs(param_pic_dir)

            #export to excel
            four_data=None

            for i in range(0, origin_param.shape[0]):
               four_data=origin_param[i,:] if four_data is None else np.row_stack((four_data,origin_param[i,:]))
               four_data=np.row_stack((four_data, model_param[i, :]))
               four_data=np.row_stack((four_data, compare_param[i, :]))
               four_data=np.row_stack((four_data, math_param[i, :]))

            data_dir=param_pic_dir+'/data.xlsx'
            df=pd.DataFrame(four_data)
            write=pd.ExcelWriter(data_dir)
            df.to_excel(write,'sheet_1',float_format='%.3f')
            write.save()

            data_dir = param_pic_dir + '/data_ori.xlsx'
            df = pd.DataFrame(tools.to_np(origin_param))
            write = pd.ExcelWriter(data_dir)
            df.to_excel(write, 'sheet_1', float_format='%.3f')
            write.save()


            data_dir = param_pic_dir + '/data_model.xlsx'
            df = pd.DataFrame(tools.to_np(model_param))
            write = pd.ExcelWriter(data_dir)
            df.to_excel(write, 'sheet_1', float_format='%.3f')
            write.save()

            data_dir = param_pic_dir + '/data_math.xlsx'
            df = pd.DataFrame(tools.to_np(math_param))
            write = pd.ExcelWriter(data_dir)
            df.to_excel(write, 'sheet_1', float_format='%.3f')
            write.save()

            data_dir = param_pic_dir + '/data_compare.xlsx'
            df = pd.DataFrame(tools.to_np(compare_param))
            write = pd.ExcelWriter(data_dir)
            df.to_excel(write, 'sheet_1', float_format='%.3f')

            write.save()
            write.close()

            print('draw param picture successfully')
            for i in range(0,origin_param.shape[1]):
                plt.ylim(min(origin_param[:,i]), max(origin_param[:,i]))
                plt.scatter(np.arange(0,origin_param.shape[0],step=1), (origin_param[:,i]), color='blue',label='true')
                plt.scatter(np.arange(0,origin_param.shape[0],step=1), (model_param[:,i]), color='red',label='model')
                plt.scatter(np.arange(0,origin_param.shape[0],step=1), (math_param[:,i]), color='black',label='math')
                plt.scatter(np.arange(0,origin_param.shape[0],step=1), (compare_param[:,i]), color='yellow',label='mlp')
                plt.xlabel('sample')
                plt.ylabel('value_{}'.format(i))
                name = 'param_sample_{}'.format(i)
                plt.legend(loc='lower right')
                plt.savefig('{}/{}'.format(param_pic_dir, name))
                plt.clf()
            # rho
            i=0
            delt_math = math_param[:, i] - origin_param[:, i]
            delt_origin = model_param[:, i] - origin_param[:, i]
            delt_vae = compare_param[:, i] - origin_param[:, i]
            sort_index = np.argsort(delt_math)
            plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_origin[sort_index]), color='red', label='model')
            plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_math[sort_index]), color='black', label='math')
            plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_vae[sort_index]), color='yellow', label='mlp')
            plt.xlabel('sample')
            plt.ylabel('value_{}'.format(i))
            name = 'param_rho'
            plt.legend(loc='lower right')
            plt.savefig('{}/{}'.format(param_pic_dir, name),dpi=500,bbox_inches='tight')
            plt.clf()

            #sigma
            for i in [1,2,3]:
                plt.subplot(3,1,(i-1)+1)
                delt_math = math_param[:, i] - origin_param[:, i]
                delt_origin = model_param[:, i] - origin_param[:, i]
                delt_vae = compare_param[:, i] - origin_param[:, i]
                sort_index = np.argsort(delt_math)
                plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_origin[sort_index]), color='red', label='model')
                plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_math[sort_index]), color='black', label='math')
                plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_vae[sort_index]), color='yellow', label='mlp')
                plt.xlabel('sample')
                plt.ylabel('value_{}'.format(i))

                plt.legend(loc='lower right')
            name = 'param_sigma'
            plt.savefig('{}/{}'.format(param_pic_dir, name),dpi=500,bbox_inches='tight')
            plt.clf()

            for i in [4, 5, 6]:
                plt.subplot(3, 1, (i - 4) + 1)
                delt_math = math_param[:, i] - origin_param[:, i]
                delt_origin = model_param[:, i] - origin_param[:, i]
                delt_vae = compare_param[:, i] - origin_param[:, i]
                sort_index = np.argsort(delt_math)
                plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_origin[sort_index]), color='red', label='model')
                plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_math[sort_index]), color='black', label='math')
                plt.plot(np.arange(1, origin_param.shape[0] + 1, step=1), (delt_vae[sort_index]), color='yellow', label='mlp')
                plt.xlabel('sample')
                plt.ylabel('value_{}'.format(i))

                plt.legend(loc='lower right')
            name = 'param_ode'
            plt.savefig('{}/{}'.format(param_pic_dir, name), dpi=500, bbox_inches='tight')
            plt.clf()


            # abc


            print('draw param picture successfully')







#
# import  deepdish as dd
# import os
# import matplotlib.pyplot as plt
# from os import makedirs
# import numpy as np
# from lib import utils
# from dataGenerate.InferDataset import InferDataset
# if __name__ == "__main__":
#         configure_file='lv_step_1_points_5_sigma_0.2_size_1000_lop'
#
#
#         print('start drawing config file {}'.format(configure_file))
#         dir = 'models_encoder/'+configure_file+'/data'
#         filename=dir+'/test_mnoise_200.hd5'
#         dataset=dd.io.load(filename)
#
#         data_noise = dataset['data_with_noise']
#         data_origin=dataset['ori_data']
#         param=dataset['params']
#         # self.ori_data = torch.DoubleTensor(dataset['ori_data'])[indexes]
#         # self.params = torch.DoubleTensor(np.array([dataset['params'][i] for i in indexes]))
#         # self.sigma = torch.DoubleTensor(np.array([dataset['sigma'][i] for i in indexes]))
#         # self.rho = torch.DoubleTensor(np.array([dataset['rho'][i] for i in indexes]))
#         t = dataset['time_points']
#         #draw each pred data
#         data_pic_dir='data_pic'
#         if not os.path.exists(data_pic_dir):
#              makedirs(data_pic_dir)
#
#         #for num in range(0,10):
#         for i in range(0,200):
#             #plt.subplot(data_noise.shape[2], 1, i+1)
#
#            # plt.ylim(min(data_noise[0:10,:,i]),max(data_noise[0:10,:,i]))
#             print(data_origin[i,:,:])
#             print('param',param[i,:])
#             # plt.plot(np.arange(0,data_noise.shape[1],step=1), (data_noise[:,:,i].reshape(5,200)), color='blue')
#             # plt.xlabel('time')
#             # plt.ylabel('x_{0}'.format(i))
#        #  plt.show()
#        #  name='data'
#        # # plt.legend()
#        #  plt.savefig('{}/{}'.format(data_pic_dir,name))
#        #  plt.clf()
#        #  print('draw data picture successfully')
# # # #         #draw param
# from torchdiffeq import odeint as odeint
from scipy.integrate import ode


def f(t,y,paras):
    """
    Your system of differential equations
    """
#
    x0 = y[0]
    x1 = y[1]
    x2 = y[2]
   # print('type', type(t[0]))

    try:
        a = paras['a'].value
        b = paras['b'].value
        c = paras['c'].value

    except Exception:
        a,b,c= paras
#     a=10
#     b=126.52
#     c=8 / 3
    # the model equations
    df_x0 = a * (x1 - x0)
    df_x1 = x0 * (b - x2) - x1
    df_x2 = x0 * x1- c * x2


    return [df_x0,df_x1,df_x2]

def draw_lv_data():
    x0=[-7.69,-15.61,90.39]
    param=[10,126.52,8/3]
    # x0=[0,1,1.05]
    # param=[10, 28, 2.677]
    t=np.arange(0,8,step=1)

    r=ode(f).set_integrator('dopri5',nsteps=5000)
    r.set_initial_value(x0,t[0]).set_f_params(param)

    t1=t[-1]
    dt=t[1]-t[0]
    data=np.array(x0).reshape((1,-1))

    while r.successful() and r.t<t1:
        tmp_data=np.array(r.integrate(r.t+dt)).reshape((1,-1))
        #print(r.t + dt, tmp_data)
        data=np.concatenate((data,tmp_data),axis=0)
    print(data.shape)



    # plt.plot(data[:,0],data[:,2], color='red')
    #
    # plt.xlabel('x')
    # plt.ylabel('z')
    #
    # plt.legend(loc='lower right')
    # plt.show()

    # #
    print(data[0:10,0])
    print(data[0:10, 1])
    print(data[0:10, 2])
    plt.plot(np.arange(0,len(t),1),data[:,0], color='red', label='x')
    plt.plot(np.arange(0, len(t), 1), data[:, 1], color='blue', label='y')
    plt.plot(np.arange(0, len(t), 1), data[:, 2], color='black', label='z')

    plt.xlabel('t')
    plt.ylabel('x')

    plt.legend(loc='lower right')
    plt.show()
    # name = 'param_ode'
    # plt.savefig('{}/{}'.format(param_pic_dir, name), dpi=500, bbox_inches='tight')
    # plt.clf()

d
