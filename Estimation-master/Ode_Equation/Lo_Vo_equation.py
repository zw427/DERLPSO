from Ode_Equation import comm_tools
from comm_lib.utils import *


def f(t,y, paras):

        x0 = y[0]
        x1 = y[1]
        #    print('type', type(t[0]))
        #print(paras)
        a, b, c,d = paras[0],paras[1],paras[2],paras[3]
        # the model equations
        df_x0 = a*x0-b*x0*x1
        df_x1 = d*x0*x1-c*x1
        return np.array([df_x0, df_x1])

def get_one_data(mu,sigma,dim_of_data,t,with_noise=True,conv_rho=None,init_data=None,init_range=None,conv_sigma=None,f=f):

    return comm_tools.get_one_data(mu,sigma,dim_of_data,t,with_noise=with_noise,conv_rho=conv_rho,init_data=init_data,conv_sigma=conv_sigma,f=f)

def est_param(t, data_all, param_init):
    return comm_tools.est_param(t, data_all, param_init, f)


