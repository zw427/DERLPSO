from Ode_Equation import comm_tools
from comm_lib.utils import *


#LV ode func
def f(t,y, paras):

        x0 = y[0]
        x1 = y[1]
        x2 = y[2]
        #    print('type', type(t[0]))

        a, b, c = paras[0],paras[1],paras[2]
        # the model equations
        df_x0 = a * (x1 - x0)
        df_x1 = x0 * (b - x2) - x1
        df_x2 = x0 * x1 - c * x2

        return np.array([df_x0, df_x1, df_x2])
def get_one_data(mu,sigma,dim_of_data,t,with_noise=True,conv_rho=None,init_data=None,init_range=None,conv_sigma=None,f=f):

    return comm_tools.get_one_data(mu,sigma,dim_of_data,t,with_noise=with_noise,conv_rho=conv_rho,init_data=init_data,conv_sigma=conv_sigma,f=f)



def est_param(t, data_all, param_init):
        return comm_tools.est_param(t, data_all, param_init,f)












