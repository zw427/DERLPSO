from scipy.integrate import solve_ivp

from Ode_Equation import comm_tools
from Ode_Equation.comm_tools import *
from comm_lib.utils import *


def f(t,y,paras):

        x0 = y[0]
        x1 = y[1]

        try:
            a = paras['a'].value
            b = paras['b'].value


        except Exception:
            a,b= paras[0],paras[1]

        # # the model equations
        df_x0 = 3.0 * (x0 - (x0 ** 3) / 3 + x1 - 0.4)
        df_x1 = (-1 / 3.0) * (x0 - a + b * x1)

        return np.array([df_x0, df_x1])


def get_one_data(mu,sigma,dim_of_data,t,with_noise=True,conv_rho=None,init_data=None,init_range=None,conv_sigma=None,f=f):


   all_data_noise = None
   all_data = None
   all_param = None

   retry = 0

   while True:
       param = np.abs(create_param_set(1, mu, sigma))

       # generate data
       # init value
       if init_range is not None:
           x0 = np.random.uniform(init_range[0], init_range[1], dim_of_data)
       else:
           x0 = np.array(init_data)
       res = solve_ivp(f, [t[0], t[-1]], [0, 0], 'RK23', t_eval=t, args=(param))

       data = np.transpose(res.y, (1, 0))

       if not data.shape[0] ==len(t):
           retry=retry+1
           if retry>50:
                return False,None
           else:
               continue

       data = data.reshape((1, len(t), dim_of_data))

       all_data = copy.deepcopy(data) if all_data is None else np.concatenate((all_data, copy.deepcopy(data)),
                                                                              axis=0)
       all_param = copy.deepcopy(param) if all_param is None else np.concatenate((all_param, copy.deepcopy(param)),
                                                                                 axis=0)
       # ori=data
       if with_noise:

           while (True):
               use_rho = np.abs(np.random.normal(loc=conv_rho[0], scale=conv_rho[1], size=1))
               if use_rho < 1 and use_rho > 0:
                   break

           tmp_sigma = None
           for i in range(data.shape[-1]):
               # add multiple variable noise
               use_sigma = np.random.normal(loc=conv_sigma[0], scale=conv_sigma[1], size=1)
               conv = use_sigma ** 2 * getCovarianceMatrix(use_rho, data.shape[1])
               data[:, :, i] = np.random.multivariate_normal(data[:, :, i].squeeze(0), conv)
               tmp_sigma = use_sigma.reshape(1, 1) if tmp_sigma is None else np.concatenate(
                   (tmp_sigma, use_sigma.reshape(1, 1)), axis=1)
           all_data_noise = copy.deepcopy(data) if all_data_noise is None else np.concatenate(
               (all_data_noise, copy.deepcopy(data)), axis=0)
       break

   # print('==========================Exporting===========================')
   all_t = t.reshape((1, len(t)))

   return True, [all_data, all_data_noise, all_param, all_t]

def est_param(t, data_all, param_init):
    return comm_tools.est_param(t, data_all, param_init, f)


