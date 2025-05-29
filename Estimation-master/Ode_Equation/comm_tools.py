
import copy

from lmfit import minimize, Parameters
from scipy.integrate import ode

from comm_lib.utils import *


#生成一条数据,使用于简单的微分方程生成数据
def get_one_data(mu,sigma,dim_of_data, t,with_noise=False,conv_rho=None,init_data=None,init_range=None,conv_sigma=None,f=None):

    # generate params
   # params = np.abs(create_param_set_1(params_size, mu, sigma))

    all_data_noise = None
    all_data = None
    all_param=None



    retry=0

    while True:
        param=np.abs(create_param_set(1,mu,sigma))

        # generate data
        # init value
        if init_range is not None:
            x0 = np.random.uniform(init_range[0], init_range[1], dim_of_data)
        else:
            x0 = np.array(init_data)
        # 1 data_dim tp :1 3 16
        r = ode(f).set_integrator('dopri5', nsteps=1000)
        r.set_initial_value(x0, t[0]).set_f_params(param[0])


        data = np.array(x0).reshape((1, -1))
        t_index=1
        while r.successful() and t_index<len(t):

          #  print('start:',r.t)
            timestep = t[t_index] - t[t_index-1]
            tmp_data = np.array(r.integrate(r.t + timestep)).reshape((1, -1))
          #  print('after:', r.t)
            data = np.concatenate((data, tmp_data), axis=0)
            t_index=t_index+1

        if not (data.shape[0]==len(t)):
            retry=retry+1

            if retry>50:
                print(data.shape[0])
                print(len(t))
                return False,None
            # print(data.shape[0])
            # print("false")
            continue

        data = data.reshape((1, len(t),dim_of_data))


        all_data = copy.deepcopy(data) if all_data is None else np.concatenate((all_data, copy.deepcopy(data)),
                                                                               axis=0)
        all_param=copy.deepcopy(param) if all_param is None else np.concatenate((all_param,copy.deepcopy(param)),axis=0)
        # ori=data
        if with_noise:

            while(True):
                use_rho =np.abs(np.random.normal(loc=conv_rho[0], scale=conv_rho[1], size=1))
                if use_rho<1 and use_rho>0:
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

    return True,[all_data,all_data_noise,all_param,all_t]


def create_param_set(batches,mu_list, sigma_list,lowers=None,uppers=None):
    param_set=None
    if lowers is not None:
        for mu, sigma, lower, upper in zip(mu_list, sigma_list, lowers, uppers):
            param = np.random.normal(loc=mu, scale=sigma, size=1)

            while (param < lower or param > upper):
                param = np.random.normal(loc=mu, scale=sigma, size=1)
            else:
                param = np.random.normal(loc=mu, scale=sigma, size=1)

            param_set = param.reshape((batches, 1)) if param_set is None else np.concatenate(
                (param_set, param.reshape(batches, 1)), axis=1)

    else:

        for mu, sigma in zip(mu_list, sigma_list):

            param = np.random.normal(loc=mu, scale=sigma, size=1)
            param_set = param.reshape((batches, 1)) if param_set is None else np.concatenate(
                (param_set, param.reshape(batches, 1)), axis=1)

    return param_set

#构建噪声的矩阵
def getCovarianceMatrix(rho,num):
    matrix=np.zeros((num,num))
    for i in range(num):
        tmp=np.logspace(start=0,stop=num-i,num=num-i,base=rho,endpoint=False)
        matrix[i,i:num]=tmp

    bottom=matrix.transpose(1,0)

    matrix=matrix+bottom-np.identity(num)

    eigenvalue, featurevector = np.linalg.eig(matrix)


    if not (eigenvalue>=0).all():

        print("value：", eigenvalue)
        print(matrix)
        raise Exception("Not positive matrix")


    return matrix

#装载数据

def prepare_data(dataset,b_train=1):

    if torch.cuda.is_available():
        train_dataset = torch.utils.data.DataLoader(dataset, batch_size=b_train, shuffle=False,pin_memory=False)

    else:
        train_dataset = torch.utils.data.DataLoader(dataset, batch_size=b_train, shuffle=False,pin_memory=False)


    return train_dataset


# -----------------------math method to estimate param-----------------------------------------#

def est_param(t, data_all, param_init,f):
    all_pred = None

    for i in range(data_all.shape[0]):
        # print('start  {0} ....'.format(i))
        data_true = data_all[i, :, :]
        y0 = data_true[0, :]
        params = Parameters()
        for v in range(len(y0)):
            params.add('x_{}'.format(v), value=y0[v], vary=False)
        for v in range(param_init.size):
            params.add(chr(v+65), value=param_init[v])

        # fit model
        result = minimize(residual, params, args=(t, data_true,f), method='powell', tol=1e-3,
                          options={'xtol': 0.001, 'ftol': 0.001, 'maxiter': 1000, 'disp': False})  # leastsq nelder
        if i%50==0:
            print('complete  {0} param ....'.format(i))
        # get pred param
        pred = []
        for name, item in result.params.items():
            if 'x_' not in name:
                pred.append(item.value)

        pred = np.array(pred)
        all_pred = pred.reshape((-1, param_init.size)) if all_pred is None else np.concatenate((all_pred, pred.reshape((-1, param_init.size))),
                                                                                 axis=0)

    return all_pred


def residual(paras, t, data,f):
    """
    compute the residual between actual data and fitted data
    """
    x0=[]
    param=[]
    for par in paras.items():
        if 'x_' in par[0]:
            x0.append(par[1].value)
        else:
            param.append(par[1].value)
    r = ode(f).set_integrator('dopri5', nsteps=5000)
    r.set_initial_value(x0, t[0]).set_f_params(param)

    x = np.array(x0).reshape((1, -1))

    t_index = 1
    while r.successful() and t_index < len(t):
        #  print('start:',r.t)
        timestep = t[t_index] - t[t_index - 1]
        tmp_data = np.array(r.integrate(r.t + timestep)).reshape((1, -1))
        #  print('after:', r.t)
        x = np.concatenate((x, tmp_data), axis=0)
        t_index = t_index + 1

    if x.shape[0] != len(t):
        x = np.concatenate((x, np.zeros((len(t) - x.shape[0], x.shape[1]))), axis=0)

    return x - data


def get_param(data, est_df):
    a = None
    b = None
    c = None
    assert (not torch.isinf(est_df).any())
    assert (not torch.isnan(est_df).any())
    # data[data ==0] = 0.01

    for i in range(data.shape[1]):
        df0 = est_df[:, i, 0]
        df1 = est_df[:, i, 1]
        df2 = est_df[:, i, 2]
        x0 = data[:, i, 0]
        x1 = data[:, i, 1]
        x2 = data[:, i, 2]

        e_a = df0 / (x1 - x0)
        e_b = (df1 + x1 + x0 * x2) / x0
        e_c = -(df2 - x0 * x1) / x2
        for i in range(data.shape[0]):
            num = e_b[i]
            if torch.isinf(num):
                print(x0[i])
                print(df1[i])
                print(x0[i] * x1[i])

        a = e_a.unsqueeze(1) if a is None else torch.cat((a, e_a.unsqueeze(1)), dim=-1)
        b = e_b.unsqueeze(1) if b is None else torch.cat((b, e_b.unsqueeze(1)), dim=-1)
        c = e_c.unsqueeze(1) if c is None else torch.cat((c, e_c.unsqueeze(1)), dim=-1)
    pred_params = torch.cat((a.unsqueeze(2), b.unsqueeze(2), c.unsqueeze(2)), dim=-1)
    assert (not torch.isnan(pred_params).any())
    assert (not torch.isinf(pred_params).any())
    return torch.mean(pred_params, dim=1)





