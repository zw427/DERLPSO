from comm_lib import utils
from comm_lib.utils import *
from model.Base_rnn import GRU_unit_ori, GRU_unit


## for param estimation model

### encoder part

class RNN(nn.Module):

    def __init__(self, latent_dim, input_dim,GRU_update=None,
                 n_gru_units=20,
                 device=torch.device("cpu")):

        super(RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_unit_ori(latent_dim,input_dim, n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_update
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device



    def forward(self, data, time_steps):

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())
        last_yi,latent_ys= self.run_odernn(data, time_steps)


        return last_yi,latent_ys

    def run_odernn(self, data, time_steps, save_info=False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj, n_tp, n_dims = data.size()

        device =get_device(data)
        # init data
        prev_y=torch.zeros((n_traj,self.latent_dim),dtype=torch.float64).to(device)
       # prev_y=self.transform(data.flatten(start_dim=1,end_dim=2))


        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []
        time_points_iter = range(0, len(time_steps))


        for i in time_points_iter:

            xi = data[:, i, :]


            yi= self.GRU_update(prev_y,  xi)


            prev_y = yi


            latent_ys.append(yi)



        latent_ys = torch.stack(latent_ys, 1)

        assert (not torch.isnan(yi).any())
        return yi,latent_ys


class ODE_RNN(nn.Module):

    def __init__(self, latent_dim, input_dim, z0_diffeq_solver=None, GRU_update=None,
                 n_gru_units=100,
                 device=torch.device("cpu")):

        super(ODE_RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_unit_ori(latent_dim,input_dim, n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

    def forward(self, data, time_steps):

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        last_yi,latent_yi= self.run_odernn(data, time_steps)

        return last_yi,latent_yi

    def run_odernn(self, data, time_steps):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj= data.shape[0]

        device = get_device(data)
        # init data
        prev_y = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(device)



        prev_t, t_i = time_steps[0], time_steps[1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []
        time_points_iter = range(1, len(time_steps))

        for i in time_points_iter:
            if abs(prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i)).to(device)
                # dt*f'
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert (not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert (not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

                time_points = linspace_vector(prev_t, t_i, n_intermediate_tp).to(device)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0] - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, 0, :] - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1]
            xi = data[:, i, :]

            yi = self.GRU_update(yi_ode, xi)


            prev_y= yi
            prev_t, t_i = time_steps[i], time_steps[i - 1]

            latent_ys.append(yi)



        latent_ys = torch.stack(latent_ys, 1)

        assert (not torch.isnan(yi).any())

        return yi, latent_ys

class VAE_ODE_RNN(nn.Module):

    def __init__(self, latent_dim, input_dim, z0_diffeq_solver=None, GRU_update=None,
                 n_gru_units=100,
                 device=torch.device("cpu")):

        super(VAE_ODE_RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim,input_dim, n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_update

        self.z0_diffeq_solver = z0_diffeq_solver
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device
        self.extra_info = None

    def forward(self, data, time_steps):

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())



        h_mu,h_std= self.run_odernn(data, time_steps)



        return h_mu,h_std

    def run_odernn(self, data, time_steps, save_info=False):

        n_traj, n_tp, n_dims = data.size()
        device=get_device(data)


        # init data
        prev_y=torch.zeros((n_traj,self.latent_dim),dtype=torch.float64).to(device)
        prev_logvar = torch.zeros((n_traj,self.latent_dim),dtype=torch.float64).to(device)

        #  prev_y=self.prev_y
        #  prev_logvar=self.prev_logvar

        prev_t, t_i = time_steps[0], time_steps[1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        latent_ys = []
        time_points_iter = range(1, len(time_steps))

        for i in time_points_iter:
            if abs(prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i)).to(device)
                # dt*f'
                inc = self.z0_diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert (not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(device)

                assert (not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp).to(device)
                ode_sol = self.z0_diffeq_solver(prev_y, time_points)

                assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0] - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, 0, :] - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1]
            xi = data[:, i, :]

            yi, yi_logvar = self.GRU_update(yi_ode, prev_logvar, xi)

            prev_y, prev_logvar = yi, yi_logvar
            prev_t, t_i = time_steps[i], time_steps[i - 1]

            latent_ys.append(yi)



        latent_ys = torch.stack(latent_ys, 1)

        assert (not torch.isnan(yi).any())
        assert (not torch.isnan(yi_logvar).any())

        return yi, yi_logvar
class VAE_RNN(nn.Module):

    def __init__(self, latent_dim, input_dim,GRU_update=None,
                 n_gru_units=20,
                 device=torch.device("cpu")):

        super(VAE_RNN, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_unit(latent_dim,input_dim, n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_update
        self.latent_dim = latent_dim
        self.input_dim = input_dim
        self.device = device



    def forward(self, data, time_steps):

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())
        h_mu,h_std= self.run_odernn(data, time_steps)


        return h_mu,h_std

    def run_odernn(self, data, time_steps, save_info=False):
        # IMPORTANT: assumes that 'data' already has mask concatenated to it

        n_traj, n_tp, n_dims = data.size()


        device = self.device
        # init data
        prev_y=torch.zeros((n_traj,self.latent_dim),dtype=torch.float64).to(device)
        prev_logvar = torch.zeros((n_traj, self.latent_dim), dtype=torch.float64).to(device)
       #rev_y=self.transform(data.flatten(start_dim=1,end_dim=2))


        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        time_points_iter = range(0, len(time_steps))


        for i in time_points_iter:

            xi = data[:, i, :]


            yi,logvar= self.GRU_update(prev_y,prev_logvar,xi)


            prev_y = yi
            prev_logvar=logvar


        assert (not torch.isnan(yi).any())
        assert (not torch.isnan(logvar).any())
        return yi,logvar
class Conv_net(nn.Module):
    def __init__(self, input_dim,conv_sizes,hidden_dims, act_func,normal=False):
        super(Conv_net, self).__init__()

        conv_modules = []
        conv_kernel = 3
        conv_strides = 2
        pool_fn = nn.AvgPool1d  # AveragePooling1D or MaxPooling1D
        pool_size = 2
        pool_strides = 2
        pool_padding = 'valid'  # 'valid' or 'same'
        N = input_dim[0]
        in_channel = input_dim[-1]
        #print("in channel",in_channel)
        out_channel = 1
        for i, filter_size in enumerate(conv_sizes):
            if 0 < i: conv_modules.append(pool_fn(pool_size, pool_strides, pool_padding))
           # if i == 0: in_channel = input_dim[-1]
            conv_modules.append(nn.Conv1d(in_channel, filter_size, conv_kernel, conv_strides, padding=0))
            in_channel = filter_size
            conv_modules.append(act_func)
            N = int((N - conv_kernel) / conv_strides) + 1
            out_channel = filter_size
        self.conv_modules=nn.Sequential(*conv_modules)

        dense_modules=[]
        flatten = int(N * out_channel)

        # create dense layers
        input_dim = flatten
        print("input_dim",input_dim)
        output_dim = hidden_dims[-1]

        for h_dim in hidden_dims[0:-1]:
            dense_modules.append(
                nn.Sequential(
                    nn.Linear(input_dim, h_dim, bias=False),
                    act_func,

                )
            )
            input_dim = h_dim
        dense_modules.append(nn.Sequential(nn.Linear(input_dim, output_dim, bias=False)))
        if normal:
            dense_modules.append(nn.Sigmoid())

        self.dense_modules=nn.Sequential(*dense_modules)



    def forward(self, data,time):
        conv_param=self.conv_modules(data.permute(0,2,1))
        input=conv_param.flatten(start_dim=1,end_dim=-1)
        pred_param=self.dense_modules(input)
        return pred_param


class MLP(nn.Module):

    def __init__(self,hidden_dims,nonlinearity):
        #super().__init__()
        super(MLP, self).__init__()
        self.model=build_modules(hidden_dims,nonlinearity)



    def forward(self, data,time):

        data=data.flatten(start_dim=1,end_dim=len(data.shape)-1)
        param=self.model(data)
        return param


## decoder part for vae

class model_rnn(nn.Module):

    def __init__(self, input_dim_param,output_dim_param,dim_of_data, latent_dim,GRU_update=None,
                 n_gru_units=100,
                 device=torch.device("cpu")):

        super(model_rnn, self).__init__()

        if GRU_update is None:
            self.GRU_update = GRU_unit_ori(input_dim_param,dim_of_data, n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_update
        self.dim_of_data= dim_of_data

        self.device = device
        self.extra_info = None

        self.tranform=nn.Sequential(
            nn.Linear(latent_dim,2*latent_dim),
            nn.ELU(),
            nn.Linear(2*latent_dim,output_dim_param)
        )



    def forward(self, data, time_steps,prior):

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())


        y_param= self.run_rnn(data, time_steps,prior)



        return y_param

    def run_rnn(self, data, time_steps,prior=None):
        prev = prior
        if prior is None:
            prev=torch.zeros((self.data.shape[0], self.latent_dim), type=torch.float64).to(self.device)


        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())


        time_points_iter = range(0, len(time_steps))


        for i in time_points_iter:
            xi = data[:, i, :]
            yi= self.GRU_update(prev, xi)
            prev=yi

        return prev




class model_mlp(nn.Module):
    def __init__(self, model):
        super(model_mlp, self).__init__()

        self.model_param =model



    def forward(self, data,time):
        return self.model_param(data)


class model_ode_rnn(nn.Module):

    def __init__(self, input_dim_param,dim_of_data, latent_dim, diffeq_solver,GRU_update=None,
                 n_gru_units=100,
                 device=torch.device("cpu")):

        super(model_ode_rnn, self).__init__()
        #
        if GRU_update is None:
            self.GRU_update = GRU_unit_ori(input_dim_param, dim_of_data, n_units=n_gru_units).to(device)
        else:
            self.GRU_update = GRU_update
        self.dim_of_data = dim_of_data

        self.device = device
        self.diffeq_solver=diffeq_solver
        self.latent_dim=latent_dim

    def forward(self, data, time_steps,prior):

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())


        y_param= self.run_odernn(data, time_steps,prior)




        return y_param

    def run_odernn(self, data, time_steps,prior=None):
        prev_y=prior
        if prior is None:
            prev_y=torch.zeros((data.shape[0],self.latent_dim),dtype=torch.float64).to(get_device(data))

        prev_t, t_i = time_steps[0], time_steps[1]

        interval_length = time_steps[-1] - time_steps[0]
        minimum_step = interval_length / 50

        assert (not torch.isnan(data).any())
        assert (not torch.isnan(time_steps).any())

        time_points_iter = range(1, len(time_steps))

        for i in time_points_iter:
            if abs(prev_t - t_i) < minimum_step:
                time_points = torch.stack((prev_t, t_i)).to(self.device)
                # dt*f'
                inc = self.diffeq_solver.ode_func(prev_t, prev_y) * (t_i - prev_t)

                assert (not torch.isnan(inc).any())

                ode_sol = prev_y + inc
                ode_sol = torch.stack((prev_y, ode_sol), 2).to(self.device)

                assert (not torch.isnan(ode_sol).any())
            else:
                n_intermediate_tp = max(2, ((t_i - prev_t) / minimum_step).int())

                time_points = utils.linspace_vector(prev_t, t_i, n_intermediate_tp).to(get_device(data))
                ode_sol = self.diffeq_solver(prev_y, time_points)

                assert (not torch.isnan(ode_sol).any())

            if torch.mean(ode_sol[:, :, 0] - prev_y) >= 0.001:
                print("Error: first point of the ODE is not equal to initial value")
                print(torch.mean(ode_sol[:, 0, :] - prev_y))
                exit()

            yi_ode = ode_sol[:, :, -1]
            xi = data[:, i, :]

            yi= self.GRU_update(yi_ode, xi)


            prev_y= yi
            prev_t, t_i = time_steps[i], time_steps[i - 1]


        return prev_y