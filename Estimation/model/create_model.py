from comm_lib import utils
from model import Base_param
from model import param_model
from model.ODENet.diffeq_solver import *
from model.ODENet.ode_func import *


def create_model(configs,num_of_param,dim_of_data,time_points,model_type):
    type = model_type.lower()
    device = torch.device(configs['device'] if torch.cuda.is_available() else 'cpu')
    cfg=configs['Net']
    latent_dim = cfg['latent_dim']

    if type =='vae':
            encoder=None
            d_input_dim, d_output_dim = latent_dim, num_of_param
            hidden_dims = list(cfg['decoder']['model_hidden_size'])
            hidden_dims.insert(0, d_input_dim)
            hidden_dims.append(d_output_dim)
            act_func = cfg['decoder']['model_nonlinearity']
            mlp = utils.build_modules(hidden_dims, act_func,False)
            if cfg['encoder']['type']=='ODE':

                ode_func_net = utils.create_net(latent_dim,
                                                latent_dim,
                                                n_layers=cfg['encoder']["ode_layers"],
                                                n_units=cfg['encoder']['ode_unit'],
                                                nonlinear=nn.ELU, drop=False)
                ode_func = ODEFunc(
                    input_dim=latent_dim,
                    ode_func_net=ode_func_net,
                    device=device).to(device)
                diffeq_solver = DiffeqSolver(ode_func, "euler", odeint_rtol = 1e-3, odeint_atol = 1e-5, device = device)

                encoder=param_model.VAE_ODE_RNN(latent_dim,dim_of_data,diffeq_solver,n_gru_units = cfg['encoder']["gru_unit"], device = device).to(device)

            elif cfg['encoder']['type']=='RNN':

                    encoder = param_model.VAE_RNN(latent_dim, dim_of_data, GRU_update=None, n_gru_units=cfg['encoder']["gru_unit"], device=device).to(device)

            if cfg['decoder']['type']=='ODE':
                d_input=d_output_dim
                ode_func_net = utils.create_net(d_input,
                                                d_input,
                                                n_layers=cfg['decoder']["ode_layers"],
                                                n_units=cfg['decoder']['ode_unit'],
                                                nonlinear=nn.ELU, drop=False)
                ode_func = ODEFunc(
                    input_dim=d_input,
                    ode_func_net=ode_func_net,
                    device=device).to(device)
                diffeq_solver = DiffeqSolver(ode_func, "euler", odeint_rtol=1e-3, odeint_atol=1e-5, device=device)

                decoder=param_model.model_ode_rnn(d_output_dim,dim_of_data, d_output_dim, diffeq_solver,
                                                  n_gru_units=cfg['decoder']["gru_unit"], device=device).to(device)

            elif cfg['decoder']['type']=='RNN':



                decoder=param_model.RNN(d_output_dim, d_output_dim, GRU_update=None,
                                          n_gru_units=cfg['decoder']["gru_unit"], device=device)





            model=Base_param.VAE(encoder,mlp,decoder)

    elif type=='ode_rnn':
        latent_dim = dim_of_data
        ode_func_net = utils.create_net(latent_dim,
                                        latent_dim,
                                        n_layers=cfg["ode_layers"],
                                        n_units=cfg['ode_unit'],
                                        nonlinear=nn.ELU, drop=False).to(device)
        ode_func = ODEFunc(
            input_dim=latent_dim,
            ode_func_net=ode_func_net,
            device=device).to(device)
        diffeq_solver = DiffeqSolver(ode_func, "euler", odeint_rtol=1e-6, odeint_atol=1e-8, device=device)

        model2_input_dim = time_points * dim_of_data+latent_dim
        model2_out_dim = num_of_param
        hidden_dims = list([5,5])
        hidden_dims.insert(0, model2_input_dim)
        hidden_dims.append(model2_out_dim)
        model2 = utils.build_modules(hidden_dims, 'elu',False)

        encoder=param_model.ODE_RNN(latent_dim,dim_of_data,diffeq_solver,GRU_update=None,n_gru_units=20,device=device)

        decoder = param_model.model_mlp(model2)

        model = Base_param.Encoder_Decoder(encoder, decoder)
    elif type=='mlp':

        input_dim, output_dim=time_points * dim_of_data, num_of_param
        hidden_dims=list(cfg['model_hidden_size'])
        hidden_dims.insert(0,input_dim)
        hidden_dims.append(output_dim)
        act_func= cfg['model_nonlinearity']
        mlp=param_model.MLP(hidden_dims,act_func)
        model=Base_param.Base(mlp)

    elif type=='rnn':
        latent_dim = num_of_param
        # + configs['Data']['num_of_sigma'] + configs['Data']['num_of_rho']
        encoder =  param_model.RNN(latent_dim, dim_of_data, GRU_update=None, n_gru_units=cfg["gru_unit"], device=device)
        model2_input_dim=latent_dim+time_points*dim_of_data
        model2_out_dim=num_of_param
        hidden_dims = list([5,5])
        hidden_dims.insert(0, model2_input_dim)
        hidden_dims.append(model2_out_dim)
        model2=utils.build_modules(hidden_dims,'elu')
        decoder=param_model.model_mlp(model2)


        model = Base_param.Encoder_Decoder(encoder, decoder)

    return model
#
