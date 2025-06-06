from  comm_lib import  utils
from comm_lib.likelihood_eval import *


### base line for Distribution estimation
class Baseline(nn.Module):
    def __init__(self):

        super(Baseline, self).__init__()

    def compute(self, data,time):

        info = self.get_reconstruction(data,time)

        pred_param= info["pred"]

        return pred_param


class Encoder_Decoder(Baseline):
    def __init__(self,encoder, decoder=None):
        super(Encoder_Decoder, self).__init__(
        )

        self.encoder = encoder
        self.decoder=decoder


    def get_reconstruction(self,  truth, truth_time_steps):


        h_mu, _= self.encoder(truth, truth_time_steps)



        truth = torch.cat((truth.flatten(1, 2), h_mu), dim=-1)
        pred_param=self.decoder(truth,truth_time_steps)




        all_extra_info = {
            "pred": (pred_param),

        }

        return all_extra_info

class VAE(nn.Module):
    def __init__(self,encoder, transform=None,decoder=None):
        super(VAE, self).__init__(
        )

        self.encoder = encoder
        self.transform=transform
        self.decoder=decoder

    def compute(self, data,time):
        info = self.get_reconstruction(data,time)

        pred_param = info["pred"]
        mu=info["mu"]
        std=info["std"]
        return pred_param,mu,std

    def get_reconstruction(self,  truth, truth_time_steps):
      #  data=truth.flatten(start_dim=1,end_dim=2)

        h_mu, h_std= self.encoder(truth, truth_time_steps)

        #sample from distribution
        z=utils.sample_standard_gaussian(h_mu,h_std)

        prior=self.transform(z)
        pred_param=self.decoder(truth,truth_time_steps,prior)

        all_extra_info = {
            "pred": (pred_param),
            "mu":h_mu,
            "std":h_std

        }

        return all_extra_info

class Base(Baseline):
    def __init__(self, model):
        super(Baseline, self).__init__()

        self.model=model


    def get_reconstruction(self,  truth, truth_time_steps):

        pred_param= self.model(truth, truth_time_steps)


        all_extra_info = {
            "pred": (pred_param),

        }

        return all_extra_info





