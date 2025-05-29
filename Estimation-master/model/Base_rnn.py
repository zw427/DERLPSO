from comm_lib.utils import *

class GRU_unit_ori(nn.Module):
    def __init__(self, latent_dim, input_dim,n_units=20,device=torch.device("cpu")):
        super(GRU_unit_ori, self).__init__()


        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim  + input_dim, n_units),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim + input_dim, n_units),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim  + input_dim, n_units),
            nn.Linear(n_units,latent_dim),
            nn.Tanh()
        )

        init_network_weights(self.new_state_net)


    def forward(self, y, x):
        y_concat = torch.cat([y, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y * reset_gate, x], -1)

        new_state= self.new_state_net(concat)


        new_y = (1 - update_gate) * new_state + update_gate * y

        assert (not torch.isnan(new_y).any())


        return new_y


class GRU_unit(nn.Module):
    def __init__(self, latent_dim, input_dim,n_units=100):
        super(GRU_unit, self).__init__()


        self.update_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.update_gate)

        self.reset_gate = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim),
            nn.Sigmoid())
        init_network_weights(self.reset_gate)

        self.new_state_net = nn.Sequential(
            nn.Linear(latent_dim * 2 + input_dim, n_units),
            nn.Tanh(),
            nn.Linear(n_units, latent_dim * 2))
        init_network_weights(self.new_state_net)


    def forward(self, y_mean, y_std, x):
        y_concat = torch.cat([y_mean, y_std, x], -1)

        update_gate = self.update_gate(y_concat)
        reset_gate = self.reset_gate(y_concat)
        concat = torch.cat([y_mean * reset_gate, y_std * reset_gate, x], -1)

        new_state, new_state_std = split_last_dim(self.new_state_net(concat))
        new_state_std = new_state_std.abs()

        new_y = (1 - update_gate) * new_state + update_gate * y_mean
        new_y_std = (1 - update_gate) * new_state_std + update_gate * y_std

        assert (not torch.isnan(new_y).any())

        new_y_std = new_y_std.abs()
        return new_y, new_y_std






