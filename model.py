import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence

from config import Config


class VRNN(nn.Module):
    '''
    Implementation of the VRNN model.
    '''

    def __init__(self, conf: Config) -> None:
        super(VRNN, self).__init__()
        self.conf = conf
        # input features
        self.phi_x = nn.Sequential(
            nn.Linear(in_features=conf.data.x_dim, out_features=conf.vrnn.x_ft),
            nn.ReLU()
        )
        # prior
        self.prior = nn.Sequential(
            nn.Linear(in_features=conf.vrnn.h_dim, out_features=conf.vrnn.h_dim),
            nn.ReLU()
        )
        self.prior_mean = nn.Linear(
            in_features=conf.vrnn.h_dim, out_features=conf.vrnn.z_dim
        )
        self.prior_std = nn.Sequential(
            nn.Linear(in_features=conf.vrnn.h_dim, out_features=conf.vrnn.z_dim),
            nn.Softplus()
        )
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(
                in_features=conf.vrnn.x_ft+conf.vrnn.h_dim,
                out_features=conf.vrnn.h_dim
            ),
            nn.ReLU()
        )
        self.encoder_mean = nn.Linear(
            in_features=conf.vrnn.h_dim, out_features=conf.vrnn.z_dim
        )
        self.encoder_std = nn.Sequential(
            nn.Linear(in_features=conf.vrnn.h_dim, out_features=conf.vrnn.z_dim),
            nn.Softplus()
        )
        # encoding features
        self.phi_z = nn.Sequential(
            nn.Linear(in_features=conf.vrnn.z_dim, out_features=conf.vrnn.z_ft),
            nn.ReLU()
        )
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(
                in_features=conf.vrnn.z_ft + conf.vrnn.h_dim,
                out_features=conf.vrnn.h_dim
            ),
            nn.ReLU()
        )
        self.decoder_mean = nn.Linear(
            in_features=conf.vrnn.h_dim, out_features=conf.data.x_dim
        )
        self.decoder_std = nn.Sequential(
            nn.Linear(in_features=conf.vrnn.h_dim, out_features=conf.data.x_dim),
            nn.Softplus()
        )
        # recurrent model
        self.rnn = nn.GRU(
            input_size=conf.vrnn.x_ft+conf.vrnn.z_ft, hidden_size=conf.vrnn.h_dim
        )
        return
    
    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        device = self.conf.general.device
        latent_timesteps = []
        # sequence losses
        kld_loss_seq = torch.zeros(1).to(device=device)
        nll_loss_seq = torch.zeros(1).to(device=device)
        batch_size = x.shape[0]
        rnn_h = torch.zeros(batch_size, self.conf.vrnn.h_dim).to(device=device)
        # process sequence
        for timestep in range(x.shape[1]):
            x_t = x[:,timestep,:]
            # x features
            phi_x_t = self.phi_x(x_t)
            # prior
            prior_t = self.prior(rnn_h)
            prior_mean_t = self.prior_mean(prior_t)
            prior_std_t = self.prior_std(prior_t)
            prior_normal_t = Normal(loc=prior_mean_t, scale=prior_std_t)
            # encoder
            encoder_t = self.encoder(torch.cat([phi_x_t, rnn_h], dim=1))
            encoder_mean_t = self.encoder_mean(encoder_t)
            encoder_std_t = self.encoder_std(encoder_t)
            encoder_normal_t = Normal(loc=encoder_mean_t, scale=encoder_std_t)
            # latent
            z_t = encoder_normal_t.rsample()
            phi_z_t = self.phi_z(z_t)
            # decoder
            decoder_t = self.decoder(torch.cat([phi_z_t, rnn_h], dim=1))
            decoder_mean_t = self.decoder_mean(decoder_t)
            decoder_std_t = self.decoder_std(decoder_t)
            decoder_normal_t = Normal(loc=decoder_mean_t, scale=decoder_std_t)
            # recurrent model
            rnn_in = torch.cat([phi_x_t, phi_z_t], dim=1)
            # PyTorch recurrent models work on sequences, thus each step
            # is converted to sequence of length 1.
            rnn_in = rnn_in.unsqueeze(dim=0)
            rnn_out, rnn_h_t = self.rnn(rnn_in, rnn_h.unsqueeze(dim=0))
            rnn_h = rnn_h_t.squeeze(dim=0)
            # loss
            # kld
            kld = kl_divergence(p=encoder_normal_t, q=prior_normal_t)
            kld_loss = torch.mean(torch.sum(kld, dim=1))
            kld_loss_seq += kld_loss
            # log prob
            lp = decoder_normal_t.log_prob(x_t)
            nll_loss = - torch.mean(torch.sum(lp, dim=1))
            nll_loss_seq += nll_loss
            # keep latent step
            latent_timesteps.append(z_t)
        
        latent_seq = torch.stack(latent_timesteps)
        # change batch - sequence order
        latent_seq = latent_seq.permute((1, 0, 2))
        kld_loss_seq = kld_loss_seq.squeeze(dim=0)
        nll_loss_seq = nll_loss_seq.squeeze(dim=0)
        return latent_seq, kld_loss_seq, nll_loss_seq
