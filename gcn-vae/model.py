import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from layers import RelGraphConv, EmbeddingLayer



class KGVAE(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0, num_encoder_output_layers=2,
                 use_self_loop=False, use_cuda=False):
        super(KGVAE, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.num_encoder_output_layers = num_encoder_output_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.build_encoder()
        self.build_decoder()


    def build_decoder(self):
        pass

    def build_encoder(self):
        self.encoder_module = nn.ModuleList()
        # i2h
        self.encoder_module.append(EmbeddingLayer(self.num_nodes, self.h_dim))
        # h2h
        common_rconv_layer = self.build_rconv_layer(activation=True)
        self.encoder_module.append(common_rconv_layer)
        # h2o
        z_mean_branch = self.build_rconv_layer()
        z_log_std_branch = self.build_rconv_layer()
        self.encoder_module.append(z_mean_branch)
        self.encoder_module.append(z_log_std_branch)

    def build_rconv_layer(self, activation=False):
        act = F.relu if activation else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)

    def get_z(self, z_mean, z_log_std, prior="normal"):
        if prior == 'normal':
           return normal.Normal(z_mean, z_log_std).sample()
        else:
            # TODO: Flow transforms
            return None

    def encoder(self, g, h, r, norm):
        for layer in self.encoder_module[:-2]:
            h = layer(g, h, r, norm)
        # z_mean and z_sigma -> latent distribution(maybe with flow transform) -> samples
        z_mean = self.encoder_module[-2](g, h, r, norm)
        z_log_std = self.encoder_module[-1](g, h, r, norm)
        z = self.get_z(z_mean, z_log_std, prior='normal')
        return z

    def forward(self,g, h, r, norm):
        return self.encoder(g, h, r, norm)
