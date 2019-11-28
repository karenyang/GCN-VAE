import torch.nn as nn
import torch
from dgl.nn.pytorch import RelGraphConv
import torch.nn.functional as F
from torch.distributions import normal



class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False,
                 vae=False):
        super(BaseRGCN, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        self.vae = vae

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        # i2h
        i2h = self.build_input_layer()
        if i2h is not None:
            self.layers.append(i2h)
        # h2h
        for idx in range(self.num_hidden_layers):
            h2h = self.build_hidden_layer(idx)
            self.layers.append(h2h)

        if self.vae:
            # h2z
            self.z_layer = self.build_z_layer()
            # z2R
            self.recon_layer = self.build_reconstruct_layer()
            # R2o
            self.output_layer = self.build_output_layer()


    def get_kl(self):
        return torch.Tensor([0]).cuda()

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_z_layer(self):
        return None

    def build_reconstruct_layer(self):
        raise NotImplementedError

    def build_output_layer(self):
        raise NotImplementedError

    def forward(self, g, h, r, norm):
        for i in range(len(self.layers)):
            layer = self.layers[i]
            h = layer(g, h, r, norm)
            if i ==0 :
                self.input_embed = h
        if not self.vae:
            return h
        h = self.z_layer(h)
        self.z_mean, self.z_log_std = torch.chunk(h, 2, dim=-1)
        z = normal.Normal(self.z_mean, self.z_log_std.exp()).sample()
        R = self.recon_layer(z)
        h = self.output_layer(R, z, self.input_embed)
        return h

class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = torch.nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)


class ReconstructLayer(nn.Module):
    def __init__(self, act=None, normalize=True, non_negative=False, **kwargs):
        super(ReconstructLayer, self).__init__()
        self.non_negative = non_negative
        self.normalize = normalize
        self.act = act

    def forward(self, z):
        # import ipdb; ipdb.set_trace()
        R = z.matmul(z.transpose(-1, -2))
        if self.non_negative:
            R = R + torch.ones_like(R)
        if self.normalize:
            d = R.sum(1).pow(-0.5)
            R = d.unsqueeze(0) * R * d.unsqueeze(1)
        if self.act is not None:
            R = self.act(R)
        return R


class UpdateLayer(nn.Module):
    def __init__(self, input_dim, out_dim, act=None, **kwargs):
        super(UpdateLayer, self).__init__()
        self.act = act
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, R, z, x=None):
        # import ipdb;
        # ipdb.set_trace()
        z = self.linear(z)
        z = R.matmul(z)
        if x is not None:
            z = z + R.matmul(x)
        if self.act:
            z = self.act(z)
        return z

class KGVAE(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)

    def build_z_layer(self):
        return nn.Linear(self.h_dim, 2*self.h_dim)

    def build_reconstruct_layer(self):
        return ReconstructLayer(act=nn.Sigmoid(), normalize=False, non_negative=True)

    def build_output_layer(self):
        return UpdateLayer(input_dim=self.h_dim,out_dim=self.h_dim, act=nn.Sigmoid())

    def get_kl(self):
        kl = 0.5 / self.z_mean.shape[0] * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
        return kl