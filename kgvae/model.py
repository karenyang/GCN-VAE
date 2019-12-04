import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
import numpy as np
import utils


class KGVAE(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False,
                 k=1, gamma=12, epsilon=2):
        super(KGVAE, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = None if num_bases < 0 else num_bases
        self.num_hidden_layers = num_hidden_layers
        self.dropout = dropout
        self.use_self_loop = use_self_loop
        self.use_cuda = use_cuda
        # RotateE initiation params
        self.gamma = gamma
        self.epsilon = epsilon
        # Mixture of Gaussians parameters
        self.k = k
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.h_dim) / np.sqrt(self.k * self.h_dim))
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

        # Build model
        self.build_encoder()

    def build_encoder(self):
        self.input_layer = EmbeddingLayer(self.num_nodes, self.h_dim, self.gamma, self.epsilon)
        self.rconv_layer_1 = RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                                          self.num_bases, activation=nn.ReLU(), self_loop=True,
                                          dropout=self.dropout)
        self.rconv_layer_2 = RelGraphConv(self.h_dim, self.h_dim * 2, self.num_rels, "bdd",
                                          self.num_bases, activation=lambda x: x, self_loop=True,
                                          dropout=self.dropout)

    def sample_z(self, batch):
        m, v = utils.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return utils.sample_gaussian(m, v)

    def get_kl(self, z):
        m_mixture, z_mixture = utils.gaussian_parameters(self.z_pre, dim=1)
        m = self.z_mean
        v = self.z_sigma
        kl = torch.mean(utils.log_normal(z, m, v) - utils.log_normal_mixture(z, m_mixture, z_mixture))
        return kl

    def get_mmd(self, z):
        m_mixture, z_mixture = utils.gaussian_parameters(self.z_pre, dim=1)
        m = self.z_mean
        v = self.z_sigma
        kl = torch.mean(utils.log_normal(z, m, v) - utils.log_normal_mixture(z, m_mixture, z_mixture))
        return kl

    def forward(self, g, h, r, norm):
        self.node_id = h.squeeze()
        h = self.input_layer(g, h, r, norm)
        h = self.rconv_layer_1(g, h, r, norm)
        h = self.rconv_layer_2(g, h, r, norm)
        self.z_mean, self.z_sigma = utils.gaussian_parameters(h)
        z = utils.sample_gaussian(self.z_mean, self.z_sigma)
        return z


#######################################################################
#
# Baseline - RGCN
#
#######################################################################


class BaseRGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=False):
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
        # h2o
        h2o = self.build_output_layer()
        if h2o is not None:
            self.layers.append(h2o)

    def build_input_layer(self):
        return None

    def build_hidden_layer(self, idx):
        raise NotImplementedError

    def build_output_layer(self):
        return None

    def forward(self, g, h, r, norm):
        for layer in self.layers:
            h = layer(g, h, r, norm)
        return h

    def get_kl(self, z):
        return torch.Tensor([0]).cuda()


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim, gamma=0, epsilon=0):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, h_dim)
        if gamma != 0:
            nn.init.uniform_(
                tensor=self.embedding.weight,
                a=-(gamma + epsilon) / (h_dim / 2),
                b=(gamma + epsilon) / (h_dim / 2)
            )

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class DistLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(DistLayer, self).__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, g, h, r, norm):
        return self.linear(h.squeeze())


class RGCN(BaseRGCN):
    def build_input_layer(self):
        return EmbeddingLayer(self.num_nodes, self.h_dim)

    def build_hidden_layer(self, idx):
        act = F.relu if idx < self.num_hidden_layers - 1 else None
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "bdd",
                            self.num_bases, activation=act, self_loop=True,
                            dropout=self.dropout)
