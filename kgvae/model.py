import torch.nn as nn
import torch
import torch.nn.functional as F
from dgl.nn.pytorch import RelGraphConv
import numpy as np
import utils
import random
from flow_network import MADE, PermuteLayer

torch.autograd.set_detect_anomaly(True)


class KGVAE(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0,
                 use_self_loop=False, use_cuda=True,
                 k=10, n_flows=0):
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
        print("use cuda", self.use_cuda)
        self.k = k  # mixture of gaussian
        print(f"Mixture of {self.k} Gaussians.")
        self.flow_log_prob = None
        self.build_encoder()

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.h_dim) / np.sqrt(self.k * self.h_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)
        # Flow transform
        self.n_flows = n_flows
        if self.n_flows > 0:
            print(f"Sequence of {self.n_flows} Flow transforms.")
            self.build_iaf()

    def build_iaf(self):
        nf_blocks = []
        for i in range(self.n_flows):
            nf_blocks.append(
                MADE(self.h_dim, self.h_dim, self.n_flows))
            nf_blocks.append(PermuteLayer(self.h_dim))  # permute dims
        self.nf = nn.Sequential(*nf_blocks)

    def build_encoder(self):
        self.input_layer = EmbeddingLayer(self.num_nodes, self.h_dim)
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
        x = utils.sample_gaussian(m, v)
        if self.n_flows > 0:
            for flow in self.nf[::-1]:
                x, _ = flow.inverse(x)
        return x

    def compute_kernel(self, x, y):
        x_size = x.size(0)
        y_size = y.size(0)
        dim = x.size(1)
        x = x.unsqueeze(1)  # (x_size, 1, dim)
        y = y.unsqueeze(0)  # (1, y_size, dim)
        tiled_x = x.expand(x_size, y_size, dim)
        tiled_y = y.expand(x_size, y_size, dim)
        kernel_input = (tiled_x - tiled_y).pow(2).mean(2) / float(dim)
        return torch.exp(-kernel_input)  # (x_size, y_size)

    def get_kl(self, z):
        m_mixture, z_mixture = utils.gaussian_parameters(self.z_pre, dim=1)
        m = self.z_mean
        v = self.z_sigma
        kl = torch.mean(utils.log_normal(z, m, v) + self.flow_log_prob - utils.log_normal_mixture(z, m_mixture, z_mixture))
        return kl

    def get_mmd(self, z):
        m_mixture, s_mixture = utils.gaussian_parameters(self.z_pre, dim=1)
        num_sample = 200
        z_pri = utils.sample_gaussian(m_mixture, s_mixture, repeat=num_sample // self.k)
        if self.n_flows > 0:
            for flow in self.nf:
                z_pri, _ = flow.forward(z_pri)
        # (Monte Carlo) sample the z otherwise GPU will be out of memory
        z_post_samples = z[random.sample(range(z.shape[0]), num_sample)]
        prior_z_kernel = self.compute_kernel(z_pri, z_pri)
        posterior_z_kernel = self.compute_kernel(z_post_samples, z_post_samples)
        mixed_kernel = self.compute_kernel(z_pri, z_post_samples)
        mmd = prior_z_kernel.mean() + posterior_z_kernel.mean() - 2 * mixed_kernel.mean()
        return mmd

    def get_flow_log_prob(self):
        return self.flow_log_prob

    def forward(self, g, h, r, norm):
        self.node_id = h.squeeze()
        h = self.input_layer(g, h, r, norm)
        h = self.rconv_layer_1(g, h, r, norm)
        h = self.rconv_layer_2(g, h, r, norm)
        self.z_mean, self.z_sigma = utils.gaussian_parameters(h)
        z = utils.sample_gaussian(self.z_mean, self.z_sigma)
        # flow transformed samples
        if self.n_flows > 0:
            self.flow_log_prob = torch.zeros(1, )
            log_det_sum = torch.zeros((z.shape[0],))
            for flow in self.nf:
                z, log_det = flow.forward(z)
                if log_det.is_cuda:
                    log_det_sum = log_det_sum.cuda()
                log_det_sum = log_det_sum + log_det.view(-1)
            self.flow_log_prob = torch.mean(log_det_sum.view(-1,1))
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
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, h_dim)

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
