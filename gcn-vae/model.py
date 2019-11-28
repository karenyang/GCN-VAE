import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from layers import RelGraphConv, EmbeddingLayer, Conv1d, RefineLatentFeatLayer, RefineRelationAdjLayer
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import numpy as np


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
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.build_encoder()
        self.build_decoder()
        self.loss_func = nn.CrossEntropyLoss()

    def build_encoder(self):
        self.input_layer = EmbeddingLayer(self.num_nodes, self.h_dim)
        self.rconv_layer_1 = self.build_rconv_layer(activation=nn.ReLU())
        self.rconv_layer_2 = self.build_rconv_layer(activation=nn.Sigmoid())
        self.z_mean_branch = nn.Linear(self.h_dim, self.h_dim)
        self.z_log_std_branch = nn.Linear(self.h_dim, self.h_dim)

    def build_rconv_layer(self, activation):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=activation, self_loop=True,
                            dropout=self.dropout)

    def get_z(self, z_mean, z_log_std, prior="normal"):
        if prior == 'normal':
            return normal.Normal(z_mean, z_log_std).sample()
        else:
            # TODO: Flow transforms
            return None

    def encoder(self, g, h, r, norm):
        h = self.input_layer(h)
        self.x_cache = h
        h = self.rconv_layer_1(g, h, r, norm)
        h = self.rconv_layer_2(g, h, r, norm)
        self.h_cache = h
        # z_mean and z_sigma -> latent distribution(maybe with flow transform) -> samples
        self.z_mean = self.z_mean_branch(h)
        self.z_log_std = self.z_log_std_branch(h)
        z = self.get_z(self.z_mean, self.z_log_std, prior='normal')
        return z

    def build_decoder(self):
        self.refine_relational_adj_0 = RefineRelationAdjLayer(act=nn.Sigmoid(), normalize=True, non_negative=False)
        self.update_recon_features1 = RefineLatentFeatLayer(input_dim=self.h_dim, out_dim=self.h_dim)
        self.refine_relational_adj_1 = RefineRelationAdjLayer(act=nn.Sigmoid(), normalize=True, non_negative=False)
        self.conv1d_1 = Conv1d(in_channels=1, out_channels=self.num_bases, kernel_size=1, bias=True)
        self.update_recon_features2 = RefineLatentFeatLayer(self.h_dim, self.h_dim)
        self.refine_relational_adj_2 = RefineRelationAdjLayer(act=nn.Sigmoid(), normalize=True, non_negative=False)
        self.conv1d_2 = Conv1d(in_channels=self.num_bases, out_channels=self.num_rels, kernel_size=3, bias=True)
        # self.update_recon_features3 = RefineLatentFeatLayer(self.h_dim, self.h_dim)
        # self.refine_relational_adj_3 = RefineRelationAdjLayer(act=nn.Sigmoid(), normalize=True, non_negative=True)

    def decoder(self, z, x):
        """
        Reconstructed relational matrix from input structure.
        :param z: encoder output, z samples
        :param x: input node attributes
        :return: reconstructed relational adjacency matrix
        """
        R = self.refine_relational_adj_0(z)  # N x N (N is self.num_nodes)
        h = self.update_recon_features1(R, z, self.x_cache)
        R = self.refine_relational_adj_1(h)
        R = self.conv1d_1(R)  # 1 x N x N -> num_bases x N x N
        h = self.update_recon_features2(R, z, self.h_cache)
        R = self.refine_relational_adj_2(h)
        R = self.conv1d_2(R)  # num_bases x N x N -> num_rel x N x N
        # h = self.update_recon_features3(R, z, None)
        # R = self.refine_relational_adj_3(h)

        return R

    def forward(self, g, h, r, norm):
        self.node_id = h
        z = self.encoder(g, h, r, norm)
        recon = self.decoder(z, h)
        return recon

    def sample(self):
        z = normal.Normal(0, 1)([self.num_nodes, self.out_dim])
        reconstruction = self.decoder(z)
        reconstruction = reconstruction.reshape(self.num_nodes, self.num_nodes, self.num_rels)
        return reconstruction

    def get_loss(self, recon, pos_samples, neg_samples):
        """
        preds will be the decoder output
        Labels will be the graph
        output reconstruction error.
        :param preds:
        :param labels:
        :return: recon + kl
        """
        pos_s, pos_r, pos_o = zip(*pos_samples)
        neg_s, neg_r, neg_o = zip(*neg_samples)
        score_pos = recon[:, pos_s, pos_o]  # filter all the positive sample edge indices
        score_pos = score_pos.view(-1, self.num_rels)  # shape into n_edges x num_rel
        score_neg = recon[:, neg_s, neg_o]
        score_neg = score_neg.view(-1, self.num_rels)
        scores = torch.cat([score_pos, score_neg], dim=0)
        targets = torch.cat([torch.LongTensor(pos_r) + 1, torch.zeros(len(neg_r), dtype=torch.long)], dim=0).cuda() # make  np-relation as class 0.
        recon_loss = self.loss_func(scores, targets)
        kl = 0.5 / recon.shape[1] * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
        loss = recon_loss + kl
        preds = scores.argmax(dim=1)
        accuracy = (preds == targets).sum().float() / targets.numel()
        f1 = f1_score(targets.cpu(), preds.cpu(), average='macro', labels=np.arange(self.num_rels))
        return loss, recon_loss, kl, accuracy.item(), f1
