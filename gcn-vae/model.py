import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from layers import RelGraphConv, EmbeddingLayer, Conv1d, RefineLatentFeatLayer, RefineRelationAdjLayer
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
import numpy as np
import utils

class RGCN(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels, num_bases,
                 num_hidden_layers=1, dropout=0, num_encoder_output_layers=2,
                 use_self_loop=False, use_cuda=False,reg_param=0):
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
        self.build_module()
        self.reg_param = reg_param

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def build_module(self):
        self.input_layer = EmbeddingLayer(self.num_nodes, self.h_dim)
        self.rconv_layer_1 = self.build_rconv_layer(activation=nn.ReLU())
        self.rconv_layer_2 = self.build_rconv_layer(activation=nn.Sigmoid())
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))


    def build_rconv_layer(self, activation):
        return RelGraphConv(self.h_dim, self.h_dim, self.num_rels, "basis",
                            self.num_bases, activation=activation, self_loop=True,
                            dropout=self.dropout)

    def forward(self, g, h, r, norm):
        h = self.input_layer(h)
        self.x_cache = h
        h = self.rconv_layer_1(g, h, r, norm)
        h = self.rconv_layer_2(g, h, r, norm)
        return h

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))


    def get_loss(self, recon, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(recon, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(recon)
        mrr = utils.calc_mrr(embed, self.w_relation, )
        return predict_loss + self.reg_param * reg_loss, mrr


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
        score_pos = torch.sigmoid(recon[pos_r, pos_s, pos_o])
        score_neg = torch.sigmoid(recon[neg_r, neg_s, neg_o])
        scores = torch.cat([score_pos, score_neg])
        targets = torch.cat([torch.ones(len(pos_o)), torch.zeros(len(neg_o))], dim=0).cuda()
        # import ipdb;ipdb.set_trace()
        recon_loss = F.binary_cross_entropy_with_logits(scores,targets)
        kl = 0.5 / recon.shape[1] * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
        loss = recon_loss + kl

        roc = roc_auc_score(targets.detach().cpu().numpy().astype(np.int), scores.detach().cpu().numpy())
        ap = average_precision_score(targets.detach().cpu().numpy().astype(np.int), scores.detach().cpu().numpy())
        # import ipdb;ipdb.set_trace()

        score_mat = torch.sigmoid(recon[pos_r, pos_s,:]).detach().cpu().numpy()
        pos_obj = torch.Tensor(pos_o).cpu().numpy()
        pos_obj = np.array(pos_obj, dtype=np.int)
        raw_ranks = [np.sum(score >= score[e]) for score, e in zip(score_mat, pos_obj)]

        inv_score_mat = torch.sigmoid(recon[pos_r, :, pos_o]).detach().cpu().numpy()
        pos_sub = torch.Tensor(pos_s).cpu().numpy()
        pos_sub = np.array(pos_sub, dtype=np.int)
        inv_raw_ranks = [np.sum(score >= score[e]) for score, e in zip(inv_score_mat, pos_sub)]
        ranks = raw_ranks + inv_raw_ranks

        hit1 = np.sum([1. for rank in ranks if rank <= 1])/len(ranks)
        hit3 = np.sum([1. for rank in ranks if rank <= 3])/len(ranks)
        hit10 = np.sum([1. for rank in ranks if rank <= 10])/len(ranks)
        mr = np.mean(ranks)
        mrr = np.mean(1./np.array(ranks))

        # f1 = f1_score(targets.cpu(), preds.cpu(), average='macro', labels=np.arange(self.num_rels))
        return loss, recon_loss, kl, hit10, mr, mrr, hit1, hit3, hit10



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
            return normal.Normal(z_mean, z_log_std.exp()).sample()
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
        self.conv1d_1 = Conv1d(in_channels=1, out_channels=self.num_bases, kernel_size=5, bias=True)
        self.update_recon_features2 = RefineLatentFeatLayer(self.h_dim, self.h_dim)
        self.refine_relational_adj_2 = RefineRelationAdjLayer(act=nn.Sigmoid(), normalize=True, non_negative=False)
        self.conv1d_2 = Conv1d(in_channels=self.num_bases, out_channels=self.num_rels, kernel_size=5, bias=True)
        # self.update_recon_features3 = RefineLatentFeatLayer(self.h_dim, self.h_dim)
        # self.refine_relational_adj_3 = RefineRelationAdjLayer(act=nn.Sigmoid(), normalize=True, non_negative=True)
        # self.output = nn.Softmax(dim=-1)

    def decoder(self, z):
        """
        Reconstructed relational matrix from input structure.
        :param z: encoder output, z samples
        :param x: input node attributes
        :return: reconstructed relational adjacency matrix
        """
        R = self.refine_relational_adj_0(z)  # 1 x N x N (N is self.num_nodes)
        # h = self.update_recon_features1(R, z, self.x_cache)
        # R = self.refine_relational_adj_1(h)
        R = self.conv1d_1(R)  # 1 x N x N -> num_bases x N x N
        # h = self.update_recon_features2(R, z, self.h_cache)
        # R = self.refine_relational_adj_2(h)
        R = self.conv1d_2(R)  # num_bases x N x N -> num_rel x N x N
        # R = self.output(R)
        # h = self.update_recon_features3(R, z, None)
        # R = self.refine_relational_adj_3(h)

        return R

    def forward(self, g, h, r, norm):
        self.node_id = h
        z = self.encoder(g, h, r, norm)
        recon = self.decoder(z)
        return recon

    def sample(self):
        z = normal.Normal(0, 1)([self.num_nodes, self.out_dim])
        reconstruction = self.decoder(z)
        reconstruction = reconstruction.reshape(self.num_nodes, self.num_nodes, self.num_rels)
        return reconstruction

    # def get_loss(self, recon, pos_samples, neg_samples):
    #     pos_s, pos_r, pos_o = zip(*pos_samples)
    #     labels[pos_r, pos_s, pos_o] = 1
    #     preds = recon.view(-1, recon.shape[-1])
    #     preds[pos_s*recon.shape[-1] + pos_o]
    #     labels = torch.zeros(recon.shape[-1], dtype=torch.long)
    #     recon_loss = self.loss_func(recon.view(-1, recon.shape[-1]), labels.view(-1, recon.shape[-1]))
    #     kl = 0.5 / recon.shape[1] * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
    #     loss = recon_loss + kl
    #     score_mat = torch.sigmoid(recon[pos_r, pos_s, :]).detach().cpu().numpy()
    #     pos_obj = torch.Tensor(pos_o).cpu().numpy()
    #     pos_obj = np.array(pos_obj, dtype=np.int)
    #     raw_ranks = [np.sum(score >= score[e]) for score, e in zip(score_mat, pos_obj)]
    #     hit1 = np.sum([1 for rank in raw_ranks if rank <= 1]) / len(raw_ranks)
    #     hit3 = np.sum([1 for rank in raw_ranks if rank <= 3]) / len(raw_ranks)
    #     hit10 = np.sum([1 for rank in raw_ranks if rank <= 10]) / len(raw_ranks)
    #     mr = np.mean(raw_ranks)
    #     mrr = np.mean(1. / np.array(raw_ranks))
    #     return loss, recon_loss, kl, hit10, mr, mrr, hit1, hit3, hit10

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
        score_pos = torch.sigmoid(recon[pos_r, pos_s, pos_o])
        score_neg = torch.sigmoid(recon[neg_r, neg_s, neg_o])
        scores = torch.cat([score_pos, score_neg])
        targets = torch.cat([torch.ones(len(pos_o)), torch.zeros(len(neg_o))], dim=0).cuda()
        # import ipdb;ipdb.set_trace()
        recon_loss = F.binary_cross_entropy_with_logits(scores,targets)
        kl = 0.5 / recon.shape[1] * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
        loss = recon_loss + kl

        roc = roc_auc_score(targets.detach().cpu().numpy().astype(np.int), scores.detach().cpu().numpy())
        ap = average_precision_score(targets.detach().cpu().numpy().astype(np.int), scores.detach().cpu().numpy())
        # import ipdb;ipdb.set_trace()

        score_mat = torch.sigmoid(recon[pos_r, pos_s,:]).detach().cpu().numpy()
        pos_obj = torch.Tensor(pos_o).cpu().numpy()
        pos_obj = np.array(pos_obj, dtype=np.int)
        raw_ranks = [np.sum(score >= score[e]) for score, e in zip(score_mat, pos_obj)]

        inv_score_mat = torch.sigmoid(recon[pos_r, :, pos_o]).detach().cpu().numpy()
        pos_sub = torch.Tensor(pos_s).cpu().numpy()
        pos_sub = np.array(pos_sub, dtype=np.int)
        inv_raw_ranks = [np.sum(score >= score[e]) for score, e in zip(inv_score_mat, pos_sub)]
        ranks = raw_ranks + inv_raw_ranks

        hit1 = np.sum([1. for rank in ranks if rank <= 1])/len(ranks)
        hit3 = np.sum([1. for rank in ranks if rank <= 3])/len(ranks)
        hit10 = np.sum([1. for rank in ranks if rank <= 10])/len(ranks)
        mr = np.mean(ranks)
        mrr = np.mean(1./np.array(ranks))

        # f1 = f1_score(targets.cpu(), preds.cpu(), average='macro', labels=np.arange(self.num_rels))
        return loss, recon_loss, kl, hit10, mr, mrr, hit1, hit3, hit10
