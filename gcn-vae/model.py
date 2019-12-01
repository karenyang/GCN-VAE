import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
from layers import RelGraphConv, EmbeddingLayer, Conv1d, RefineLatentFeatLayer, RefineRelationAdjLayer
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score
import numpy as np
import utils
from tqdm import tqdm


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
        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.h_dim) / np.sqrt(self.k * self.h_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def build_encoder(self):
        self.input_layer = EmbeddingLayer(self.num_nodes, self.h_dim)
        self.rconv_layer_1 = self.build_rconv_layer(out_dim=self.h_dim, activation=nn.ReLU())
        self.rconv_layer_2 = self.build_rconv_layer(out_dim=2*self.h_dim, activation=lambda x:x)
        self.z_mean_branch = nn.Linear(self.h_dim, self.h_dim)
        self.z_log_std_branch = nn.Linear(self.h_dim, self.h_dim)

    def build_rconv_layer(self,  out_dim, activation):
        return RelGraphConv(self.h_dim, out_dim, self.num_rels, "bdd",
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
        h = self.rconv_layer_1(g, h, r, norm)
        h = self.rconv_layer_2(g, h, r, norm)
        self.z_mean, self.z_sigma = utils.gaussian_parameters(h)
        z = utils.sample_gaussian(self.z_mean, self.z_sigma)
        return z

    def build_decoder(self):
        self.w_relation = nn.Parameter(torch.Tensor(self.num_rels, self.h_dim))
        nn.init.xavier_uniform_(self.w_relation, gain=nn.init.calculate_gain('relu'))

    def decoder(self, z, cond_relation, score_method='DistMult'):
        """
        Condition on head and relations, output mostly likely tail entities based on scores
        :param z: encoder output, z samples
        :param x: input node attributes
        :return: reconstructed score matrix for tail entities
        """
        h_id, r_id = zip(*cond_relation)
        H, R = z[h_id, :], self.w_relation[r_id, :]
        T = torch.zeros(len(H), self.in_dim)
        if score_method == 'DistMult':
            head_rel_emb = (H * R).transpose(0, 1).unsqueeze(2)  # D x num_samples x 1
            tail_emb = z.transpose(0, 1).unsqueeze(1)  # D x 1 X in_dim
            T = torch.bmm(head_rel_emb, tail_emb)  # D x num_samples x in_dim
            T = T.sum(dim=0)
        return T

    # def decoder_0(self, z, cond_relation, score_method='DistMult'):
    #     """
    #     Condition on head and relations, output mostly likely tail entities based on scores
    #     :param z: encoder output, z samples
    #     :param x: input node attributes
    #     :return: reconstructed score matrix for tail entities
    #     """
    #     h_id, r_id = zip(*cond_relation)
    #     H, R = z[h_id, :], self.w_relation[r_id, :]
    #     T = torch.zeros(len(H), self.in_dim)
    #     if score_method == 'DistMult':
    #         head_rel_emb = (H * R) #  num_samples x D
    #         tail_emb = z.transpose(0,1) # D x num_samples
    #         T = torch.matmul(head_rel_emb, tail_emb)  # D x num_samples x in_dim
    #         recon = F.softmax(T)
    #     return recon

    def forward(self, g, h, r, norm, cond_relation):
        self.node_id = h
        self.in_dim = self.node_id.shape[0]
        z = self.encoder(g, h, r, norm)
        self.z = z
        recon = self.decoder(z, cond_relation=cond_relation, score_method='DistMult')
        return recon

    def sample_z(self, batch):
        m, v = utils.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return utils.sample_gaussian(m, v)

    def get_kl(self):
        m_mixture, z_mixture = utils.gaussian_parameters(self.z_pre, dim=1)
        m = self.z_mean
        v = self.z_sigma
        kl = torch.mean(utils.log_normal(self.z, m, v) - utils.log_normal_mixture(self.z, m_mixture, z_mixture))
        return kl

    def get_loss(self, recon, pos_samples, neg_samples):
        # import ipdb;ipdb.set_trace()
        recon_loss =  self.loss_func(F.softmax(recon), pos_samples[:,2])
        kl = self.get_kl()
        loss = recon_loss + kl
        return loss, recon_loss, kl

    # def get_loss_0(self, recon, pos_samples, neg_samples):
    #     # import ipdb;ipdb.set_trace()
    #     O = pos_samples[:, 2]
    #     negO = neg_samples[:, 2]
    #     pos_score = torch.Tensor([recon[i, O[i]] for i in range(len(pos_samples))])
    #     neg_score = torch.Tensor([recon[i, negO[i]] for i in range(len(neg_samples))])
    #     recon_loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones(len(pos_score))) + \
    #                  F.binary_cross_entropy_with_logits(neg_score, torch.zeros(len(neg_score)))
    #     if self.use_cuda:
    #         recon_loss = recon_loss.cuda()
    #     kl = 0.5 / recon.shape[1] * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
    #     loss = recon_loss + kl
    #     return loss, recon_loss, kl

    def get_metrics(self, recon, triplets):
        S, R, O = zip(*triplets)
        labels = torch.LongTensor(O)
        if self.use_cuda:
            labels = labels.cuda()
        preds = torch.argmax(recon, dim=-1)
        accu = (preds == labels).sum().float() / len(preds)
        print("Accu: {:.6f}".format(accu.item()))

        # get ranks for the standard metrics
        bz = 100
        n_batch = (len(triplets) + bz - 1) // bz
        ranks = []
        # import ipdb; ipdb.set_trace()
        for idx in range(n_batch):
            start = idx * bz
            end = min(len(triplets), (idx + 1) * bz)
            si = S[start: end]
            ri = R[start: end]
            oi = O[start: end]
            # filter for correct ranking
            # for s,r in zip(si, ri):
            #     valid_os = [o for o in oi if (s,r,o) in triplets[start:end]]
            # other_os = [i for s,r in zip(si,ri)  for i in range(oi) if (s,r,oi[i]) in triplets[start: end]]
            # si =  = [ _o[i] for i in range(_o) if (_s[i], _r[i], _o[i]) not in ]
            oi = torch.LongTensor(oi)
            if self.use_cuda:
                oi = oi.cuda()
            _, indices = torch.sort(recon[start:end, :], dim=1, descending=True)
            indices = torch.nonzero(indices == oi.view(-1, 1))
            indices = indices[:, 1].view(-1)
            ranks.append(indices)
        ranks = torch.cat(ranks) + 1  # start from 1
        mr = torch.mean(ranks.float())
        print("MR: {:.6f}".format(mr.item()))
        mrr = torch.mean(1.0 / ranks.float())
        print("MRR: {:.6f}".format(mrr.item()))
        _hits = []
        for hit in [1, 3, 10]:
            avg_count = torch.mean((ranks <= hit).float())
            _hits.append(avg_count.item())
            print("Hits @ {}: {:.6f}".format(hit, avg_count.item()))
        return accu, mr.item(), mrr.item(), _hits[0], _hits[1], _hits[2]

    # def get_loss_0(self, embed, pos_samples, neg_samples):
    #     S,R,O = pos_samples[:,0], pos_samples[:,1], pos_samples[:,2]
    #     negO = neg_samples[:,2]
    #     head_rel_emb = embed[S,:] * self.w_relation[R, :]
    #     tail_emb = embed[O,:]
    #     neg_tail_emb = embed[negO,:]
    #
    #     pos_score = head_rel_emb * tail_emb # D x num_samples x in_dim
    #     pos_score = pos_score.sum(dim=0).sigmoid()
    #     pos_target = torch.ones(len(pos_score))
    #
    #     neg_score = head_rel_emb * neg_tail_emb  # D x num_samples x in_dim
    #     neg_score = neg_score.sum(dim=0).sigmoid()
    #     neg_target = torch.zeros(len(neg_score))
    #     if self.use_cuda:
    #         pos_target = pos_target.cuda()
    #     if self.use_cuda:
    #         neg_target = neg_target.cuda()
    #
    #     recon_loss = F.binary_cross_entropy_with_logits(pos_score, pos_target) + \
    #                  F.binary_cross_entropy_with_logits(neg_score, neg_target)
    #     kl = 0.5 / self.in_dim * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
    #     loss = recon_loss + kl
    #     return loss, recon_loss, kl
    #
    # def get_metrics_0(self, embed, triplets):
    #     S, R, O = zip(*triplets)
    #     labels = torch.LongTensor(O)
    #     if self.use_cuda:
    #         labels = labels.cuda()
    #     head_rel_emb = embed[S, :] * self.w_relation[R, :]
    #     head_rel_emb = head_rel_emb.transpose(0, 1).unsqueeze(2)  # D x num_samples x 1
    #     tail_emb = embed[O, :].transpose(0, 1).unsqueeze(1)  # D x 1 X in_dim
    #     score = torch.bmm(head_rel_emb, tail_emb)  # D x num_samples x in_dim
    #     score = score.sum(dim=0).sigmoid()
    #
    #     preds = torch.argmax(score, dim=-1)
    #     accu = (preds == labels).sum().float() / len(preds)
    #     print("Accu: {:.6f}".format(accu.item()))
    #
    #     # get ranks for the standard metrics
    #     bz = 100
    #     n_batch = (len(triplets) + bz - 1) // bz
    #     ranks = []
    #     # import ipdb; ipdb.set_trace()
    #     for idx in range(n_batch):
    #         start = idx * bz
    #         end = min(len(triplets), (idx + 1) * bz)
    #         si = S[start: end]
    #         ri = R[start: end]
    #         oi = O[start: end]
    #         # filter for correct ranking
    #         # for s,r in zip(si, ri):
    #         #     valid_os = [o for o in oi if (s,r,o) in triplets[start:end]]
    #         # other_os = [i for s,r in zip(si,ri)  for i in range(oi) if (s,r,oi[i]) in triplets[start: end]]
    #         # si =  = [ _o[i] for i in range(_o) if (_s[i], _r[i], _o[i]) not in ]
    #         oi = torch.LongTensor(oi)
    #         if self.use_cuda:
    #             oi = oi.cuda()
    #         _, indices = torch.sort(score[start:end, :], dim=1, descending=True)
    #         indices = torch.nonzero(indices == oi.view(-1, 1))
    #         indices = indices[:, 1].view(-1)
    #         ranks.append(indices)
    #     ranks = torch.cat(ranks) + 1  # start from 1
    #     mr = torch.mean(ranks.float())
    #     print("MR: {:.6f}".format(mr.item()))
    #     mrr = torch.mean(1.0 / ranks.float())
    #     print("MRR: {:.6f}".format(mrr.item()))
    #     _hits = []
    #     for hit in [1, 3, 10]:
    #         avg_count = torch.mean((ranks <= hit).float())
    #         _hits.append(avg_count.item())
    #         print("Hits @ {}: {:.6f}".format(hit, avg_count.item()))
    #     return accu, mr.item(), mrr.item(), _hits[0], _hits[1], _hits[2]


    #
    # def get_loss(self, recon, pos_samples, neg_samples):
    #     """
    #     preds will be the decoder output
    #     Labels will be the graph
    #     output reconstruction error.
    #     :param preds:
    #     :param labels:
    #     :return: recon + kl
    #     """
    #     pos_s, pos_r, pos_o = zip(*pos_samples)
    #     neg_s, neg_r, neg_o = zip(*neg_samples)
    #     score_pos = torch.sigmoid(recon[pos_r, pos_s, pos_o])
    #     score_neg = torch.sigmoid(recon[neg_r, neg_s, neg_o])
    #     target_pos = torch.zeros_like(score_pos)
    #     target_pos[:, pos_o] = 1
    #     target_neg = torch.zeros_like(score_neg)
    #     import ipdb;ipdb.set_trace()
    #
    #     recon_loss = F.binary_cross_entropy_with_logits(score_pos, target_pos) + \
    #                  F.binary_cross_entropy_with_logits(score_neg, target_neg)
    #
    #     kl = 0.5 / recon.shape[1] * torch.sum(torch.exp(self.z_log_std) + self.z_mean ** 2 - 1. - self.z_log_std)
    #     loss = recon_loss + kl
    #
    #     preds = score_pos.argmax(-1)
    #     labels = torch.Tensor(pos_o)
    #     accu = (preds==labels).sum().float()/len(preds)
    #     # roc = roc_auc_score(targets.detach().cpu().numpy().astype(np.int), scores.detach().cpu().numpy())
    #     # ap = average_precision_score(targets.detach().cpu().numpy().astype(np.int), scores.detach().cpu().numpy())
    #
    #
    #     # eval for mr, mrr, and hit@1, hit@3, hit@10
    #     mr, mrr, hit1,hit3,hit10 = self.get_metrics(recon, pos_samples, batch_size=100)
    #
    #     # f1 = f1_score(targets.cpu(), preds.cpu(), average='macro', labels=np.arange(self.num_rels))
    #     return loss, recon_loss, kl, accu, mr, mrr, hit1, hit3, hit10
