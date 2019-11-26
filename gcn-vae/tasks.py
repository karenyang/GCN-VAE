import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import KGVAE

class LinkPredict(nn.Module):
    def __init__(self, model_class, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=False, reg_param=0):
        super(LinkPredict, self).__init__()
        self.kgvae = KGVAE(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.rgcn.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        return predict_loss + self.reg_param * reg_loss


# class LinkPredict(nn.Module):
#     def __init__(self, model_class, in_dim, h_dim, num_rels, num_bases=-1,
#                  num_hidden_layers=1, dropout=0, use_cuda=True, reg_param=0):
#         super(LinkPredict, self).__init__()
#         self.reg_param = reg_param
#         self.model = model_class(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
#                          num_hidden_layers, dropout, use_cuda)
#
#     def calc_score(self, model_output, triplets, labels):
#         return self.model.calc_score(model_output, triplets, labels)
#
#     def forward(self, g, h, r, norm):
#         return self.model(g, h, r, norm)
#
#     def regularization_loss(self, model_output):
#         return self.model.regularization_loss( model_output)
#
#     def get_loss(self, g, model_output, triplets, labels):
#         # triplets is a list of data samples (positive and negative)
#         # each row in the triplets is a 3-tuple of (source, relation, destination)
#         predict_loss = self.calc_score(model_output, triplets, labels)
#         reg_loss = self.regularization_loss(model_output)
#         return predict_loss + self.reg_param * reg_loss