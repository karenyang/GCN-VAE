"""Torch Module for Relational graph convolution layer"""
# pylint: disable= no-member, arguments-differ, invalid-name
import torch
from torch import nn
import torch.nn.functional as F
from dgl import function as fn
from dgl.nn.pytorch import utils


class Conv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, bias=True, **kwargs):
        super(Conv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=True, **kwargs)

    def forward(self, A):
        # since pytorch's conv1d assume channel in the middle (N,C,L), we need to transform the input's channel to the
        # middle and at the end transform it back into (C,N,L)
        if len(A.shape) == 2:
            A = A.unsqueeze(1)
        else:
            A = A.transpose(0, 1)
        A = self.conv1d(A)
        A = A.transpose(0, 1)
        return A


class RefineRelationAdjLayer(nn.Module):
    def __init__(self, act=None, normalize=True, non_negative=False, **kwargs):
        super(RefineRelationAdjLayer, self).__init__()
        self.normalize = normalize
        self.non_negative = non_negative
        self.act = act

    def forward(self, z):
        # import ipdb; ipdb.set_trace()
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
        R = z.matmul(z.transpose(-1, -2))
        if self.non_negative:
            R = R + torch.ones_like(R)
        if self.act is not None:
            R = self.act(z)
        if len(R.shape) == 2:
            R = R.unsqueeze(0)
        return R


class RefineLatentFeatLayer(nn.Module):
    def __init__(self, input_dim, out_dim, act=None, **kwargs):
        super(RefineLatentFeatLayer, self).__init__()
        self.act = act
        self.linear = nn.Linear(input_dim, out_dim)

    def forward(self, R, z, x=None):
        import ipdb;
        ipdb.set_trace()
        z = self.linear(z)
        z = R.matmul(R.transpose(-1, -2).matmul(z))
        if x is not None:
            z = z + R.matmul(R.transpose(-1, -2).matmul(x))
        if self.act is not None:
            return self.act(z)
        return z


class EmbeddingLayer(nn.Module):
    def __init__(self, num_nodes, h_dim):
        super(EmbeddingLayer, self).__init__()
        self.embedding = nn.Embedding(num_nodes, h_dim)

    def forward(self, g, h, r, norm):
        return self.embedding(h.squeeze())


class RelGraphConv(nn.Module):
    r"""Relational graph convolution layer.
    Relational graph convolution is introduced in "`Modeling Relational Data with Graph
    Convolutional Networks <https://arxiv.org/abs/1703.06103>`__"
    ----------
    in_feat : int
        Input feature size.
    out_feat : int
        Output feature size.
    num_rels : int
        Number of relations.
    regularizer : str
        Which weight regularizer to use "basis" or "bdd"
    num_bases : int, optional
        Number of bases. If is none, use number of relations. Default: None.
    bias : bool, optional
        True if bias is added. Default: True
    activation : callable, optional
        Activation function. Default: None
    self_loop : bool, optional
        True to include self loop message. Default: False
    dropout : float, optional
        Dropout rate. Default: 0.0
    """

    def __init__(self,
                 in_feat,
                 out_feat,
                 num_rels,
                 regularizer="basis",
                 num_bases=None,
                 bias=True,
                 activation=None,
                 self_loop=False,
                 dropout=0.0):
        super(RelGraphConv, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.regularizer = regularizer
        self.num_bases = num_bases
        if self.num_bases is None or self.num_bases > self.num_rels or self.num_bases < 0:
            self.num_bases = self.num_rels
        self.bias = bias
        self.activation = activation
        self.self_loop = self_loop

        if regularizer == "basis":
            # add basis weights
            self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
            if self.num_bases < self.num_rels:
                # linear combination coefficients
                self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            if self.num_bases < self.num_rels:
                nn.init.xavier_uniform_(self.w_comp,
                                        gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.basis_message_func
        elif regularizer == "bdd":
            if in_feat % num_bases != 0 or out_feat % num_bases != 0:
                raise ValueError('Feature size must be a multiplier of num_bases.')
            # add block diagonal weights
            self.submat_in = in_feat // self.num_bases
            self.submat_out = out_feat // self.num_bases

            # assuming in_feat and out_feat are both divisible by num_bases
            self.weight = nn.Parameter(torch.Tensor(
                self.num_rels, self.num_bases * self.submat_in * self.submat_out))
            nn.init.xavier_uniform_(self.weight, gain=nn.init.calculate_gain('relu'))
            # message func
            self.message_func = self.bdd_message_func
        else:
            raise ValueError("Regularizer must be either 'basis' or 'bdd'")

        # bias
        if self.bias:
            self.h_bias = nn.Parameter(torch.Tensor(out_feat))
            nn.init.zeros_(self.h_bias)

        # weight for self loop
        if self.self_loop:
            self.loop_weight = nn.Parameter(torch.Tensor(in_feat, out_feat))
            nn.init.xavier_uniform_(self.loop_weight,
                                    gain=nn.init.calculate_gain('relu'))

        self.dropout = nn.Dropout(dropout)

    def basis_message_func(self, edges):
        """Message function for basis regularizer"""
        if self.num_bases < self.num_rels:
            # generate all weights from bases
            weight = self.weight.view(self.num_bases,
                                      self.in_feat * self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(
                self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight
        # Message Update per relation channel:
        # msg[i, :] = matmul(edge_src[i, :], weights[edge_type[i], :, :])
        # N_edge_batch x n_feat = N_edge_batch x n_feat , n_rels x n_feat x n_feat, n_rels
        msg = utils.bmm_maybe_select(edges.src['h'], weight, edges.data['type'])
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def bdd_message_func(self, edges):
        """Message function for block-diagonal-decomposition regularizer"""
        if edges.src['h'].dtype == torch.int64 and len(edges.src['h'].shape) == 1:
            raise TypeError('Block decomposition does not allow integer ID feature.')
        weight = self.weight.index_select(0, edges.data['type']).view(
            -1, self.submat_in, self.submat_out)
        node = edges.src['h'].view(-1, 1, self.submat_in)
        msg = torch.bmm(node, weight).view(-1, self.out_feat)
        if 'norm' in edges.data:
            msg = msg * edges.data['norm']
        return {'msg': msg}

    def forward(self, g, x, etypes, norm=None):
        """ Forward computation
        Parameters
        ----------
        g : DGLGraph
            The graph.
        x : torch.Tensor
            Input node features. Could be either
                * :math:`(|V|, D)` dense tensor
                * :math:`(|V|,)` int64 vector, representing the categorical values of each
                  node. We then treat the input feature as an one-hot encoding feature.
        etypes : torch.Tensor
            Edge type tensor. Shape: :math:`(|E|,)`
        norm : torch.Tensor
            Optional edge normalizer tensor. Shape: :math:`(|E|, 1)`
        Returns
        -------
        torch.Tensor
            New node features.
        """
        g = g.local_var()
        g.ndata['h'] = x
        g.edata['type'] = etypes
        if norm is not None:
            g.edata['norm'] = norm
        if self.self_loop:
            loop_message = utils.matmul_maybe_select(x, self.loop_weight)
        # message passing
        g.update_all(self.message_func, fn.sum(msg='msg', out='h'))
        # apply bias and activation
        node_repr = g.ndata['h']
        if self.bias:
            node_repr = node_repr + self.h_bias
        if self.self_loop:
            node_repr = node_repr + loop_message
        if self.activation:
            node_repr = self.activation(node_repr)
        node_repr = self.dropout(node_repr)
        return node_repr
