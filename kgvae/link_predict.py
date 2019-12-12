"""
Modeling Relational Data with Graph Convolutional Networks
Paper: https://arxiv.org/abs/1703.06103
Code: https://github.com/MichSchli/RelationPrediction

Difference compared to MichSchli/RelationPrediction
* Report raw metrics instead of filtered metrics.
* By default, we use uniform edge sampling instead of neighbor-based edge
  sampling used in author's code. In practice, we find it achieves similar MRR
  probably because the model only uses one GNN layer so messages are propagated
  among immediate neighbors. User could specify "--edge-sampler=neighbor" to switch
  to neighbor-based edge sampling.
"""

import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data
from dgl.nn.pytorch import RelGraphConv
from torch.distributions import normal
from model import RGCN, KGVAE

import utils


class LinkPredict(nn.Module):
    def __init__(self, model_class, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=True, reg_param=0, kl_param=0, mmd_param=0, k=1, n_flows=0):
        super(LinkPredict, self).__init__()

        self.encoder = model_class(num_nodes=in_dim,
                                   h_dim=h_dim,
                                   out_dim=h_dim,
                                   num_rels=num_rels * 2,
                                   num_bases=num_bases,
                                   num_hidden_layers=num_hidden_layers,
                                   dropout=dropout,
                                   use_self_loop=use_cuda,
                                   use_cuda=use_cuda,
                                   k=k,
                                   n_flows=n_flows,
                                   )
        self.reg_param = reg_param
        self.kl_param = kl_param
        self.mmd_param = mmd_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.use_cuda = use_cuda
        self.k = k
        self.n_flows = n_flows
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:, 0]]
        r = self.w_relation[triplets[:, 1]]
        o = embedding[triplets[:, 2]]
        score = torch.sum(s * r * o, dim=1)
        return score

    def forward(self, g, h, r, norm):
        return self.encoder.forward(g, h, r, norm)

    def regularization_loss(self, embedding):
        return torch.mean(embedding.pow(2)) + torch.mean(self.w_relation.pow(2))

    def get_loss(self, g, embed, triplets, labels):
        # triplets is a list of data samples (positive and negative)
        # each row in the triplets is a 3-tuple of (source, relation, destination)
        score = self.calc_score(embed, triplets)
        if self.n_flows > 0:
            score = score + self.encoder.get_flow_log_prob()
        predict_loss = F.binary_cross_entropy_with_logits(score, labels)
        reg_loss = self.regularization_loss(embed)
        if self.kl_param > 0:
            kl = self.encoder.get_kl(embed)
        else:
            kl = nn.Parameter(torch.zeros(1), requires_grad=False)
            if self.use_cuda:
                kl = kl.cuda()
        if self.mmd_param > 0:
            mmd = self.encoder.get_mmd(embed)
        else:
            mmd = nn.Parameter(torch.zeros(1), requires_grad=False)
            if self.use_cuda:
                mmd = mmd.cuda()
        loss = predict_loss + self.reg_param * reg_loss + self.kl_param * kl + self.mmd_param * mmd
        return loss, predict_loss, kl, mmd


def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges: {'norm': edges.dst['norm']})
    return g.edata['norm']


def main(args):
    # load graph data
    data = load_data(args.dataset)
    num_nodes = data.num_nodes
    train_data = data.train
    valid_data = data.valid
    test_data = data.test
    num_rels = data.num_rels

    # check cuda
    use_cuda = args.gpu >= 0 and torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(args.gpu)

    # create model
    if args.model_class == "KGVAE":
        model_class = KGVAE
    else:
        model_class = RGCN

    model = LinkPredict(model_class=model_class,
                        in_dim=num_nodes,
                        h_dim=args.n_hidden,
                        num_rels=num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=use_cuda,
                        reg_param=args.regularization,
                        kl_param=args.kl_param,
                        mmd_param=args.mmd_param,
                        k=args.mog_k,
                        n_flows=args.n_flows)

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)

    # build test graph
    val_graph, val_rel, val_norm = utils.build_test_graph(
        num_nodes, num_rels, valid_data)
    # val_deg = val_graph.in_degrees(
    #     range(val_graph.number_of_nodes())).float().view(-1, 1)
    val_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
    val_rel = torch.from_numpy(val_rel)
    val_norm = node_norm_to_edge_norm(val_graph, torch.from_numpy(val_norm).view(-1, 1))

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    forward_time = []
    backward_time = []

    if args.test_mode is True:
        print("\nstart testing:")

        # use best model checkpoint
        checkpoint = torch.load(args.model_state_file)
        test_data = torch.LongTensor(test_data)

        # build test graph
        test_graph, test_rel, test_norm = utils.build_test_graph(
            num_nodes, num_rels, test_data)
        # test_deg = test_graph.in_degrees(
        #     range(test_graph.number_of_nodes())).float().view(-1, 1)
        test_node_id = torch.arange(0, num_nodes, dtype=torch.long).view(-1, 1)
        test_rel = torch.from_numpy(test_rel)
        test_norm = node_norm_to_edge_norm(test_graph, torch.from_numpy(test_norm).view(-1, 1))
        if use_cuda:
            model.cpu()  # test on CPU
        model.eval()
        model.load_state_dict(checkpoint['state_dict'])
        print("Using best epoch: {}".format(checkpoint['epoch']))
        embed = model(test_graph, test_node_id, test_rel, test_norm)
        if args.generate:
            print("\nstart generating obama neighborhood:")
            # use best model checkpoint
            utils.generate(embed, model.w_relation, test_data)
        utils.calc_mrr(embed, model.w_relation, test_data,
                       hits=[1, 3, 10], eval_bz=args.eval_batch_size, all_batches=True,
                       flow_log_prob=model.encoder.get_flow_log_prob())
        exit(0)
    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    if args.load is True:
        print(f"Loading checkpoint file {args.model_state_file} for training")
        checkpoint = torch.load(args.model_state_file)
        model.load_state_dict(checkpoint['state_dict'])
        epoch = checkpoint['epoch']

    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        data, labels = torch.from_numpy(data), torch.from_numpy(labels)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)
        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            data, labels = data.cuda(), labels.cuda()

        t0 = time.time()
        embed = model(g, node_id, edge_type, edge_norm)
        loss, pred_loss, kl, mmd = model.get_loss(g, embed, data, labels)
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()

        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print("Epoch {:04d} | Loss {:.4f} | Best MRR {:.4f} | pred_loss {:.4f} | kl {:.4f} | mmd {:.4f}".
              format(epoch, loss.item(), best_mrr, pred_loss.item(), kl.item(), mmd.item()))

        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every == 0:
            # perform validation on CPU because full graph is too large
            if use_cuda:
                model.cpu()
            model.eval()
            print("start eval")
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                       args.model_state_file)

            embed = model(val_graph, val_node_id, val_rel, val_norm)

            mrr = utils.calc_mrr(embed, model.w_relation, valid_data,
                                 hits=[1, 3, 10], eval_bz=args.eval_batch_size, all_batches=False)
            # save best model
            if mrr < best_mrr:
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           args.model_state_file + "_latest")
            else:
                best_mrr = mrr
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           args.model_state_file)
            if use_cuda:
                model.cuda()

        if epoch >= args.n_epochs:
            break

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Link Prediction')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=1e5,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=400,
                        help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--kl-param", type=float, default=1e-5,
                        help="kl regularization weight")
    parser.add_argument("--mmd-param", type=float, default=0,
                        help="mmd regularization weight")
    parser.add_argument("--mog-k", type=int, default=10,
                        help="number of mixture of gaussian")
    parser.add_argument("--n-flows", type=int, default=0,
                        help="number of flow transform layers")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=20000,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
                        help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=200,
                        help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="uniform",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--test-mode", type=bool, default=False,
                        help="only evaluate on test dataset")
    parser.add_argument("--model-state-file", type=str, default='model_state.pth',
                        help="model state file to load or save")
    parser.add_argument("--model-class", type=str, default='KGVAE',
                        help="model class")
    parser.add_argument("--load", type=bool, default=False,
                        help="whether to load a model state file for training")
    parser.add_argument("--generate", type=bool, default=False,
                        help="whether to print generated triplets around Obama")

    args = parser.parse_args()
    print(args)
    main(args)
