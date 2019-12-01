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

from model import RGCN, KGVAE

import utils



class LinkPredict(nn.Module):
    def __init__(self, model_class, in_dim, h_dim, num_rels, num_bases=-1,
                 num_hidden_layers=1, dropout=0, use_cuda=True, reg_param=0, kl_param=1):
        super(LinkPredict, self).__init__()
        self.rgcn = model_class(in_dim, h_dim, h_dim, num_rels * 2, num_bases,
                         num_hidden_layers, dropout, use_cuda, vae=model_class==KGVAE)
        self.reg_param = reg_param
        self.w_relation = nn.Parameter(torch.Tensor(num_rels, h_dim))
        self.kl_param = kl_param
        nn.init.xavier_uniform_(self.w_relation,
                                gain=nn.init.calculate_gain('relu'))

    def calc_score(self, embedding, triplets):
        # DistMult
        s = embedding[triplets[:,0]]
        r = self.w_relation[triplets[:,1]]
        o = embedding[triplets[:,2]]
        score = torch.sum(s * r * o, dim=1).sigmoid()
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
        kl = self.rgcn.get_kl()
        loss = (predict_loss + self.reg_param * reg_loss + self.kl_param * kl).cuda()
        return loss, predict_loss, kl

def node_norm_to_edge_norm(g, node_norm):
    g = g.local_var()
    # convert to edge norm
    g.ndata['norm'] = node_norm
    g.apply_edges(lambda edges : {'norm' : edges.dst['norm']})
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
    model = LinkPredict(model_class=RGCN,#KGVAE, #RGCN
                        in_dim=num_nodes,
                        h_dim=args.n_hidden,
                        num_rels=num_rels,
                        num_bases=args.n_bases,
                        num_hidden_layers=args.n_layers,
                        dropout=args.dropout,
                        use_cuda=True,
                        reg_param=args.regularization,
                        kl_param=args.kl_param,
                        )

    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    # build val graph
    val_adj_list, val_degrees = utils.get_adj_and_degrees(num_nodes, valid_data)

    if use_cuda:
        model.cuda()

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model_state_file = 'model_state.pth'
    forward_time = []
    backward_time = []

    # training loop
    print("start training...")

    epoch = 0
    best_mrr = 0
    train_records = open("train_records.txt", "w")
    val_records = open("val_records.txt", "w")
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, data, labels = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)
        # print("Done edge sampling")

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
        loss, pred_loss, kl = model.get_loss(g, embed, data, labels)
        train_records.write("{:d};{:.4f};{:.4f};{:.4f}\n".format(epoch, loss.item(), pred_loss.item(), kl.item()))
        train_records.flush()
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm) # clip gradients
        optimizer.step()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)

        print("Epoch {:04d} | Loss {:.4f} |  Best MRR {:.4f} | pred {:.4f} | reg {:.4f} | kl {:.4f} ".
              format(epoch, loss.item(), best_mrr, pred_loss, loss-pred_loss-kl, kl))
        optimizer.zero_grad()

        # validation
        if epoch % args.evaluate_every ==1:
            val_g, val_node_id, val_edge_type, val_node_norm, val_data, val_labels = utils.generate_sampled_graph_and_labels(
                valid_data, args.graph_batch_size, args.graph_split_size,
                num_rels, val_adj_list, val_degrees, args.negative_sample,
                args.edge_sampler)
            # print("Done edge sampling for validation")
            val_node_id = torch.from_numpy(val_node_id).view(-1, 1).long()
            val_edge_type = torch.from_numpy(val_edge_type)
            val_edge_norm = node_norm_to_edge_norm(val_g, torch.from_numpy(val_node_norm).view(-1, 1))
            val_data, val_labels = torch.from_numpy(val_data), torch.from_numpy(val_labels)
            if use_cuda:
                val_node_id = val_node_id.cuda()
                val_edge_type, val_edge_norm = val_edge_type.cuda(), val_edge_norm.cuda()
                val_data, val_neg_samples = val_data.cuda(), val_labels.cuda()
            embed = model(val_g, val_node_id, val_edge_type, val_edge_norm)
            mr, mrr, hits = utils.calc_mrr(embed, model.w_relation, val_data[val_labels==1], hits=[1, 3, 10], eval_bz=args.eval_batch_size)
            val_records.write(
                "{:d};{:.4f};{:.4f};\n".format(epoch, mr, mrr)+ ';'.join([str(i) for i in hits]) + "\n" )
            val_records.flush()
            if mrr < best_mrr:
                if epoch >= args.n_epochs:
                    break
            else:
                best_mrr = mrr
                print("Best mrr", mrr)
            torch.save({'state_dict': model.state_dict(), 'epoch': epoch}, model_state_file)

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))

    # print("\nstart testing:")
    # # use best model checkpoint
    # checkpoint = torch.load(model_state_file)
    # if use_cuda:
    #     model.cpu() # test on CPU
    # model.eval()
    # model.load_state_dict(checkpoint['state_dict'])
    # print("Using best epoch: {}".format(checkpoint['epoch']))
    # embed = model(test_graph, test_node_id, test_rel, test_norm)
    # utils.calc_mrr(embed, model.w_relation, test_data,
    #                hits=[1, 3, 10], eval_bz=args.eval_batch_size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='RGCN')
    parser.add_argument("--dropout", type=float, default=0.2,
            help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
            help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
            help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
            help="learning rate")
    parser.add_argument("--n-bases", type=int, default=100,
            help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
            help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=20000,
            help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
            help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=100,
            help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=1e-2,
            help="regularization weight")
    parser.add_argument("--kl-param", type=float, default=1e-3,
                        help="kl regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
            help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=10000,
            help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
            help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=10,
            help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=100,
            help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="neighbor",
            help="type of edge sampler: 'uniform' or 'neighbor'")

    args = parser.parse_args()
    print(args)
    main(args)
