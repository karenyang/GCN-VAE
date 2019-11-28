import argparse
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from dgl.contrib.data import load_data

from model import KGVAE
import utils


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
    model = KGVAE(
        num_nodes=num_nodes,
        h_dim=args.n_hidden,
        out_dim=args.n_hidden,
        num_rels=num_rels * 2,
        num_bases=args.n_bases,
        num_hidden_layers=args.n_layers,
        dropout=args.dropout,
        num_encoder_output_layers=2,
        use_self_loop=True,
        use_cuda=use_cuda,
    )

    # if args.test_mode:
    #     print("\nstart testing:")
    #     # use best model checkpoint
    #     checkpoint = torch.load(args.model_state_file)
    #     model.eval()
    #     model.load_state_dict(checkpoint['state_dict'])
    #     print("Using best epoch: {}".format(checkpoint['epoch']))
    #
    #     embed = model(test_graph, test_node_id, test_rel, test_norm)
    #     utils.calc_mrr(embed, model.w_relation, test_data,
    #                    hits=[1, 3, 10], eval_bz=args.eval_batch_size)
    #     return

    if use_cuda:
        model.cuda()
    # validation and testing triplets
    valid_data = torch.LongTensor(valid_data)
    test_data = torch.LongTensor(test_data)

    original_graph, original_rel, _ = utils.build_graph(num_nodes, num_rels,
                                                        np.concatenate([data.train, data.valid, data.test]))
    # build val graph
    val_graph, val_rel, val_norm = utils.build_graph(
        num_nodes, num_rels, valid_data)
    val_deg = val_graph.in_degrees(
        range(val_graph.number_of_nodes())).float().view(-1, 1)
    val_rel = torch.from_numpy(val_rel)
    val_norm = utils.node_norm_to_edge_norm(val_graph, torch.from_numpy(val_norm).view(-1, 1))
    val_adj_list, val_degrees = utils.get_adj_and_degrees(num_nodes, valid_data)

    # build adj list and calculate degrees for sampling
    adj_list, degrees = utils.get_adj_and_degrees(num_nodes, train_data)

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    forward_time = []
    backward_time = []

    # training loop
    print("start training...")
    if args.model_state_file != "":
        print("start loeading checkpoint...", args.model_state_file)
        checkpoint = torch.load(args.model_state_file)
        model.load_state_dict(checkpoint['state_dict'])

    epoch = 0
    best_acc = 0.
    accuracies = []
    f1s = []
    train_records =  open("train_records.txt", "w")
    val_records = open("val_records.txt", "w")
    while True:
        model.train()
        epoch += 1

        # perform edge neighborhood sampling to generate training graph and data
        g, node_id, edge_type, node_norm, pos_samples, neg_samples = \
            utils.generate_sampled_graph_and_labels(
                train_data, args.graph_batch_size, args.graph_split_size,
                num_rels, adj_list, degrees, args.negative_sample,
                args.edge_sampler)
        # print("Done edge sampling for training")

        # set node/edge feature
        node_id = torch.from_numpy(node_id).view(-1, 1).long()
        edge_type = torch.from_numpy(edge_type)
        edge_norm = utils.node_norm_to_edge_norm(g, torch.from_numpy(node_norm).view(-1, 1))
        pos_samples, neg_samples = torch.from_numpy(pos_samples), torch.from_numpy(neg_samples)
        deg = g.in_degrees(range(g.number_of_nodes())).float().view(-1, 1)

        if use_cuda:
            node_id, deg = node_id.cuda(), deg.cuda()
            edge_type, edge_norm = edge_type.cuda(), edge_norm.cuda()
            pos_samples, neg_samples = pos_samples.cuda(), neg_samples.cuda()

        t0 = time.time()
        recon = model(g, node_id, edge_type, edge_norm)
        loss, recon_loss, kl, accu, f1 = model.get_loss(recon, pos_samples, neg_samples)
        train_records.write("{:d};{:.4f};{:.4f};{:.4f}\n".format(epoch,loss,recon_loss,kl))
        train_records.flush()
        t1 = time.time()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)  # clip gradients
        optimizer.step()
        t2 = time.time()
        forward_time.append(t1 - t0)
        backward_time.append(t2 - t1)
        print(
            "Epoch {:04d} | Loss {:.4f} | Recon loss {:.4f} | KL {:.4f} | acc {:.4f} | f1 {:.4f} | Forward {:.4f}s | Backward {:.4f}s".
                format(epoch, loss.item(), recon_loss.item(), kl.item(), accu, f1,
                       forward_time[-1], backward_time[-1]))

        optimizer.zero_grad()

        # validation with monte carlo sampling
        if epoch % args.evaluate_every == 1:
            model.eval()
            print("start eval")
            _accuracy_l = []
            _f1_l = []
            for i in range(args.eval_batch_size):
                val_g, val_node_id, val_edge_type, val_node_norm, val_pos_samples, val_neg_samples = utils.generate_sampled_graph_and_labels(
                    valid_data, args.graph_batch_size, args.graph_split_size,
                    num_rels, val_adj_list, val_degrees, args.negative_sample,
                    args.edge_sampler)
                # print("Done edge sampling for validation")
                val_node_id = torch.from_numpy(val_node_id).view(-1, 1).long()
                val_edge_type = torch.from_numpy(val_edge_type)
                val_edge_norm = utils.node_norm_to_edge_norm(val_g, torch.from_numpy(val_node_norm).view(-1, 1))
                val_pos_samples, val_neg_samples = torch.from_numpy(val_pos_samples), torch.from_numpy(val_neg_samples)
                if use_cuda:
                    val_node_id = val_node_id.cuda()
                    val_edge_type, val_edge_norm = val_edge_type.cuda(), val_edge_norm.cuda()
                    val_pos_samples, val_neg_samples = val_pos_samples.cuda(), val_neg_samples.cuda()

                recon = model(val_g, val_node_id, val_edge_type, val_edge_norm)
                _, _, _, _accu, _f1 = model.get_loss(recon, val_pos_samples, val_neg_samples)
                _accuracy_l.append(_accu)
                _f1_l.append(_f1)
            accuracies.append(np.mean(_accuracy_l))
            f1s.append(np.mean(_f1_l))
            print(
                "[EVAL] Epoch {:04d} | acc {:.4f} | f1  {:.4f} ".
                    format(epoch, accuracies[-1], f1s[-1]))
            val_records.write("{:d};{:.4f};{:.4f}\n".format(epoch,accuracies[-1], f1s[-1]))
            val_records.flush()
            # save best model
            if accuracies[-1] < best_acc:
                if epoch >= args.n_epochs:
                    break
            else:
                best_acc = accuracies[-1]
                torch.save({'state_dict': model.state_dict(), 'epoch': epoch},
                           'model_state.pth')

    print("training done")
    print("Mean forward time: {:4f}s".format(np.mean(forward_time)))
    print("Mean Backward time: {:4f}s".format(np.mean(backward_time)))
    val_records.close()
    train_records.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GCN_VAE for Knowledge Graph ')
    parser.add_argument("--dropout", type=float, default=0.2,
                        help="dropout probability")
    parser.add_argument("--n-hidden", type=int, default=500,
                        help="number of hidden units")
    parser.add_argument("--gpu", type=int, default=-1,
                        help="gpu")
    parser.add_argument("--lr", type=float, default=1e-2,
                        help="learning rate")
    parser.add_argument("--n-bases", type=int, default=20,
                        help="number of weight blocks for each relation")
    parser.add_argument("--n-layers", type=int, default=2,
                        help="number of propagation rounds")
    parser.add_argument("--n-epochs", type=int, default=20000,
                        help="number of minimum training epochs")
    parser.add_argument("-d", "--dataset", type=str, required=True,
                        help="dataset to use")
    parser.add_argument("--eval-batch-size", type=int, default=5,
                        help="batch size when evaluating")
    parser.add_argument("--regularization", type=float, default=0.01,
                        help="regularization weight")
    parser.add_argument("--grad-norm", type=float, default=1.0,
                        help="norm to clip gradient to")
    parser.add_argument("--graph-batch-size", type=int, default=800,
                        help="number of edges to sample in each iteration")
    parser.add_argument("--graph-split-size", type=float, default=0.5,
                        help="portion of edges used as positive sample")
    parser.add_argument("--negative-sample", type=int, default=3,
                        help="number of negative samples per positive sample")
    parser.add_argument("--evaluate-every", type=int, default=100,
                        help="perform evaluation every n epochs")
    parser.add_argument("--edge-sampler", type=str, default="neighbor",
                        help="type of edge sampler: 'uniform' or 'neighbor'")
    parser.add_argument("--model-state-file", type=str, default="",
                        help="model checkpoint to load")

    args = parser.parse_args()
    print(args)
    main(args)
