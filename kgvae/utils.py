"""
Utility functions for link prediction
Most code is adapted from authors' implementation of RGCN link prediction:
https://github.com/MichSchli/RelationPrediction

"""

import numpy as np
import torch
import dgl
import torch.nn.functional as F
#######################################################################
#
# Utility function for building training and testing graphs
#
#######################################################################

def get_adj_and_degrees(num_nodes, triplets):
    """ Get adjacency list and degrees of the graph
    """
    adj_list = [[] for _ in range(num_nodes)]
    for i,triplet in enumerate(triplets):
        adj_list[triplet[0]].append([i, triplet[2]])
        adj_list[triplet[2]].append([i, triplet[0]])

    degrees = np.array([len(a) for a in adj_list])
    adj_list = [np.array(a) for a in adj_list]
    return adj_list, degrees

def sample_edge_neighborhood(adj_list, degrees, n_triplets, sample_size):
    """Sample edges by neighborhool expansion.

    This guarantees that the sampled edges form a connected graph, which
    may help deeper GNNs that require information from more than one hop.
    """
    edges = np.zeros((sample_size), dtype=np.int32)

    #initialize
    sample_counts = np.array([d for d in degrees])
    picked = np.array([False for _ in range(n_triplets)])
    seen = np.array([False for _ in degrees])

    for i in range(0, sample_size):
        weights = sample_counts * seen

        if np.sum(weights) == 0:
            weights = np.ones_like(weights)
            weights[np.where(sample_counts == 0)] = 0

        probabilities = (weights) / np.sum(weights)
        chosen_vertex = np.random.choice(np.arange(degrees.shape[0]),
                                         p=probabilities)
        chosen_adj_list = adj_list[chosen_vertex]
        seen[chosen_vertex] = True

        chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
        chosen_edge = chosen_adj_list[chosen_edge]
        edge_number = chosen_edge[0]

        while picked[edge_number]:
            chosen_edge = np.random.choice(np.arange(chosen_adj_list.shape[0]))
            chosen_edge = chosen_adj_list[chosen_edge]
            edge_number = chosen_edge[0]

        edges[i] = edge_number
        other_vertex = chosen_edge[1]
        picked[edge_number] = True
        sample_counts[chosen_vertex] -= 1
        sample_counts[other_vertex] -= 1
        seen[other_vertex] = True

    return edges

def sample_edge_uniform(adj_list, degrees, n_triplets, sample_size):
    """Sample edges uniformly from all the edges."""
    all_edges = np.arange(n_triplets)
    return np.random.choice(all_edges, sample_size, replace=False)

def generate_sampled_graph_and_labels(triplets, sample_size, split_size,
                                      num_rels, adj_list, degrees,
                                      negative_rate, sampler="uniform"):
    """Get training graph and signals
    First perform edge neighborhood sampling on graph, then perform negative
    sampling to generate negative samples
    """
    # perform edge neighbor sampling
    if sampler == "uniform":
        edges = sample_edge_uniform(adj_list, degrees, len(triplets), sample_size)
    elif sampler == "neighbor":
        edges = sample_edge_neighborhood(adj_list, degrees, len(triplets), sample_size)
    else:
        raise ValueError("Sampler type must be either 'uniform' or 'neighbor'.")

    # relabel nodes to have consecutive node ids
    edges = triplets[edges]
    src, rel, dst = np.array(edges).transpose()
    uniq_v, edges = np.unique((src, dst), return_inverse=True)
    src, dst = np.reshape(edges, (2, -1))
    relabeled_edges = np.stack((src, rel, dst)).transpose()

    # negative sampling
    samples, labels = negative_sampling(relabeled_edges, len(uniq_v),
                                        negative_rate)

    # further split graph, only half of the edges will be used as graph
    # structure, while the rest half is used as unseen positive samples
    split_size = int(sample_size * split_size)
    graph_split_ids = np.random.choice(np.arange(sample_size),
                                       size=split_size, replace=False)
    src = src[graph_split_ids]
    dst = dst[graph_split_ids]
    rel = rel[graph_split_ids]

    # build DGL graph
    g, rel, norm = build_graph_from_triplets(len(uniq_v), num_rels,
                                             (src, rel, dst))
    return g, uniq_v, rel, norm, samples, labels

def comp_deg_norm(g):
    g = g.local_var()
    in_deg = g.in_degrees(range(g.number_of_nodes())).float().numpy()
    norm = 1.0 / in_deg
    norm[np.isinf(norm)] = 0
    return norm

def build_graph_from_triplets(num_nodes, num_rels, triplets):
    """ Create a DGL graph. The graph is bidirectional because RGCN authors
        use reversed relations.
        This function also generates edge type and normalization factor
        (reciprocal of node incoming degree)
    """
    g = dgl.DGLGraph()
    g.add_nodes(num_nodes)
    src, rel, dst = triplets
    src, dst = np.concatenate((src, dst)), np.concatenate((dst, src))
    rel = np.concatenate((rel, rel + num_rels))
    edges = sorted(zip(dst, src, rel))
    dst, src, rel = np.array(edges).transpose()
    g.add_edges(src, dst)
    norm = comp_deg_norm(g)
    return g, rel, norm

def build_test_graph(num_nodes, num_rels, edges):
    src, rel, dst = np.array(edges).transpose()
    return build_graph_from_triplets(num_nodes, num_rels, (src, rel, dst))

def negative_sampling(pos_samples, num_entity, negative_rate):
    size_of_batch = len(pos_samples)
    num_to_generate = size_of_batch * negative_rate
    neg_samples = np.tile(pos_samples, (negative_rate, 1))
    labels = np.zeros(size_of_batch * (negative_rate + 1), dtype=np.float32)
    labels[: size_of_batch] = 1
    values = np.random.randint(num_entity, size=num_to_generate)
    choices = np.random.uniform(size=num_to_generate)
    subj = choices > 0.5
    obj = choices <= 0.5
    neg_samples[subj, 0] = values[subj]
    neg_samples[obj, 2] = values[obj]

    return np.concatenate((pos_samples, neg_samples)), labels

#######################################################################
#
# Utility function for evaluations
#
#######################################################################

def sort_and_rank(score, target):
    _, indices = torch.sort(score, dim=1, descending=True)
    indices = torch.nonzero(indices == target.view(-1, 1))
    indices = indices[:, 1].view(-1)
    return indices


def perturb_and_get_rank(embedding, w, a, r, b, test_size, batch_size=100, all_batches=True):
    """ Perturb one element in the triplets
    """
    n_batch = (test_size + batch_size - 1) // batch_size
    ranks = []
    if all_batches is False:
        n_batch = 1
    for idx in range(n_batch): #n_batch TODO
        batch_start = idx * batch_size
        batch_end = min(test_size, (idx + 1) * batch_size)
        batch_a = a[batch_start: batch_end]
        batch_r = r[batch_start: batch_end]
        emb_ar = embedding[batch_a] * w[batch_r]
        emb_ar = emb_ar.transpose(0, 1).unsqueeze(2) # size: D x E x 1
        emb_c = embedding.transpose(0, 1).unsqueeze(1) # size: D x 1 x V
        # out-prod and reduce sum
        out_prod = torch.bmm(emb_ar, emb_c) # size D x E x V
        score = torch.sum(out_prod, dim=0) # size E x V
        score = torch.sigmoid(score)
        target = b[batch_start: batch_end]
        ranks.append(sort_and_rank(score, target))

        _running_ranks = 1.0+torch.cat((ranks)).float()
        mrr = torch.mean(1.0 / _running_ranks)
        mr = torch.mean(_running_ranks)
        avg_count = []
        for hit in [1,5,10]:
            avg_count.append( torch.mean((_running_ranks <= hit).float()).item())
        print("batch {} / {}: MR : {:.6f} |  MRR : {:.6f} | Hit1: {:.6f} | Hit5: {:.6f} | Hit10: {:.6f}".format(
            idx, n_batch, mr.item(), mrr.item(), avg_count[0], avg_count[1], avg_count[2] ) )

    return torch.cat(ranks)

# return MRR (raw), and Hits @ (1, 3, 10)
def calc_mrr(embedding, w, test_triplets, hits=[], eval_bz=100, all_batches=True):
    with torch.no_grad():
        s = test_triplets[:, 0]
        r = test_triplets[:, 1]
        o = test_triplets[:, 2]
        test_size = test_triplets.shape[0]

        # perturb subject
        ranks_s = perturb_and_get_rank(embedding, w, o, r, s, test_size, eval_bz, all_batches)
        # perturb object
        ranks_o = perturb_and_get_rank(embedding, w, s, r, o, test_size, eval_bz, all_batches)

        ranks = torch.cat([ranks_s, ranks_o])
        ranks += 1 # change to 1-indexed

        mrr = torch.mean(1.0 / ranks.float())
        print("MRR (raw): {:.6f}".format(mrr.item()))

        for hit in hits:
            avg_count = torch.mean((ranks <= hit).float())
            print("Hits (raw) @ {}: {:.6f}".format(hit, avg_count.item()))
    return mrr.item()


#######################################################################
#
# Probability utils
#
#######################################################################

def gaussian_parameters(h, dim=-1):
    """
    Converts generic real-valued representations into mean and variance
    parameters of a Gaussian distribution

    Args:
        h: tensor: (batch, ..., dim, ...): Arbitrary tensor
        dim: int: (): Dimension along which to split the tensor for mean and
            variance

    Returns:
        m: tensor: (batch, ..., dim / 2, ...): Mean
        v: tensor: (batch, ..., dim / 2, ...): Variance
    """
    m, h = torch.split(h, h.size(dim) // 2, dim=dim)
    v = F.softplus(h) + 1e-8
    return m, v

def sample_gaussian(m, v):
    """
    Element-wise application reparameterization trick to sample from Gaussian

    Args:
        m: tensor: (batch, ...): Mean
        v: tensor: (batch, ...): Variance

    Return:
        z: tensor: (batch, ...): Samples
    """
    sqrt_v = torch.sqrt(v)
    sample = m + torch.randn_like(sqrt_v) * sqrt_v
    return sample

def log_normal_mixture(z, m, v):
    """
    Computes log probability of a uniformly-weighted Gaussian mixture.

    Args:
        z: tensor: (batch, dim): Observations
        m: tensor: (batch, mix, dim): Mixture means
        v: tensor: (batch, mix, dim): Mixture variances

    Return:
        log_prob: tensor: (batch,): log probability of each sample
    """
    log_prob = log_normal(z.unsqueeze(1), m, v)
    log_prob = log_mean_exp(log_prob, dim=-1)
    return log_prob

def log_normal(x, m, v):
    """
    Computes the elem-wise log probability of a Gaussian and then sum over the
    last dim. Basically we're assuming all dims are batch dims except for the
    last dim.

    Args:
        x: tensor: (batch_1, batch_2, ..., batch_k, dim): Observation
        m: tensor: (batch_1, batch_2, ..., batch_k, dim): Mean
        v: tensor: (batch_1, batch_2, ..., batch_k, dim): Variance

    Return:
        log_prob: tensor: (batch_1, batch_2, ..., batch_k): log probability of
            each sample. Note that the summation dimension is not kept
    """
    log_prob = -(x - m).pow(2) / (2 * v) - v.sqrt().log() - np.log(np.sqrt((2 * np.pi)))
    log_prob = torch.sum(log_prob, dim=-1)
    return log_prob

def log_sum_exp(x, dim=0):
    """
    Compute the log(sum(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which sum is computed

    Return:
        _: tensor: (...): log(sum(exp(x), dim))
    """
    max_x = torch.max(x, dim)[0]
    new_x = x - max_x.unsqueeze(dim).expand_as(x)
    return max_x + (new_x.exp().sum(dim)).log()

def log_mean_exp(x, dim):
    """
    Compute the log(mean(exp(x), dim)) in a numerically stable manner

    Args:
        x: tensor: (...): Arbitrary tensor
        dim: int: (): Dimension along which mean is computed

    Return:
        _: tensor: (...): log(mean(exp(x), dim))
    """
    return log_sum_exp(x, dim) - np.log(x.size(dim))

