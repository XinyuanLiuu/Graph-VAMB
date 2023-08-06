import numpy as np
import torch as _torch


def neighbors_dict(
        graph_path: str,
        identifiers: list,
):
    source = []
    target = []
    degree = np.zeros(len(identifiers))


    # add other neighbor information from graph file
    with open(graph_path,'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            id0 = identifiers.index(parts[0])
            id1 = identifiers.index(parts[1])
            source += [id0]
            target += [id1]
            degree[id0] += 1

    edge_index = _torch.tensor([source,target],dtype=_torch.long)

    return edge_index, degree


def neighbors_embedding(
        graph_dict: dict, # whole graph dict
        nbatch: int, # size of batch
        nlatent: _torch.Tensor, # updated embeding of minibatch
        embeddings_old: _torch.Tensor, # whole embeding tensor in epoch k-1
        contigs_index: _torch.Tensor, # indexs of minibatch
):

    neighbors_emb = _torch.empty((nbatch,nlatent))
    degree_weights = _torch.empty(nbatch)

    # calculate average neighbor embedding for each contig in minibatch
    for i,contig_index in enumerate(list(contigs_index)):
        # get the index and degree of target contig
        contig_index = contig_index.item()
        degree = len(graph_dict[contig_index])
        degree_weights[i] = degree

        # if contig has neighbor, calculate the mean of its neighbors
        if degree > 0:
            neighbor_index = graph_dict[contig_index]
            embedings_nb = embeddings_old[neighbor_index,:]
            average_emb = embedings_nb.mean(axis=0) 

        # if contig has no neighbor, let neighbor embedding be itself
        else:
            average_emb = embeddings_old[contig_index,:].reshape(1,nlatent)

        neighbors_emb[i] = average_emb

    # create a weight according to degree, and positive proportional to degree
    degree_weights = (degree_weights/1).clip(max=1)

    return neighbors_emb, degree_weights

def cosine_distance(x1, x2, eps=1e-6):
    w1 = x1/(x1.norm(p=2, dim=1, keepdim=True) + eps)
    w2 = x2/(x2.norm(p=2, dim=1, keepdim=True) + eps)
    dist = (w1 * w2).sum(dim=1)
    return _torch.sigmoid(dist)



