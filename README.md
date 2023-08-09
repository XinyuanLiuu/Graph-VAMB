# Graph VAMB
Hi, This is Xinyuan's master thesis project about Metagenomic Binning using Graph Neural Networks. 

Metagenomic is an area of research about genomes recovered from the environment. Metagenomic binning is a process of clustering the assembled long contigs from DNA reads, that is, grouping contigs that come from the same organism together and getting the reconstructed complete genome. The composition and abundance of contigs have proven useful in measuring contig similarity. VAMB utilizes both composition and abundance alongside a Variational Autoencoder (VAE) to encode contigs in a latent space and perform clustering. Additionally, the adjacency of the assembly graph also provides valuable information regarding the connections between contigs during the assembly process

The primary objective is to develop a graph model that can incorporate both content features (composition and abundance) and graph features into the encoding process, hence enhancing the latent embedding which not only reflects the similarity of content but also the adjacency information in assembly graph.

This repository includes three different implementations of GNN model based on VAMB, which contains different graph layers and graph-based loss. 

## Graph construction from assembly graph--Propagated assembly graph
The assembly graph we use for graph construction is from assembler 'metaSPAdes'. After splitting DNA reads into k-mers, the metagenomic assemblers turn k-mers into a de Bruijn graph according to the overlap between them, and merge the non-branching path into one single sequence, which we refer to as a contig. In the assembly graph, two contigs are connected if they have overlapping regions.
![meta_assembly](https://github.com/XinyuanLiuu/GraphVAMB/assets/67774133/cf3a41b6-4335-4fc6-ad71-3f6b251f0a89)

Before construction, we first filter out contigs that are shorter than 2000bp to reduce the disturbance of noise. To avoid the loss of adjacency information, we applied an propagation strategy before filtering: given a propagation distance k, if the distance between two contigs in the assembly graph is shorter than k edges, we add a new edge between them. 
<img width="489" alt="image" src="https://github.com/XinyuanLiuu/GraphVAMB/assets/67774133/4f198ca6-fa7c-4c33-8a42-c77fa76d4c0c">

Given a constructed graph $\mathcal{G}=(\mathcal{V}, \mathcal{E})$, $\mathcal{V}$ denotes the set of nodes, that are assembled contigs longer than 2000bp in assembly graph, $mathcal{E}$ denotes the sets of edges, that are edges in propagated assembly graph. The feature of each node is according to TNF and abundance of each contig, which is also the input feature of VAMB. We treat each edge equally.

## Cosdistloss-VAMB
Cosdistloss-VAMB model keeps the main structure of VAMB, but applies an additional graph-based loss 'Cosdistloss', where we calculate the cosine distance between the inferred latent encoding of connected contigs.

$$    \mathcal{L}_
{D} = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \frac{1}{2} \left(1-\hat{\mu_i}^{kT}(\frac{1}{|N_{i}|}\sum_{j \in N_{i}}\hat{\mu_j}^{k-1})\right)$$

where $\mathcal{B}$ is the minibatch data, $N_i$ represents the neighbors of contig $i$, $\hat{\mu_i}^k$ is the L2-normalized $\mu$ of latent distribution for contig $i$ in k epoch. In this formula, the graph-loss for contig $C_i$ is the cosine distance between it and the average $\mu$ in the previous epoch of its neighbors $N_i$. Here we define cosine distance as $0.5 * (1-i^Tj)$ to limit it between 0-1 (0 means the most similar).

## GraphSAGE-VAMB
GraphSage-VAMB uses the GraphSage message passing layer to replace the original linear layer in VAE's encoder, and applies a graph-based loss as well. 

$$\mathbf{x}_
{i}^{\prime}=\gamma\left(\mathbf{x}_
{i}, \bigoplus_{j \in \mathcal{N}(i)} \phi\left(\mathbf{x}_
i,\mathbf{x}_ {j}, \mathbf{e}_ {ji}\right)\right)$$

It consists of two parts: message aggregation and update. During message aggregation, it applies a differentiable function $\phi$ on each edge $e_{ji}$ to project each neighbor to another space, then the projected neighbors' features are aggregated by a defined differentiable, permutation invariant aggregation function $\bigoplus$, such as mean, max. It provides an aggregation feature of all sampled neighbors, which could be used to update target node $\mathbf{x}_i$ using the updation function $\gamma$. Typically, in my implementation, the update of each node in each hidden layer is according to the formula below:

$$\mathbf{x}_ {i}^{\prime}=\mathbf{W} \cdot concat\left(\mathbf{x}_ {i},  mean_{j \in \mathcal{N}(i)} \mathbf{x}_j\right)$$

For the sage-layer encoder, because of the edges sampling process in each minibatch, I implement another form of graph-loss "Sage-loss", where the divergence is measured in the unit of edge:
$$\mathcal{L}_ {pos} = \sum_{e_{ij} \in \mathcal{E}_ P} \left(-log(sigmoid(\hat{z}_{i}^{kT}\hat{z}_j^k))\right)$$

$$\mathcal{L}_ {neg} = \sum_{e_{ij} \in \mathcal{E}_ N} \left(-log(1-sigmoid(\hat{z}_{i}^{kT}\hat{z}_j^k))\right)$$

$$\mathcal{L}_ {D} = \frac{1}{|\mathcal{E}_ P|} (\mathcal{L}_ {pos} + \mathcal{L}_{neg})$$
In this loss function, I consider both the positive and negative edges. The positive edges are $\mathcal{E}_P$ that are sampled in one minibatch, while the negative edges are $\mathcal{E}_N$ that are randomly sampled pairs of unconnected contigs. In my implementation, the number of negative edges is set to three times the number of positive edges, since negative edges are much more than positive edges. As the graph loss scale is calculated based on each edge, the total loss $\mathcal{L}_D$ is averaged over the number of positive edges.

![Loss_process](https://github.com/XinyuanLiuu/GraphVAMB/assets/67774133/56de3b83-16c7-4b22-8994-8a1aba98ebfc)

## GAT-VAMB
GAT-VAMB uses the same 'Sage-loss' as GraphSage-VAMB, but applies different message passing layer in graph encoder: the Graph Attention layer.

The hidden feature of nodes are updated by:
$$\mathbf{x}_ i^{\prime}=\alpha_{i, i} \boldsymbol{\Theta} \mathbf{x}_ i+\sum_{j \in \mathcal{N}(i)} \alpha_{i, j} \boldsymbol{\Theta} \mathbf{x}_j$$

$$\alpha_{i, j}=\frac{\exp \left(LeakyReLU\left(\mathbf{a}^{\top}\left \[\boldsymbol{\Theta} \mathbf{x}_ i \| \boldsymbol{\Theta} \mathbf{x}_ j\right\] \right)\right)}{\sum_{k \in \mathcal{N}(i) \cup\{i\}} \exp \left(LeakyReLU\left(\mathbf{a}^{\top}\left\[\boldsymbol{\Theta} \mathbf{x}_i \| \boldsymbol{\Theta} \mathbf{x}_k\right\]\right)\right)}$$

where $\alpha_{i,j}$ are attention coefficients, which measure the coefficient for each neighbor of one target node.The computation of it takes into consider the feature of this neighbor and all other neighbors, with trainable parameter matrix $\Theta$. It can automatically calculate the weight for each neighbor, rather than treat all of them equally. 

