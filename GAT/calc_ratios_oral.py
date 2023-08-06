import torch as _torch
import numpy as np
import vamb4
import json
import matplotlib.pyplot as plt

def _calc_distances(matrix: _torch.Tensor, index: int) -> _torch.Tensor:
    "Return vector of cosine distances from rows of normalized matrix to given row."
    dists = 0.5 - matrix.matmul(matrix[index])
    dists[index] = 0.0  # avoid float rounding errors
    return dists

def _calc_densities(
    histogram, pdf = vamb4.cluster._NORMALPDF
):
    """Given an array of histogram, smoothes the histogram."""
    pdf_len = len(pdf)

    densities = _torch.zeros(len(histogram) + pdf_len - 1)
    for i in range(len(densities) - pdf_len + 1):
        densities[i : i + pdf_len] += pdf * histogram[i]

    densities = densities[15:-15]

    return densities

def calc_ratios(matrix,genomes_idx,genome_contig_dict,all_genomes_type):
    ratios = dict()
    for genome_i in [all_genomes_type[idx] for idx in genomes_idx]:
        min_ratio = 999999999
        center = 0
        ingenome_idx = genome_contig_dict[genome_i]
        outgenome_idx = [i for i in range(len(matrix)) if i not in genome_contig_dict[genome_i]]
        for contig_id in ingenome_idx:
            dist = _calc_distances(matrix,contig_id)
            within_distances = dist[ingenome_idx]
            without_distances = dist[outgenome_idx]
            histogram_within = _torch.empty((180,))
            histogram_without = _torch.empty((180,))
            dens = []
            for (h, d) in ((histogram_within, within_distances), (histogram_without, without_distances)):
                _torch.histc(d, len(histogram_within), 0, 0.9, out=h)
                h[0] -= 1  # Remove distance to self
                dens.append(_calc_densities(h))
            densities_within, densities_without = dens
            sum_overlap = sum(np.stack((densities_within,densities_without),axis=1).min(axis=1)).clip(min=0)
            sum_within = sum(densities_within)
            ratio = sum_overlap/sum_within
            if ratio < min_ratio:
                center = contig_id
                min_ratio=ratio
        ratios[genome_i] = (min_ratio.item(),center,len(ingenome_idx))
    return ratios

def plot_ratio(matrix,genome_idx,ratios_dict,genome_contig_dict,name):
    genome_i = list(ratios_dict.keys())[genome_idx]
    center = ratios_dict[genome_i][1]
    ingenome_idx = genome_contig_dict[genome_i]
    outgenome_idx = [i for i in range(len(matrix)) if i not in genome_contig_dict[genome_i]]
    dist = _calc_distances(matrix,center)
    within_distances = dist[ingenome_idx]
    without_distances = dist[outgenome_idx]
    histogram_within = _torch.empty((180,))
    histogram_without = _torch.empty((180,))
    dens = []
    for (h, d) in ((histogram_within, within_distances), (histogram_without, without_distances)):
        _torch.histc(d, len(histogram_within), 0, 0.9, out=h)
        h[0] -= 1  # Remove distance to self
        dens.append(_calc_densities(h))
    densities_within, densities_without = dens
    sum_overlap = sum(np.stack((densities_within,densities_without),axis=1).min(axis=1)).clip(min=0)
    sum_within = sum(densities_within)
    ratio = sum_overlap/sum_within
    # print('genome:',genome_i)
    # print('Overlap/within ratio:',ratio)
    # print("Number of in-genome contigs before 0.3:",sum(within_distances < 0.3))
    # print("Number of in-genome contigs after 0.3:",sum(within_distances > 0.3))
    plt.plot([i*0.005 for i in range(180)], densities_within[:180])
    plt.plot([i*0.005 for i in range(180)], densities_without[:180])
    plt.ylim(0, 100)
    plt.xlim(0, 1)
    plt.title('{},{}, {}'.format(name,round(ratio.item(),3),genome_i))


dataset = 'oral'

#load genomr_contigs_dict
with open('/home/projects/cpr_10006/projects/xinyuan/data/{}/errorfree_overlap/genome_contigs_{}_dict.json'.format(dataset,dataset),'r') as f:
    genome_contigs = json.load(f)

# only use genomes have >10 contigs
all_genomes_type = []
for genome_type, contigs in genome_contigs.items():
    if len(contigs) > 10:
        all_genomes_type.append(genome_type)

# load the genome idx we randomly choose, need to be same for each running
random_idx = np.load('/home/projects/cpr_10006/projects/xinyuan/data/{}/errorfree_overlap/random_idx.npy'.format(dataset))

# load ratios dist of original vamb
with open('/home/projects/cpr_10006/projects/xinyuan/original/graph/{}/errorfree/ratios_{}_0504.json'.format(dataset,dataset), 'r') as f:
    ratios_ori = json.load(f)
ratios_ori_value = [ratio for ratio,_,_ in ratios_ori.values()]

# load latent space of original vamb
latent_ori = _torch.load('/home/projects/cpr_10006/projects/xinyuan/original/graph/{}/errorfree/latent_{}_0504.npz'.format(dataset,dataset))
matrix_ori = vamb4.cluster._normalize(latent_ori)


date = '0509'
beta_list = [280,300,320,340]
gamma_list = [0,0.001,0.002]

#initialize ratios_dict_list and ratios_value_dict_list
ratios_list_onemodel = [ratios_ori]
ratios_value_list_onemodel = [ratios_ori_value]
matrix_list = [matrix_ori]
y_label = ['original']
for beta in beta_list:
    for gamma in gamma_list:
        LATENT_PATH = 'graph/{}/errorfree/latent_{}_{}_{}_{}.npz'.format(dataset,dataset,date,beta,gamma)
        latent = _torch.load(LATENT_PATH)
        matrix = vamb4.cluster._normalize(latent)
        ratios = calc_ratios(matrix=matrix,genomes_idx=random_idx,genome_contig_dict=genome_contigs,all_genomes_type=all_genomes_type)
        
        ratios_list_onemodel.append(ratios)
        ratios_value_list_onemodel.append([ratio for ratio,_,_ in ratios.values()])
        matrix_list.append(matrix)
        
        with open('graph/{}/errorfree/ratios_{}_{}_{}.json'.format(dataset,date,beta,gamma), 'w') as f:
            json.dump(ratios, f)
        
        y_label.append(str(beta)+'_'+str(gamma))

plt.figure(figsize=(6,9))
plt.boxplot(ratios_value_list_onemodel,vert=False)
loc = range(1,len(y_label)+1)
plt.yticks(loc, y_label)
plt.xlim((0.3,1))
plt.ylabel('weights')
plt.title('{}, {}'.format(date,dataset))
plt.savefig('graph/{}/errorfree/overlap_ratio_{}.png'.format(dataset,date))

# plot the overlap inside specific genome
plt.figure(figsize=(25,20))
for i in range(len(y_label)):
    plt.subplot(5,4,i+1)
    plot_ratio(matrix_list[i],14,ratios_list_onemodel[i],genome_contigs,y_label[i])
plt.savefig('graph/{}/errorfree/overlap_{}.png'.format(dataset,date))
