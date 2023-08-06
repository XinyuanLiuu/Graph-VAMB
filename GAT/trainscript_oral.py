if __name__ == '__main__':
    import sys
    import vamb4
    from datetime import datetime
    import torch as _torch
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    now = datetime.now()
    NUM_NEIGHBORS = [4]
    BETA = 230
    GAMMA_list = [0.001,0.0025]
    NEPOCH_list = [300,400]
    BATCH_STEPS = [25,75,150,225]
    LRATE_list = [0.0001]
    # LRATE = 0.001
    dataset_name = 'oral'
    date = '0515'

    print('Runing Vamb with graph on: ' + dataset_name)
    print('The time is: ' + str(now))
    print('Error_free data')
    print('Loading TNF and Depths')
    composition = vamb4.parsecontigs.Composition.load("/home/projects/cpr_10006/projects/xinyuan/data/{}/errorfree/vambout/composition.npz".format(dataset_name))
    abundance = vamb4.parsebam.Abundance.load("/home/projects/cpr_10006/projects/xinyuan/data/{}/errorfree/vambout/abundance.npz".format(dataset_name), composition.metadata.refhash)
    identifiers = composition.metadata.identifiers
    edge_index, degree_array = vamb4.graph.neighbors_dict('/home/projects/cpr_10006/projects/ptracker/tmp/neighbour_files/{}/ptracker_spades_ef/neighbours_dg_10_only_TPs.txt'.format(dataset_name),list(identifiers))
    print('Create dataloader and mask')
    print('Number of neighbors in multihops:',NUM_NEIGHBORS)

    (dataloader, mask, identifiers) = vamb4.encode.make_dataloader(
        abundance.matrix, # Coab
        composition.matrix, # TNF
        edge_index, #Edge_index in coo format, (2,N_edges)
        degree_array,
        composition.metadata.identifiers, # identifiers of contigs
        composition.metadata.lengths, # Sequence lengths (used in loss function)
        NUM_NEIGHBORS,
        256, # Batch size
        False, # False here means copy underlying memory. We want this.
        True # Run on GPU
    )
    for NEPOCH in NEPOCH_list:
        for GAMMA in GAMMA_list:
            for LRATE in LRATE_list:
            
                LOG_PATH = 'graph/{}/errorfree/log_{}_{}_{}_{}_{}.txt'.format(dataset_name,dataset_name,date,NEPOCH,GAMMA,LRATE)

                sys.stdout = open(LOG_PATH, 'w')
                print('\tNumber of neighbors in multihops:',NUM_NEIGHBORS)
                print('\tNumber of original contigs: ' + str(abundance.matrix.shape[0]))
                print('\tNumber of contigs remaining: ' + str(len(dataloader.data.x)))
                print('\tStarting to train VAE')

                start = datetime.now()

                vae = vamb4.encode.VAE(nsamples=len(abundance.samplenames), beta=BETA, gamma=GAMMA,cuda=True)
                train_latent = vae.trainmodel(
                    dataloader,
                    nepochs=NEPOCH,
                    batchsteps=BATCH_STEPS,
                    lrate=LRATE,
                    logfile=sys.stdout,
                    modelfile='graph/{}/errorfree/model_{}_{}_{}_{}_{}.pt'.format(dataset_name,dataset_name,date,NEPOCH,GAMMA,LRATE),
                    num_neighbors_=NUM_NEIGHBORS,
                    use_z=True,
                )

                end = datetime.now()
                print('\tThe end time is: ' + str(end))
                print('\tThe training costs: ' + str(end-start))
                
                # Extract the encoding from the VAE
                print('\tEncoding the latent representation')
                latent = vae.encode(dataloader,NUM_NEIGHBORS)

                # save the encoding of latent space
                print('\tSave the latent space')
                _torch.save(latent,'graph/{}/errorfree/latent_{}_{}_{}_{}_{}.npz'.format(dataset_name,dataset_name,date,NEPOCH,GAMMA,LRATE))

                # save the encodings in the training process
                print('\tTrain epoch latent saving')
                _torch.save(train_latent, 'graph/{}/errorfree/train_latent_{}_{}_{}_{}_{}.npz'.format(dataset_name,dataset_name,date,NEPOCH,GAMMA,LRATE))

                # Create an object which can cluster the latent space
                print('\tClustering on latent representation')
                clusterer = vamb4.cluster.ClusterGenerator(
                    latent,
                    cuda=False # Run on GPU
                )
                print('\tMin successful thresholds detected: ' + str(clusterer.minsuccesses))
                print('\tUsing CUDA: ' + str(clusterer.cuda))
                print('\tThe max number of step is: ' + str(clusterer.maxsteps))

                # Convert the clusterer from an iterator of Cluster object, to an iterator of (str, list[str]).
                renamed = (
                    (str(cluster_index + 1), [composition.metadata.identifiers[i] for i in members])
                    for (cluster_index, (_, members)) in enumerate(map(lambda x: x.as_tuple(), clusterer))
                )

                # Now do the so-called binsplitting, mentioned in my thesis and in the Vamb paper
                sep = 'C'
                print('\tThe separator is: ' + sep)
                binsplit = vamb4.vambtools.binsplit(renamed, separator=sep)

                # Now, we write the clusters to disk.
                print('\tWriting the cluster result')
                with open("graph/{}/errorfree/clusters_{}_{}_{}_{}_{}.tsv".format(dataset_name,dataset_name,date,NEPOCH,GAMMA,LRATE), "w") as file:
                    for (clustername, contigs) in binsplit:
                        for contig in contigs:
                            print(clustername, contig, sep='\t', file=file)

    print('Done')    
