from sklearn.neighbors import NearestNeighbors
import numpy as np



def get_kNN(archive, bc, n_neighbors):
    """
    Searches and samples the K-nearest-neighbors from the archive and a new behavior characterization
    returns the summed distance between input behavior characterization and the bc in the archive
    
    """

    archive = np.concatenate(archive).reshape(-1,1)
    neigh = NearestNeighbors(n_neighbors=n_neighbors)
    neigh.fit(archive)
    distances, idx = neigh.kneighbors(X = bc.reshape(-1,1), n_neighbors=n_neighbors)
    #k_nearest_neighbors = archive[idx].squeeze(0)

    return sum(distances.squeeze(0))


def add_bc_to_archive(bc_storage, archive, archive_prob):
    """
    For each behavior characterization in the storage it gets added to the archive by a given probability
    bc_storage = list of bc from the current population
    Probability = ARCHIVE_PROB
    
    """
    for bc in bc_storage:
        if np.random.random() <= archive_prob:
            archive.append(bc)
        
    return archive