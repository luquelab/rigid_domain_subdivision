import os
import time

from prody import *
from sklearnex import patch_sklearn
patch_sklearn()
import numpy as np
import matplotlib.pyplot as plt
import psutil
from scipy import sparse
from input import *


def subdivide_model(pdb, cluster_start, cluster_stop, cluster_step):

    print('Loading Model')
    sims = sparse.load_npz('../results/models/' + pdb + 'sims.npz')
    calphas = loadAtoms('../results/models/' + 'calphas_' + pdb + '.ag.npz')



    if not os.path.exists('../results/subdivisions/' + pdb):
        os.mkdir('../results/subdivisions/' + pdb)

    print('Spectral Clustering')
    from input import onlyTs, T
    if onlyTs:
        n_range = np.array([10*T+2, 20*T, 30*T, 30*T+2, 60*T])
    else:
        n_range = np.arange(cluster_start, cluster_stop+cluster_step, cluster_step)
    n_evecs = max(n_range)

    from input import prebuilt_embedding
    if prebuilt_embedding and os.path.exists('../results/models/' + pdb + 'embedding.npy'):
        maps = np.load('../results/models/' + pdb + 'embedding.npy')
        if maps.shape[1] < cluster_stop:
            print('Insufficient Eigenvectors For # Of Clusters')
            quit()
    else:
        if prebuilt_embedding:
            print('No saved embedding found, rebuilding')
        start = time.time()
        maps = embedding(n_evecs, sims)
        end = time.time()
        print(end - start, ' Seconds')
        print(maps.shape)
        from sklearn.preprocessing import normalize
        normalize(maps, copy=False)
        np.save('../results/models/' + pdb + 'embedding.npy', maps)

    start = time.time()
    from input import cluster_method as method
    labels, scores, var, ntypes = cluster_embedding(n_range, maps, calphas, method)
    end = time.time()
    print(end - start, ' Seconds')

    print('Plotting')
    fig, ax = plt.subplots(3, 1, figsize=(18, 10), sharex=True)
    fig.suptitle(pdb)
    ax[0].scatter(n_range, scores, marker='D', label='mbk')
    ax[0].plot(n_range, scores)
    ax[1].plot(n_range, ntypes)
    ax[1].scatter(n_range, ntypes)
    ax[1].set_ylabel('# of unique clusters')
    ax[2].plot(n_range, var)
    ax[2].scatter(n_range, var)
    ax[2].set_ylabel('Variance In Cluster Size')
    ax[0].axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    nc = str(n_range[np.argmax(scores)])
    maxticks = 100
    step = max(int((cluster_stop - cluster_start)/maxticks),2)
    ax[0].set_xticks(np.arange(cluster_start, cluster_stop, step))
    ax[2].set_xlabel('n_clusters')
    ax[0].set_ylabel('Silhouette Score')
    ax[0].legend()
    fig.tight_layout()
    print(pdb + '_' + nc + '_domains.png')
    fig.tight_layout()
    plt.savefig('../results/subdivisions/' + pdb + '_' + nc + '_domains.png')
    plt.show()

    return calphas, labels

def embedding(n_evecs, sims):
    print('Performing Spectral Embedding')
    from sklearn.manifold import spectral_embedding
    from scipy.sparse.csgraph import connected_components
    print(connected_components(sims))
    X_transformed = spectral_embedding(sims, n_components=n_evecs, drop_first=False)
    print('Memory Usage: ', psutil.virtual_memory().percent)
    return X_transformed

def cluster_embedding(n_range, maps, calphas, method):
    print('Clustering Embedded Points')

    from sklearn.cluster import k_means, AgglomerativeClustering
    #from sklearn.cluster import MiniBatchKMeans, AgglomerativeClustering
    #from sklearn.metrics import pairwise_distances
    # from sklearn_extra.cluster import KMedoids

    #from sklearn.metrics import silhouette_score
    #from sklearn.metrics import davies_bouldin_score
    from score import median_score, cluster_types
    from score import calcCentroids
    from sklearn.preprocessing import normalize

    randmaps = np.random.randn(maps.shape[0]*2, maps.shape[1])


    labels = []
    scores = []
    variances = []
    numtypes = []
    print('mapshape', maps.shape)
    for n in range(len(n_range)):
        n_clusters = n_range[n]
        emb = maps[:, :n_clusters]
        normalize(emb, copy=False)
        embrand = randmaps[:, :n_clusters]
        normalize(embrand,copy=False)
        print('Clusters: ' + str(n_clusters))
        start1 = time.time()
        # mbk = MiniBatchKMeans(n_clusters=n_clusters, batch_size=4096, n_init=10, reassignment_ratio=0.15, max_no_improvement=10).fit(maps[:, :n_clusters])
        # mbk = DBSCAN(n_clusters=n_clusters, eps=0.25).fit(maps[:, :n_clusters])
        # label = mbk.labels_
        # centroids = mbk.cluster_centers_
        print(method)
        print('emshape', emb.shape)
        if method == 'discretize':
            label = discretize(emb)
            print('labelshape',label.shape)
            centroids = calcCentroids(emb, label, n_clusters)
            # print('centroids', centroids.shape)
        elif method == 'kmeans':
            centroids, label, _, n_iter = k_means(emb, n_clusters=n_clusters, n_init=10, tol=1e-8,
                                                  return_n_iter=True)
        elif method == 'both':
            label = discretize(emb)
            centroids = calcCentroids(emb, label, n_clusters)
            centroids, label, _, n_iter = k_means(emb, n_clusters=n_clusters, init=discreteInit,
                                                      return_n_iter=True)
        elif method == 'agg':
            clust = AgglomerativeClustering(n_clusters=n_clusters, linkage='single')
            label = clust.fit_predict(emb)
            centroids = calcCentroids(emb, label, n_clusters)
        else:
            print('method should be kmeans or discretize. Defaulting to kmeans')

        # normalize(centroids)
        labels.append(label)
        cl = np.unique(label)
        print(cl.shape)
        end1 = time.time()

        # start2 = time.time()
        # centroids, label, _, n_iter = k_means(maps[:, :n_clusters], n_clusters=n_clusters, n_init=10, tol=1e-8, return_n_iter=True)
        # end2 = time.time()
        # print('mbk improvement:' + str((end1-start1)/(end2-start2)))
        # print(n_iter)
        # kmed = KMedoids(n_clusters=n_clusters).fit(maps[:, :n_clusters])
        # _, label, _ = spherical_k_means(maps[:, :n_clusters], n_clusters=n_clusters)

        print('Scoring')
        # testScore = median_score(maps[:, :n_clusters], centroids)
        testScore_rand = median_score(embrand, centroids)
        testScore = median_score(emb, centroids)/testScore_rand
        # testScore = davies_bouldin_score(maps[:, :n_clusters], label)
        # scores_km.append(testScore)
        # var, ntypes = cluster_types(label)
        # variances.append(var)
        # numtypes.append(ntypes)

        scores.append(testScore)
        var, ntypes = cluster_types(label)
        variances.append(var)
        numtypes.append(ntypes)
        print('Memory Usage: ', psutil.virtual_memory().percent)

        print('Saving Results')
        nc = str(n_range[n])
        np.savez('../results/subdivisions/' + pdb + '/' + pdb + '_' + nc + '_results', labels=label, score=testScore,
                 var=var, ntypes=ntypes, n=n, method=cluster_method)

    best = np.argpartition(scores, -5)[-5:]  # indices of 4 best scores
    for ind in best:
        writePDB('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(n_range[ind]) + '_domains.pdb', calphas,
                 beta=labels[ind],
                 hybrid36=True)

    return labels, scores, variances, numtypes

def discreteInit(vectors, n_clusters, *, copy=False, max_svd_restarts=30, n_iter_max=20, random_state=None):
    from score import calcCentroids
    label = discretize(vectors)
    centroids = calcCentroids(vectors, label, n_clusters)
    return centroids

def discretize(
    vectors, n_clusters=10, *, copy=False, max_svd_restarts=30, n_iter_max=20, random_state=None
):
    """Search for a partition matrix which is closest to the eigenvector embedding.
    This implementation was proposed in [1]_.
    Parameters
    ----------
    vectors : array-like of shape (n_samples, n_clusters)
        The embedding space of the samples.
    copy : bool, default=True
        Whether to copy vectors, or perform in-place normalization.
    max_svd_restarts : int, default=30
        Maximum number of attempts to restart SVD if convergence fails
    n_iter_max : int, default=30
        Maximum number of iterations to attempt in rotation and partition
        matrix search if machine precision convergence is not reached
    random_state : int, RandomState instance, default=None
        Determines random number generation for rotation matrix initialization.
        Use an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.
    Returns
    -------
    labels : array of integers, shape: n_samples
        The labels of the clusters.
    References
    ----------
    .. [1] `Multiclass spectral clustering, 2003
           Stella X. Yu, Jianbo Shi
           <https://www1.icsi.berkeley.edu/~stellayu/publication/doc/2003kwayICCV.pdf>`_
    Notes
    -----
    The eigenvector embedding is used to iteratively search for the
    closest discrete partition.  First, the eigenvector embedding is
    normalized to the space of partition matrices. An optimal discrete
    partition matrix closest to this normalized embedding multiplied by
    an initial rotation is calculated.  Fixing this discrete partition
    matrix, an optimal rotation matrix is calculated.  These two
    calculations are performed until convergence.  The discrete partition
    matrix is returned as the clustering solution.  Used in spectral
    clustering, this method tends to be faster and more robust to random
    initialization than k-means.
    """

    from scipy.sparse import csc_matrix
    from scipy.linalg import LinAlgError
    from sklearn.utils import check_random_state, as_float_array

    random_state = check_random_state(random_state)

    vectors = as_float_array(vectors, copy=copy)

    eps = np.finfo(float).eps
    n_samples, n_components = vectors.shape

    # Normalize the eigenvectors to an equal length of a vector of ones.
    # Reorient the eigenvectors to point in the negative direction with respect
    # to the first element.  This may have to do with constraining the
    # eigenvectors to lie in a specific quadrant to make the discretization
    # search easier.
    norm_ones = np.sqrt(n_samples)
    for i in range(vectors.shape[1]):
        vectors[:, i] = (vectors[:, i] / np.linalg.norm(vectors[:, i])) * norm_ones
        if vectors[0, i] != 0:
            vectors[:, i] = -1 * vectors[:, i] * np.sign(vectors[0, i])

    # Normalize the rows of the eigenvectors.  Samples should lie on the unit
    # hypersphere centered at the origin.  This transforms the samples in the
    # embedding space to the space of partition matrices.
    vectors = vectors / np.sqrt((vectors ** 2).sum(axis=1))[:, np.newaxis]

    svd_restarts = 0
    has_converged = False

    # If there is an exception we try to randomize and rerun SVD again
    # do this max_svd_restarts times.
    while (svd_restarts < max_svd_restarts) and not has_converged:

        # Initialize first column of rotation matrix with a row of the
        # eigenvectors
        rotation = np.zeros((n_components, n_components))
        rotation[:, 0] = vectors[random_state.randint(n_samples), :].T

        # To initialize the rest of the rotation matrix, find the rows
        # of the eigenvectors that are as orthogonal to each other as
        # possible
        c = np.zeros(n_samples)
        for j in range(1, n_components):
            # Accumulate c to ensure row is as orthogonal as possible to
            # previous picks as well as current one
            c += np.abs(np.dot(vectors, rotation[:, j - 1]))
            rotation[:, j] = vectors[c.argmin(), :].T

        last_objective_value = 0.0
        n_iter = 0

        while not has_converged:
            n_iter += 1

            t_discrete = np.dot(vectors, rotation)

            labels = t_discrete.argmax(axis=1)
            vectors_discrete = csc_matrix(
                (np.ones(len(labels)), (np.arange(0, n_samples), labels)),
                shape=(n_samples, n_components),
            )

            t_svd = vectors_discrete.T * vectors

            try:
                U, S, Vh = np.linalg.svd(t_svd)
            except LinAlgError:
                svd_restarts += 1
                print("SVD did not converge, randomizing and trying again")
                break

            ncut_value = 2.0 * (n_samples - S.sum())
            if (abs(ncut_value - last_objective_value) < eps) or (n_iter > n_iter_max):
                has_converged = True
            else:
                # otherwise calculate rotation and continue
                last_objective_value = ncut_value
                rotation = np.dot(Vh.T, U.T)

    if not has_converged:
        raise LinAlgError("SVD did not converge")
    return labels