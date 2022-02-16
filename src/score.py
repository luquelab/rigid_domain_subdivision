import numpy as np
from sklearn.metrics import pairwise_distances

def calcCentroids(X, labels, n_clusters):
    centroids = []
    for i in range(n_clusters):
        mask = (labels==i)
        if not np.any(mask):
            print('Some clusters unassigned')
            centroids.append((np.random.rand(n_clusters)))
        else:
            clust = X[mask,:]
            cent = np.mean(clust, axis=0)
            centroids.append(cent)

    return np.array(centroids)



def median_score(coords, centroids):
    from input import scoreMethod
    from sklearn.metrics import pairwise_distances
    dists = pairwise_distances(coords,centroids)
    d2min = np.partition(dists, kth=2)[:,:2]
    if scoreMethod is 'median':
        b = np.median(d2min[:,1])
        a = np.median(d2min[:,0])
    else:
        b = np.mean(d2min[:, 1])
        a = np.mean(d2min[:, 0])
    score = b/a
    return score

def cluster_types(labels):
    _, counts = np.unique(labels, return_counts=True)
    var = np.std(counts)
    ntypes = np.unique(counts).shape[0]
    return var, ntypes

def plotScores(pdb, n_range, save=False):
    import matplotlib.pyplot as plt
    import numpy as np
    scores = []
    vars = []
    ntypes = []
    for i in range(len(n_range)):
        nc = n_range[i]
        results = np.load('../results/subdivisions/' + pdb + '/' + pdb + '_' + str(nc) + '_results.npz')
        score = results['score']
        ntype = results['ntypes']
        var = results['var']
        scores.append(score)
        vars.append(var)
        ntypes.append(ntype)
    scores = np.array(scores)
    vars = np.array(vars)
    ntypes = np.array(ntypes)
    print('Plotting')
    fig, ax = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    ax[0].scatter(n_range, scores, marker='D', label=pdb)
    ax[0].plot(n_range, scores)
    ax[1].plot(n_range, ntypes)
    ax[1].scatter(n_range, ntypes)
    ax[1].set_ylabel('# of unique clusters')
    ax[2].plot(n_range, vars)
    ax[2].scatter(n_range, vars)
    ax[2].set_ylabel('Variance In Cluster Size')
    ax[0].axvline(x=n_range[np.argmax(scores)], label='Best Score', color='black')
    nc = str(n_range[np.argmax(scores)])
    # ax[0].set_xticks(n_range)
    ax[2].set_xlabel('n_clusters')
    ax[0].set_ylabel('Silhouette Score')
    ax[0].legend()
    fig.tight_layout()
    print(pdb + '_' + nc + '_domains.png')
    if save:
        plt.savefig('../results/subdivisions/' + pdb + '_' + nc + '_domains.png')
    plt.show()
