from prody import *
import os
import wget
import shutil
import gzip
from scipy.sparse.linalg import eigsh
import time
import numba as nb
import numpy as np
from scipy import sparse

def make_model(pdb, n_modes):
    from input import rebuild_hessian, rebuild_modes, sample
    os.chdir('../data/capsid_pdbs')
    filename = pdb + '_full.pdb'
    if not os.path.exists(filename):
        vdb_url = 'https://files.rcsb.org/download/' + pdb + '.pdb.gz'
        print(vdb_url)
        vdb_filename = wget.download(vdb_url)
        with gzip.open(vdb_filename, 'rb') as f_in:
            with open(filename, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)

    capsid = parsePDB(filename, biomol=True)
    calphas = capsid.select('ca').copy()
    print('Number Of Residues: ', calphas.getCoords().shape[0])

    os.chdir('../../src')

    anm = ANM(pdb + '_full')
    if not rebuild_hessian:
        if not os.path.exists('../results/models/' + pdb + 'hess.npz') or not os.path.exists('../results/models/' + pdb + 'kirch.npz'):
            print('No hessian found. Building Hessian.')
        else:
            anm._hessian = sparse.load_npz('../results/models/' + pdb + 'hess.npz')
            kirch = sparse.load_npz('../results/models/' + pdb + 'kirch.npz')
    else:
        anm.buildHessian(calphas, cutoff=7.5, kdtree=True, sparse=True)
        sparse.save_npz('../results/models/' + pdb + 'hess.npz', anm.getHessian())
        sparse.save_npz('../results/models/' + pdb + 'kirch.npz', anm.getKirchhoff())
        kirch = anm.getKirchhoff()

    if not rebuild_modes:
        if not os.path.exists('../results/models/' + pdb + 'hess.npz'):
            print('No evecs found. Calculating')
        else:
            anm = loadModel('../results/models/' + pdb + 'anm.npz')
            print('Slicing Modes up to ' + str(n_modes))
            print()
            evals = anm.getEigvals()[:n_modes].copy()
            evecs = anm.getEigvecs()[:,:n_modes].copy()
            print(evecs.shape)
    else:
        from scipy.sparse.linalg import lobpcg
        from pyamg import ruge_stuben_solver
        print('Calculating Normal Modes')
        import matplotlib.pyplot as plt

        start = time.time()
        evals, evecs = eigsh(anm._hessian, k=n_modes, sigma=0, which='LM')
        print('scipy', evals)
        end = time.time()
        print('scipy', end - start)
        print(evals[evals > 1e-8])
        fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        ax.scatter(np.arange(evals.shape[0]), evals, marker='D', label='Scipy')
        fig.tight_layout()
        plt.show()

        # start = time.time()
        # n = calphas.getCoords().shape[0]
        # ml = ruge_stuben_solver(anm.getHessian())
        # M = ml.aspreconditioner()
        # epredict = np.random.rand(3 * n, n_modes)
        # evals, evecs = lobpcg(anm.getHessian(), epredict, M=M, largest=False, tol=0)
        # print('lob', evals)
        # end = time.time()
        # print('lob', end - start)
        # print(evals[evals > 1e-8])
        # fig, ax = plt.subplots(1, 1, figsize=(16, 6))
        # ax.scatter(np.arange(evals.shape[0]), evals, marker='D', label='lob')
        # fig.tight_layout()
        # plt.show()

        anm._eigvals = evals
        anm._n_modes = len(evals)
        anm._eigvecs = evecs.copy
        anm._array = evecs.copy
        saveModel(anm, filename='../results/models/' + pdb + 'anm.npz')

    import matplotlib.pyplot as plt
    print('Plotting')
    fig, ax = plt.subplots(1, 1, figsize=(16, 6))
    ax.scatter(np.arange(evals.shape[0]), evals, marker='D', label='eigs')
    fig.tight_layout()
    plt.show()

    model = anm


    n_d = int(evecs.shape[0] / 3)

    kirch = kirch.tocoo()
    print('Calculating Covariance Matrix')
    if sample:
        from sampling import calcSample
        print('Sampling Method')
        coords = calphas.getCoords()
        d = calcSample(coords, evals, evecs, 2000, kirch.row, kirch.col)
        n_d = int(evecs.shape[0] / 3)
        d = sparse.coo_matrix((d, (kirch.row, kirch.col)), shape=(n_d, n_d))
    else:
        print('Direct Calculation Method')
        covariance = sparse.lil_matrix((n_d, n_d))
        df = sparse.lil_matrix((n_d, n_d))
        covariance = con_c(evals.copy(), evecs.copy(), covariance, kirch.row, kirch.col)
        covariance = covariance.tocsr()
        d = con_d(covariance, df, kirch.row, kirch.col)

    d = d.tocsr()
    d.eliminate_zeros()
    nnDistFlucts = np.mean(np.sqrt(d.data))

    sigma = 1 / (2 * nnDistFlucts ** 2)
    sims = -sigma * d ** 2
    data = sims.data
    data = np.exp(data)
    sims.data = data

    sparse.save_npz('../results/models/' + pdb + 'sims.npz', sims)

    saveAtoms(calphas, filename='../results/models/' + 'calphas_' + pdb)

    return sims, calphas

# @nb.njit(parallel=True)
# def cov(evals, evecs, i, j):
#     n_e = evals.shape[0]
#     n_d = evecs.shape[1]
#     tr1 = 0
#     tr2 = 0
#     tr3 = 0
#     for n in nb.prange(n_e):
#         l = evals[n]
#         tr1 += 1 / l * (evecs[3 * i, n] * evecs[3 * j, n] + evecs[3 * i + 1, n] * evecs[3 * j + 1, n] + evecs[
#             3 * i + 2, n] * evecs[3 * j + 2, n])
#         # tr2 += 1 / l * (evecs[3 * i, n] * evecs[3 * i, n] + evecs[3 * i + 1, n] * evecs[3 * i + 1, n] + evecs[
#         #     3 * i + 2, n] * evecs[3 * i + 2, n])
#         # tr3 += 1 / l * (evecs[3 * j, n] * evecs[3 * j, n] + evecs[3 * j + 1, n] * evecs[3 * j + 1, n] + evecs[
#         #     3 * j + 2, n] * evecs[3 * j + 2, n])
#     cov = tr1  # / np.sqrt(tr2 * tr3)
#     return cov

def con_c(evals, evecs, c, row, col):
    from pythranFuncs import cov
    n_d = int(evecs.shape[0] / 3)
    n_e = evals.shape[0]

    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        c[i, j] = cov(evals, evecs, i, j)
    return c

def con_d(c, d, row, col):
    for k in range(row.shape[0]):
        i, j = (row[k], col[k])
        d[i, j] = c[i,i] + c[j,j] - 2 * c[i, j]
    return d