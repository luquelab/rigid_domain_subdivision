import numba as nb
import numpy as np
from settings import *

def buildENM(atoms):
    from scipy import sparse
    import numpy as np
    from sklearn.neighbors import BallTree, radius_neighbors_graph
    from settings import cbeta

    sel = 'protein and name CA'
    if cbeta:
        sel = sel + ' CB'
    calphas = atoms.select(sel)
    coords = calphas.getCoords()
    n_atoms = coords.shape[0]
    print('# Atoms ',n_atoms)
    dof = n_atoms * 3


    tree = BallTree(coords)
    kirch = radius_neighbors_graph(tree, cutoff, mode='distance', n_jobs=-1)
    kc = kirch.tocoo().copy()
    kc.sum_duplicates()
    kirch = kirchGamma(kc, calphas, d2=d2, flexibilities=flexibilities, cbeta=cbeta, struct=False).tocsr()
    dg = np.array(kirch.sum(axis=0))
    kirch.setdiag(-dg[0])
    kirch.sum_duplicates()
    print(kirch.data)

    if model=='anm':
        kc = kirch.tocoo().copy()
        kc.sum_duplicates()
        hData = hessCalc(kc.row, kc.col, kirch.data, coords)
        indpt = kirch.indptr
        inds = kirch.indices
        hessian = sparse.bsr_matrix((hData, inds, indpt), shape=(dof,dof)).tocsr()
        hessian = fanm*hessian + (1-fanm)*sparse.kron(kirch, np.identity(3))
    else:
        hessian = kirch.copy()
    print('done constructing matrix')
    return kirch, hessian



def kirchGamma(kirch, atoms, **kwargs):
    kg = kirch.copy()

    if 'struct' in kwargs and kwargs['struct']:
        chains = atoms.getData('chainNum')
        print(chains.shape)
        #sstr, ssid, chainNum = tup
        print('# secstr residues:', np.count_nonzero(atoms.getSecstrs() != 'C'))
        tup = (atoms.getSecstrs(), atoms.getSecids(), chains)
        abg = cooOperation(kirch.row, kirch.col, kirch.data, secStrGamma, tup)
        kg.data = abg
        # kg = addSulfideBonds(sulfs, kg)
    elif 'd2' in kwargs and kwargs['d2']:
        print('d2 mode')
        kg.data = -1/(kg.data**2)
    else:
        kg.data = -np.ones_like(kg.data)

    if 'flexibilities' in kwargs and kwargs['flexibilities']:
        flex = flexes(1/atoms.getBetas())
        fl = cooOperation(kirch.row, kirch.col, kirch.data, flexFunc, flex)
        kg.data = kg.data*fl

    if 'cbeta' in kwargs and kwargs['cbeta']:
        names = atoms.getNames()
        abgamma = np.array([0 if x == 'CA' else np.sqrt(0.5) for x in names])
        abg = cooOperation(kirch.row, kirch.col, kirch.data, abFunc, abgamma)
        kg.data = kg.data * abg
    return kg


@nb.njit()
def cooOperation(row, col, data, func, arg):
    r = np.copy(data)
    for n, (i, j, v) in enumerate(zip(row, col, data)):
        if i==j:
            continue
        r[n] = func(i,j,v, arg)
    return r

@nb.njit()
def abFunc(i, j, d, abg):
    return -abg[i]*abg[j] + 1.0

def flexes(bfactors):
    return bfactors/np.mean(bfactors)

@nb.njit()
def d2Func(d2):
    return 1/d2

@nb.njit()
def flexFunc(i,j,d, fl):
    return np.sqrt(fl[i]*fl[j])

@nb.njit()
def structFunc(i,j,d, chainNum):
    sij = np.abs(i-j)
    if sij <= 3 and chainNum[i]==chainNum[j]:
        return -100/(sij)**2
    else:
        return -1/(d)**2

@nb.njit()
def secStrGamma(i, j, d, tup):
    """Returns force constant."""
    sstr, ssid, chainNum = tup
    if d <= 4 and chainNum[i]==chainNum[j]:
        return -10
    # if residues are in the same secondary structure element
    if ssid[i] == ssid[j] and chainNum[i]==chainNum[j]:
        i_j = abs(i - j)
        if ((i_j <= 4 and sstr[i] == 'H') or
            (i_j <= 3 and sstr[i] == 'G') or
            (i_j <= 5 and sstr[i] == 'I')) and d <= 7:
            return -6.
    elif sstr[i] == sstr[j] == 'E' and d <= 6:
        return -6.

    return -1/d**2


@nb.njit()
def hessCalc(row, col, kGamma, coords):
    hData = np.zeros((row.shape[0], 3, 3))
    dinds = np.nonzero(row==col)[0]
    for k, (i,j) in enumerate(zip(row,col)):
        if i==j:
            continue
        dvec = coords[j] - coords[i]
        d2 = np.dot(dvec, dvec)
        g = kGamma[k]
        hblock = np.outer(dvec, dvec) * (g / d2) # + (1-fanm)*np.identity(3)*g
        hData[k,:,:] = hblock
        hData[dinds[i]] += -hblock/2
        hData[dinds[j]] += -hblock/2

    return hData





def addSulfideBonds(sulfs, kirch):
    sCoords = sulfs.getCoords()
    from sklearn.neighbors import BallTree, radius_neighbors_graph
    tree = BallTree(sCoords)
    adjacency = radius_neighbors_graph(tree, 3.0, n_jobs=-1)
    print(adjacency.nnz)
    sNodes = sulfs.getData('nodeid')
    kirch = kirch.tocsr()
    for i, n in enumerate(sNodes):
        neighbors = np.nonzero(adjacency[i,:]==1)
        print(neighbors)
        kirch[i, neighbors] = kirch[i, neighbors]*100

    return kirch.tocoo()


def addIonBonds(atoms, kirch, dists):
    anion = atoms.select('resname ASP GLU')
    cation = atoms.select('resname ASP GLU')

    for i, at in enumerate(atoms.iterAtoms()):
        if at.getData('resname') == 'ASP or GLU':
            neighbors = dists[i, :] <= 4
            print(np.count_nonzero(neighbors))
            kirch[i, neighbors] = kirch[i, neighbors] * 100

    return kirch



