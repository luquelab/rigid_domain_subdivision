import sys
import scipy
from scipy import sparse
from prody import LOGGER, SETTINGS
from make_model import make_model
from subdivide_model import subdivide_model



pdb = sys.argv[1]
type = sys.argv[2]
n_modes = int(sys.argv[3])
cluster_start = int(sys.argv[4])
cluster_stop = int(sys.argv[5])
cluster_step = int(sys.argv[6])


gnm, calphas = make_model(pdb, n_modes, type)

calphas, domains = subdivide_model(pdb, cluster_start, cluster_stop, cluster_step)
