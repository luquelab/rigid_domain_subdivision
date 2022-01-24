#! /home/ctbrown/anaconda3/bin/python3
#PBS -l nodes=2:ppn=24
#PBS -l walltime=72:00:00

import sys
sys.path.append('/home/ctbrown/mechanical_subdivisions/mechanical_subdivision_ProDy/src')

from make_model import make_model
from subdivide_model import subdivide_model
from input import *

print(pdb)
if rebuild_model:
    model, calphas = make_model(pdb, n_modes)

calphas, domains = subdivide_model(pdb, cluster_start, cluster_stop, cluster_step)
