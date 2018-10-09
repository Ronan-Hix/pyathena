import matplotlib as mpl
mpl.use('agg')

import glob
import sys,os,shutil
import pandas as pd

sys.path.insert(0,'../')
from pyathena.plot_tools import plot_slices,plot_projection,set_aux
from pyathena.utils import compare_files
import cPickle as p

narg=len(sys.argv)
system='tigress'
base='/tigress/changgoo/'
if narg > 1:
    system = sys.argv[1]
    if system == 'pleiades':
        base='/u/ckim14/'
    elif system =='tigress':
        base='/tigress/changgoo/'
    else:
        print '{} is not supported'.format(system)
        sys.exit()
if narg > 2:
    dirs=glob.glob('{}/{}'.format(base,sys.argv[2]))
else:
    dirs=glob.glob('{}/*'.format(base))
ids=[]
for dd in dirs:
    if os.path.isdir(dd):
        if os.path.isdir(dd+'/slice/'):
            ids.append(os.path.basename(dd))

for pid in ids:
    print pid
    slc_files=glob.glob('{}{}/slice/{}.????.slice.p'.format(base,pid,pid))
    slc_files.sort()
    nf=len(slc_files)
    aux=set_aux.set_aux(pid)
    aux_surf=aux['surface_density']
    field_list=['star_particles','nH','temperature','pok',
           'velocity_z']
    slcdata=p.load(open(slc_files[0]))
    if 'magnetic_field_strength' in slcdata['x']:
        field_list += ['magnetic_field_strength']
    for slcname in slc_files:
        starname=slcname.replace('slice.p','starpar.vtk').replace('slice','starpar')
        projname=slcname.replace('slice','surf')
        if not compare_files(slcname,slcname+'ng'):
            plot_slices.slice2(slcname,starname,field_list,aux=aux)
        if not compare_files(projname,projname+'ng'):
            plot_projection.plot_projection(projname,starname,runaway=False,aux=aux_surf)