#!/usr/bin/env python

"""AMBER simulations"""

__author__ = "Zack Scholl"
__copyright__ = "Copyright 2016, Duke University"
__credits__ = ["Zack Scholl"]
__license__ = "None"
__version__ = "0.1"
__maintainer__ = "Zack Scholl"
__email__ = "zns@duke.edu"
__status__ = "Production"

import time
import json
import os
import sys
import shutil
import subprocess
import shlex
import logging
import glob


def initiate(params):
    testing = False

    # DON'T EDIT THIS STUFF UNLESS NEED BE
    timefactor = 1
    if testing:
        timefactor = 10  # 1 = normal, 100 = testing

    files = {}
    files['01_Min.in'] = """Minimize
 &cntrl
  imin=1,
  ntx=1,
  irest=0,
  maxcyc=%(maxcyc)d,
  ncyc=%(ncyc)d,
  ntpr=%(ntpr)d,
  ntwx=0,
  cut=8.0,
 /
    """ % {"maxcyc": 2000 / timefactor, "ncyc": 1000 / timefactor, "ntpr": 100 / timefactor}

    files['02_Heat.in'] = """Heat
 &cntrl
  imin=0,
  ntx=1,
  irest=0,
  nstlim=%(nstlim)d,
  dt=0.00%(timestep)s,
  ntf=2,
  ntc=2,
  tempi=0.0,
  temp0=%(temp)s,
  ntpr=100,
  ntwx=100,
  cut=8.0,
  ntb=1,
  ntp=0,
  ntt=3,
  gamma_ln=2.0,
  nmropt=1,
  ig=-1,
 /
&wt type='TEMP0', istep1=0, istep2=%(istep2)d, value1=0.0, value2=%(temp)s /
&wt type='TEMP0', istep1=%(istep1)d, istep2=%(nstlim)d, value1=%(temp)s, value2=%(temp)s /
&wt type='END' /
    """ % {'timestep': str(int(params['timestep'] * 10)), 'temp': str(params['temp']), "nstlim": 10000 / timefactor, "istep2": 9000 / timefactor, "istep1": 1 + 9000 / timefactor}

    files['025_PreProd.in'] = """Typical Production MD NPT, MC Bar 4fs HMR
 &cntrl
   ntx=5, irest=1,
   ntc=2, ntf=2,
   nstlim=10000,
   ntpr=200, ntwx=500,
   ntwr=200,
   dt=0.00%(timestep)s, cut=9.5,
   ntt=1, tautp=10.0,
   temp0=%(temp)d,
   ntb=2, ntp=1, barostat=2,
   ioutfm=1,
 /
    """ % {'timestep': str(int(params['timestep'] * 10)), 'temp': params['temp']}

    files['03_Prod.in'] = """Typical Production MD NPT, MC Bar 4fs HMR
 &cntrl
   ntx=5, irest=1,
   ntc=2, ntf=2,
   nstlim=%(time)d,
   ntpr=1000, ntwx=%(ntwx)d,
   ntwr=10000,
   dt=0.00%(timestep)s, cut=9.5,
   ntt=1, tautp=10.0,
   temp0=%(temp)d,
   ntb=2, ntp=1, barostat=2,
   ioutfm=1,
 /
    """ % {'timestep': str(int(params['timestep'] * 10)), 'temp': params['temp'], 'time': params['nanoseconds'] * 400000 / timefactor, 'ntwx': int(params['nanoseconds'] * 400000 / params['numFrames'] / timefactor)}

    files['03_Prod_collapse.in'] = """Typical Production MD NPT, MC Bar 4fs HMR
 &cntrl
   ntx=5, irest=1,
   ntc=2, ntf=2,
   nstlim=%(time)d,
   ntpr=1000, ntwx=%(ntwx)d,
   ntwr=10000,
   dt=0.00%(timestep)s, cut=9.5,
   ntt=1, tautp=10.0,
   temp0=%(temp)d,
   ntb=2, ntp=1, barostat=2,
   ioutfm=1,
 /
    """ % {'timestep': str(int(params['timestep'] * 10)), 'temp': params['temp'], 'time': 2 * 400000 / timefactor, 'ntwx': int(2 * 400000 / 10 / timefactor)}

    tleapConf = """source leaprc.ff14SB
loadAmberParams frcmod.ionsjc_tip3p
mol = loadpdb %s
solvatebox mol TIP3PBOX 15.0
addions mol Na+ 0
addions mol Cl- 0
saveamberparm mol prmtop inpcrd
quit""" % params['pdb']
    with open("tleap.conf", "w") as f:
        f.write(tleapConf)
    cmd = "tleap -f tleap.conf"
    proc = subprocess.Popen(shlex.split(cmd), shell=False)
    proc.wait()
    os.system("$AMBERHOME/bin/ambpdb -p prmtop < inpcrd > startingConfiguration.pdb")

    # Write configuration files
    for f in files:
        with open(f, 'w') as f2:
            f2.write(files[f])

    # Write parameters
    with open('params.json', 'w') as f:
        f.write(json.dumps(params, indent=2))


if __name__ == "__main__":
    params = {}
    params['temp'] = 360.0
    params['pdb'] = raw_input('Enter pdb file (make sure to remove hydrogens first): ')
    params['nanoseconds'] = 10  # nanoseconds per translation setp
    params['nanoseconds'] = float(
        raw_input('Enter simulation time (in nanoseconds): '))
    params['numFrames'] = int(
        raw_input('How many total frames do you want to generate (e.g. 400): '))
    params['temp'] = float(
        raw_input('Enter the simulation temperature (in Kelvin): '))
    params['removeOxt'] = False
    params['boxSize'] = float(
        raw_input('Enter the box padding size (in angstroms, e.g. 10): '))
    params['cudaDevice'] = raw_input(
        'Enter which CUDA device (e.g. 0 or 1): ').strip()
    if params['cudaDevice'] != '1':
        params['cudaDevice'] = '0'
    params['reweighting'] = 'y' in raw_input(
        'Would you like to reweight the hydrogens? (y/n): ').strip().lower()
    params['timestep'] = float(2.5)
    if params['reweighting']:
        timestep = raw_input(
            'What timestep would you like (femtoseconds)? (e.g. 5): ')
        try:
            params['timestep'] = float(timestep)
        except:
            print("Error parsing timestep, defaulting to %s" %
                  params['timestep'])
    initiate(params)
    print("-" * 60)
    print("-" * 60)
    print("\n\n\nSetup complete!\n\nNow just run:\n\n      nohup python simulate.py &\n\n\nand then use\n")
    print("      tail -f log    \n\nto watch the log.")