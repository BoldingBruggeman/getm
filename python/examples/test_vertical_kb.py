import numpy as np
from pygetm import vertical_coordinates

# from pygetm import operators
import pygetm

import pygetm.operators

D = np.linspace(5.0, 505.0, 50 + 1)
imax = 50
nz = 10

H = 5.0 + 5 * np.arange(100)

x = np.linspace(0.0, 100000, 100)
y = np.linspace(0.0, 200, 2)

domain = pygetm.domain.create_cartesian(x=x, y=y, f=0.0, H=H)

# Adaptive Coordinates
ddl = 1.0
ddu = 1.0
Dgamma = 50.0
gamma_surf = True
vc = vertical_coordinates.Adaptive(
    nz,
    ddl,
    ddu,
    Dgamma=Dgamma,
    gamma_surf=gamma_surf,
    csigma=0.01,
    hpow=3,
    cgvc=-1.0,  # 0.01
    chsurf=0.5,
    hsurf=0.5,
    chmidd=0.2,
    hmidd=-4.0,
    chbott=0.3,
    hbott=-0.25,
    cneigh=0.1,
    rneigh=0.25,
    decay=2.0 / 3.0,
    cNN=-1.0,
    drho=0.3,
    cSS=-1.0,
    dvel=0.1,
    chmin=-0.1,
    hmin=0.3,
    nvfilter=1,
    vfilter=0.2,
    nhfilter=1,
    hfilter=0.1,
    tgrid=14400.0,
    split=1,
)
sim = pygetm.Simulation(domain, vertical_coordinates=vc, runtype=pygetm.BAROTROPIC_3D)

# need an initial layer thickness for vertical coordinates
if True:
    # Sigma coordinates
    ddl = 0.0
    ddu = 0.0
    vc_sigma = vertical_coordinates.Sigma(nz, ddl=ddl, ddu=ddu)
    sim.T.hn[...] = vc_sigma(sim.T.D)
else:
    # General Vertical Coordinates
    ddl = 1.0
    ddu = 1.0
    Dgamma = 10.0
    gamma_surf = True
    vc = vertical_coordinates.GVC(nz, ddl, ddu, Dgamma=Dgamma, gamma_surf=gamma_surf)

sim.U.hn[...] = 0.
sim.V.hn[...] = 0.
sim.X.hn[...] = 0.
sim.vertical_coordinates(D)
sim.vertical_coordinates.update(10.)
print(sim.T.hn[-1,1,55:56+1])
print(sim.U.hn[-1,1,55])
print(sim.V.hn[-1,0,55])
print(sim.X.hn[-1,1,55])
