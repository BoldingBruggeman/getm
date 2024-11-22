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

domain = pygetm.domain.create_cartesian(x=x, y=y, nz=nz, f=0.0, H=H)
# domain.initialize(pygetm.BAROCLINIC)
print(domain.T)
# print('AAAAA')
# print(domain._initialized)
# print('BBBBB')

sim = pygetm.Simulation(domain, runtype=pygetm.BAROTROPIC_3D)

ddl = 0.75
ddu = 1.0

ddl = 0.0
ddu = 0.0

# Sigma coordinates
vc_sigma = vertical_coordinates.Sigma(nz, ddl, ddu)
# print(vc_sigma.dga)
# print(vc_sigma(D[np.newaxis, :])[:, 0, :])
# KBh = vc_sigma(D[np.newaxis, :])[:, 0, :]
h = vc_sigma(domain.T.H)[:, :, :]

print(np.shape(h))
z = np.zeros((nz + 1, h.shape[1]))
# z[1:] = h.cumsum(axis=0)

# General Vertical Coordinates
ddl = 1.0
ddu = 1.0
Dgamma = 50.0
gamma_surf = True
vc = vertical_coordinates.GVC(nz, ddl, ddu, Dgamma=Dgamma, gamma_surf=gamma_surf)
print(vc.dbeta.shape)
print(D.shape)
print(vc.dbeta.shape + D.shape)

# h = vc(D[np.newaxis, :int(imax)])[:, 0, :]
# z = np.zeros((nz + 1, h.shape[1]))
# z[1:] = h.cumsum(axis=0)

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

# sim = pygetm.Simulation(domain, runtype=pygetm.BAROCLINIC)

vc.initialize(domain.T, domain.U, domain.V, domain.X)
#vc.initialize(domain.T)

domain.T.hn[...] = 0.
domain.U.hn[...] = 0.
domain.V.hn[...] = 0.
domain.X.hn[...] = 0.
print(domain.T.hn[-1,1,55:65])
print(domain.U.hn[-1,1,55:65])
vc(D)
vc.update(10.)
print(domain.T.hn[-1,1,55:56+1])
print(domain.U.hn[-1,1,55])
print(domain.V.hn[-1,0,55])
print(domain.X.hn[-1,1,55])
