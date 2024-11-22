#!/usr/bin/env python

import argparse
import datetime
import os.path

import numpy as np

import pygetm

parser = argparse.ArgumentParser()
parser.add_argument("setup_dir", help="Path to configuration files", default=".")
parser.add_argument(
    "--start",
    help="Simulation start time [YYYY-mm-dd HH:MM:SS]",
    default="2004-01-01 00:00:00",
)
parser.add_argument(
    "--stop",
    help="Simulation stop time [YYYY-mm-dd HH:MM:SS]",
    default="2004-01-11 00:00:00",
)
parser.add_argument(
    "--nx",
    type=int,
    help="Number of tracer points in x direction",
    default=100,
)
parser.add_argument(
    "--ny", type=int, help="Number of tracer points in y direction", default=30
)
parser.add_argument("--profile", help="File to save profiling report to")
parser.add_argument(
    "--no_output",
    action="store_false",
    dest="output",
    help="Do not save any results to NetCDF",
)
args = parser.parse_args()

simstart = datetime.datetime.strptime(args.start, "%Y-%m-%d %H:%M:%S")
simstop = datetime.datetime.strptime(args.stop, "%Y-%m-%d %H:%M:%S")

ry, rx = np.meshgrid(np.linspace(-0.5, 0.5, args.nx), np.linspace(-0.5, 0.5, args.ny))
H = 100 * np.exp(-0.5 * 10 * (rx**2 + ry**2))

domain = pygetm.domain.create_spherical(
    lon=np.linspace(0.0, 1.25, args.nx + 1),
    lat=np.linspace(45.0, 45.25, args.ny + 1),
    interfaces=True,
    z0=0.01,
    H=H,
    f=pygetm.domain.coriolis(45.0),
)

sim = pygetm.Simulation(
    domain,
    gotm=os.path.join(args.setup_dir, "gotmturb.nml"),
    airsea=pygetm.airsea.Fluxes(taux=0.1),
    internal_pressure=pygetm.internal_pressure.ShchepetkinMcwilliams(),
    vertical_coordinates=pygetm.vertical_coordinates.GVC(
        30, Dgamma=40.0, ddu=1.0, ddl=1.0
    ),
    Dcrit=0.1,
    Dmin=0.02,
)

sim.radiation.set_jerlov_type(pygetm.Jerlov.Type_I)

sim.salt.set(0.0)

# Load temperature profile from text file
dat = np.loadtxt(os.path.join(args.setup_dir, "tprof.dat"), skiprows=1)
z, t = dat[::-1, 0], dat[::-1, 1]
sim.temp.values[...] = np.interp(sim.T.zc.values.ravel(), z, t).reshape(sim.temp.shape)

if args.output:
    sim.logger.info("Setting up output")
    basename = os.path.splitext(os.path.basename(__file__))[0]
    output = sim.output_manager.add_netcdf_file(
        f"{basename}_2d.nc", interval=datetime.timedelta(hours=1)
    )
    output.request("zt", "Dt", "u1", "v1", "tausxu", "tausyv")
    output = sim.output_manager.add_netcdf_file(
        f"{basename}_3d.nc", interval=datetime.timedelta(hours=1)
    )
    output.request("uk", "vk", "ww", "SS", "num")
    output.request("temp", "salt", "rho", "NN", "sst", "hnt", "nuh", "idpdx", "idpdy")

sim.start(simstart, timestep=10.0, split_factor=60, report=360, profile=args.profile)
while sim.time < simstop:
    sim.advance(check_finite=True)
sim.finish()
