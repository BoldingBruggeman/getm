import unittest
import logging

import numpy as np
import netCDF4

import pygetm


class TestDomain(unittest.TestCase):
    def test_halos(self):
        x = np.linspace(0.0, 10000.0, 101)
        y = np.linspace(0.0, 10000.0, 100)
        for halox in (0, 1, 2):
            for haloy in (0, 1, 2):
                with self.subTest(halox=halox, haloy=haloy):
                    domain = pygetm.domain.create_cartesian(
                        x,
                        y,
                        H=100,
                        f=0.0,
                        logger=pygetm.parallel.get_logger(level="ERROR"),
                    )
                    domain.create_grids(
                        nz=30,
                        halox=halox,
                        haloy=haloy,
                        velocity_grids=2,
                    )

    def test_create(self):
        with netCDF4.Dataset("test.nc", "w") as nc:
            nx, ny = 50, 51
            nc.createDimension("x", nx)
            nc.createDimension("y", ny)
            nc.createVariable("lon", "f4", ("x",))
            nc.createVariable("lat", "f4", ("y",))
            nc.createVariable("x", "f4", ("x",))
            nc.createVariable("y", "f4", ("y",))

            nc.createDimension("xi", nx + 1)
            nc.createDimension("yi", ny + 1)
            nc.createVariable("loni", "f4", ("xi",))
            nc.createVariable("lati", "f4", ("yi",))
            nc.createVariable("xi", "f4", ("xi",))
            nc.createVariable("yi", "f4", ("yi",))

            nc.createVariable("H", "f4", ("y", "x"))
        logger = logging.getLogger()
        logger.setLevel("ERROR")
        with netCDF4.Dataset("test.nc") as nc:
            pygetm.domain.create_spherical(
                nc["lon"], nc["lat"], H=nc["H"], logger=logger
            )
            pygetm.domain.create_spherical(
                nc["lon"], nc["lat"], x=nc["x"], y=nc["y"], H=nc["H"], logger=logger
            )
            pygetm.domain.create_cartesian(
                nc["x"], nc["y"], H=nc["H"], f=0.0, logger=logger
            )
            pygetm.domain.create_cartesian(
                nc["x"], nc["y"], lon=nc["lon"], lat=nc["lat"], H=nc["H"], logger=logger
            )

            pygetm.domain.create_spherical(
                nc["loni"], nc["lati"], H=nc["H"], interfaces=True, logger=logger
            )
            pygetm.domain.create_spherical(
                nc["loni"],
                nc["lati"],
                x=nc["x"],
                y=nc["y"],
                H=nc["H"],
                interfaces=True,
                logger=logger,
            )
            pygetm.domain.create_spherical(
                nc["lon"],
                nc["lat"],
                x=nc["xi"],
                y=nc["yi"][:][:, np.newaxis],
                H=nc["H"],
                logger=logger,
            )
            pygetm.domain.create_cartesian(
                nc["xi"], nc["yi"], H=nc["H"], f=0.0, interfaces=True, logger=logger
            )
            pygetm.domain.create_cartesian(
                nc["xi"],
                nc["yi"],
                lon=nc["lon"],
                lat=nc["lat"],
                H=nc["H"],
                interfaces=True,
                logger=logger,
            )
            pygetm.domain.create_cartesian(
                nc["x"],
                nc["y"],
                lon=nc["loni"],
                lat=nc["lati"][:][:, np.newaxis],
                H=nc["H"],
                logger=logger,
            )


if __name__ == "__main__":
    unittest.main()
