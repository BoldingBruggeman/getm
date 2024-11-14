import unittest
import logging
from typing import Tuple

import numpy as np
import netCDF4

import pygetm
from pygetm import FILL_VALUE


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

    def _create_netcdf(self, file: str, nx: int, ny: int):
        with netCDF4.Dataset(file, "w") as nc:
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

    def _check(self, arr, shape: Tuple[int], dtype=float):
        self.assertEqual(type(arr), np.ndarray)
        self.assertEqual(arr.shape, shape)
        self.assertEqual(arr.dtype, np.dtype(dtype))

    def test_supergrid_mapping_nc(self):
        nx, ny = 50, 51
        self._create_netcdf("test.nc", nx, ny)
        with netCDF4.Dataset("test.nc") as nc:
            self._check(
                pygetm.domain.centers_to_supergrid_1d(nc["lon"]), (1 + nx * 2,), "f4"
            )
            self._check(
                pygetm.domain.centers_to_supergrid_1d(nc["lon"], dtype=float),
                (1 + nx * 2,),
            )
            self._check(
                pygetm.domain.centers_to_supergrid_2d(nc["H"]),
                (1 + ny * 2, 1 + nx * 2),
                "f4",
            )
            self._check(
                pygetm.domain.centers_to_supergrid_2d(nc["H"], dtype=float),
                (1 + ny * 2, 1 + nx * 2),
            )
            self._check(
                pygetm.domain.interfaces_to_supergrid_1d(nc["loni"]),
                (1 + nx * 2,),
                "f4",
            )
            self._check(
                pygetm.domain.interfaces_to_supergrid_1d(nc["loni"], dtype=float),
                (1 + nx * 2,),
            )
            self._check(
                pygetm.domain.interfaces_to_supergrid_2d(nc["H"]),
                (ny * 2 - 1, nx * 2 - 1),
                "f4",
            )
            self._check(
                pygetm.domain.interfaces_to_supergrid_2d(nc["H"], dtype=float),
                (ny * 2 - 1, nx * 2 - 1),
            )

    def test_supergrid_mapping_list(self):
        x = [1, 2, 3]
        nx = len(x)
        self._check(
            pygetm.domain.centers_to_supergrid_1d(x, missing_value=0),
            (1 + nx * 2,),
            int,
        )
        self._check(
            pygetm.domain.centers_to_supergrid_1d(x, dtype=float),
            (1 + nx * 2,),
        )
        self._check(pygetm.domain.interfaces_to_supergrid_1d(x), (nx * 2 - 1,), int)
        self._check(
            pygetm.domain.interfaces_to_supergrid_1d(x, dtype=float),
            (nx * 2 - 1,),
        )

    def test_create(self):
        nx, ny = 50, 51
        self._create_netcdf("test.nc", nx, ny)

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

    def test_grid_mapping(self):
        Lx, Ly = 100e3, 100e3
        nx, ny = 100, 52
        HALO = 2
        x = np.linspace(-Lx / 2, Lx / 2, nx)
        y = np.linspace(-Ly / 2, Ly / 2, ny)
        logger = logging.getLogger()
        logger.setLevel("ERROR")
        domain = pygetm.domain.create_cartesian(x, y, H=10.0, lat=0.0, logger=logger)

        self.assertTrue((domain.H == 10.0).all())
        self.assertTrue((domain.mask == 1).all())
        domain.infer_UVX_masks2()
        self.assertTrue((domain.mask[1:-1, 1:-1] == 1).all())
        self.assertTrue((domain.mask[0, :] == 0).all())
        self.assertTrue((domain.mask[-1, :] == 0).all())
        self.assertTrue((domain.mask[:, 0] == 0).all())
        self.assertTrue((domain.mask[:, -1] == 0).all())

        T = domain.create_grids(1, halox=HALO, haloy=HALO, velocity_grids=2)
        self.assertTrue((T.mask.values == 1).all())
        self.assertTrue((T.ugrid.mask.values[:, :-1] == 1).all())
        self.assertTrue((T.ugrid.mask.values[:, -1] == 0).all())
        self.assertTrue((T.vgrid.mask.values[:-1, :] == 1).all())
        self.assertTrue((T.vgrid.mask.values[-1, :] == 0).all())
        self.assertTrue((T.xgrid.mask.values[1:-1, 1:-1] == 1).all())
        self.assertTrue((T.xgrid.mask.values[:, 0] == 0).all())
        self.assertTrue((T.xgrid.mask.values[:, -1] == 0).all())
        self.assertTrue((T.xgrid.mask.values[0, :] == 0).all())
        self.assertTrue((T.xgrid.mask.values[-1, :] == 0).all())

        self.assertTrue((T.mask.all_values[:, :HALO] == 0).all())
        self.assertTrue((T.mask.all_values[:, HALO + nx :] == 0).all())
        self.assertTrue((T.mask.all_values[:HALO, :] == 0).all())
        self.assertTrue((T.mask.all_values[HALO + nx :, :] == 0).all())
        self.assertTrue((T.ugrid.mask.all_values[:, :HALO] == 0).all())
        self.assertTrue((T.ugrid.mask.all_values[:, HALO + nx :] == 0).all())
        self.assertTrue((T.ugrid.mask.all_values[:HALO, :] == 0).all())
        self.assertTrue((T.ugrid.mask.all_values[HALO + nx :, :] == 0).all())
        self.assertTrue((T.vgrid.mask.all_values[:HALO, :] == 0).all())
        self.assertTrue((T.vgrid.mask.all_values[HALO + nx :, :] == 0).all())
        self.assertTrue((T.vgrid.mask.all_values[:, :HALO] == 0).all())
        self.assertTrue((T.vgrid.mask.all_values[:, HALO + nx :] == 0).all())
        self.assertTrue((T.xgrid.mask.all_values[:HALO, :] == 0).all())
        self.assertTrue((T.xgrid.mask.all_values[HALO + nx + 1 :, :] == 0).all())
        self.assertTrue((T.xgrid.mask.all_values[:, :HALO] == 0).all())
        self.assertTrue((T.xgrid.mask.all_values[:, HALO + nx + 1 :] == 0).all())

        for grid in (T, T.ugrid, T.vgrid, T.xgrid):
            self.assertTrue((grid._water == ~grid._land).all())

            # NB the land values of H and z0b_min are set to FILL_VALUE by Grid.freeze
            self.assertTrue((grid.H.all_values[grid._water] == 10).all())
            self.assertTrue((grid.H.all_values[grid._land] == FILL_VALUE).all())
            self.assertTrue((grid.H.all_values[grid._land] == FILL_VALUE).all())

            self.assertTrue((grid.z0b_min.all_values[grid._water] == 0).all())
            self.assertTrue((grid.z0b_min.all_values[grid._land] == FILL_VALUE).all())
            self.assertTrue((grid.z0b_min.all_values[grid._land] == FILL_VALUE).all())

        domain = pygetm.domain.create_cartesian(
            x, y, H=10.0, lat=0.0, logger=logger, periodic_x=True, periodic_y=True
        )
        self.assertTrue((domain.H == 10.0).all())
        self.assertTrue((domain.mask == 1).all())
        domain.infer_UVX_masks2()
        self.assertTrue((domain.mask == 1).all())

        T = domain.create_grids(1, halox=HALO, haloy=HALO, velocity_grids=2)
        self.assertTrue((T.mask.values == 1).all())
        self.assertTrue((T.ugrid.mask.values == 1).all())
        self.assertTrue((T.vgrid.mask.values == 1).all())
        self.assertTrue((T.xgrid.mask.values == 1).all())
        self.assertTrue((T.ugrid.ugrid.mask.values == 1).all())
        self.assertTrue((T.ugrid.vgrid.mask.values == 1).all())
        self.assertTrue((T.vgrid.ugrid.mask.values == 1).all())
        self.assertTrue((T.vgrid.vgrid.mask.values == 1).all())


if __name__ == "__main__":
    unittest.main()
