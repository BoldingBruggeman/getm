import unittest

import numpy as np

import pygetm

TOLERANCE = 1e-14


class TestGrid(unittest.TestCase):
    def test_interpolation(self):
        domain = pygetm.domain.create_cartesian(
            np.linspace(0.0, 10000.0, 101),
            np.linspace(0.0, 10000.0, 100),
            H=100,
            f=0.0,
            logger=pygetm.parallel.get_logger(level="ERROR"),
        )
        T = domain.create_grids(nz=30, halox=2, haloy=2, velocity_grids=2)

        for z in (False, pygetm.CENTERS, pygetm.INTERFACES):
            with self.subTest(z=z):
                # Random initialization of T
                t = T.array(z=z)
                t.all_values[...] = np.random.random(t.all_values.shape)

                # From T to U
                u = t.interp(T.ugrid)
                u_control = 0.5 * (t.all_values[..., :, :-1] + t.all_values[..., :, 1:])
                self.assertTrue((u.all_values[..., :, :-1] == u_control).all())

                # From T to V
                v = t.interp(T.vgrid)
                v_control = 0.5 * (t.all_values[..., :-1, :] + t.all_values[..., 1:, :])
                self.assertTrue((v.all_values[..., :-1, :] == v_control).all())

                # From T to X
                x = t.interp(T.xgrid)
                x_control = 0.25 * (
                    t.all_values[..., :-1, :-1]
                    + t.all_values[..., :-1, 1:]
                    + t.all_values[..., 1:, :-1]
                    + t.all_values[..., 1:, 1:]
                )
                self.assertLess(
                    np.abs(x.all_values[..., 1:-1, 1:-1] - x_control).max(), TOLERANCE
                )

                # From T to UU
                uu = t.interp(T.ugrid.ugrid)
                self.assertTrue(
                    np.abs(uu.all_values[..., :-1] == t.all_values[..., 1:]).all()
                )

                # From T to VV
                vv = t.interp(T.vgrid.vgrid)
                self.assertTrue(
                    np.abs(vv.all_values[..., :-1, :] == t.all_values[..., 1:, :]).all()
                )

                # From T to UV
                uv = t.interp(T.ugrid.vgrid)
                uv_control = x_control
                self.assertLess(
                    np.abs(uv.all_values[..., :-1, :-1] - uv_control).max(), TOLERANCE
                )

                # From T to VU
                vu = t.interp(T.vgrid.ugrid)
                self.assertTrue(
                    (uv.all_values[..., :-1, :-1] == vu.all_values[..., :-1, :-1]).all()
                )

                # Random initialization of X
                x.all_values[...] = np.random.random(x.all_values.shape)

                # From X to T
                t = x.interp(T)
                t_control = 0.25 * (
                    x.all_values[..., :-1, :-1]
                    + x.all_values[..., :-1, 1:]
                    + x.all_values[..., 1:, :-1]
                    + x.all_values[..., 1:, 1:]
                )
                self.assertLess(np.abs(t.all_values - t_control).max(), TOLERANCE)

                # Random initialization of U
                u.all_values[...] = np.random.random(u.all_values.shape)

                # From U to UU
                uu = u.interp(T.ugrid.ugrid)
                uu_control = 0.5 * (
                    u.all_values[..., :, :-1] + u.all_values[..., :, 1:]
                )
                self.assertTrue((uu.all_values[..., :, :-1] == uu_control).all())

                # From U to VU
                vu = u.interp(T.vgrid.ugrid)
                vu_control = 0.5 * (
                    u.all_values[..., :-1, :] + u.all_values[..., 1:, :]
                )
                self.assertTrue((vu.all_values[..., :-1, :] == vu_control).all())

                # From U to T
                t = u.interp(T)
                t_control = 0.5 * (u.all_values[..., :, :-1] + u.all_values[..., :, 1:])
                self.assertTrue((t.all_values[..., :, 1:] == t_control).all())

                # From U to V
                v = u.interp(T.vgrid)
                v_control = 0.25 * (
                    u.all_values[..., :-1, :-1]
                    + u.all_values[..., :-1, 1:]
                    + u.all_values[..., 1:, :-1]
                    + u.all_values[..., 1:, 1:]
                )
                self.assertLess(
                    np.abs(v.all_values[..., :-1, 1:] - v_control).max(), TOLERANCE
                )

                # Random initialization of V
                v.all_values[...] = np.random.random(v.all_values.shape)

                # From V to UV
                uv = v.interp(T.ugrid.vgrid)
                uv_control = 0.5 * (
                    v.all_values[..., :, :-1] + v.all_values[..., :, 1:]
                )
                self.assertTrue((uv.all_values[..., :, :-1] == uv_control).all())

                # From V to VV
                vv = v.interp(T.vgrid.vgrid)
                vv_control = 0.5 * (
                    v.all_values[..., :-1, :] + v.all_values[..., 1:, :]
                )
                self.assertTrue((vv.all_values[..., :-1, :] == vv_control).all())

                # From V to T
                t = v.interp(T)
                t_control = 0.5 * (v.all_values[..., :-1, :] + v.all_values[..., 1:, :])
                self.assertTrue((t.all_values[..., 1:, :] == t_control).all())

                # From V to U
                u = v.interp(T.ugrid)
                u_control = 0.25 * (
                    v.all_values[..., :-1, :-1]
                    + v.all_values[..., :-1, 1:]
                    + v.all_values[..., 1:, :-1]
                    + v.all_values[..., 1:, 1:]
                )
                self.assertLess(
                    np.abs(u.all_values[..., 1:, :-1] - u_control).max(), TOLERANCE
                )


if __name__ == "__main__":
    unittest.main()
