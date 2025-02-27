import sys
import argparse
import unittest

import numpy as np

import pygetm
import pygetm.domain
import pygetm.debug


class Test3DBarotropic(unittest.TestCase):
    def test(self):
        for apply_bottom_friction in (True, False):
            for tau_x in (-0.1, 0.0, 0.1):
                for tau_y in (-0.1, 0.0, 0.1):
                    with self.subTest(
                        tau_x=tau_x,
                        tau_y=tau_y,
                        apply_bottom_friction=apply_bottom_friction,
                    ):
                        self._test(
                            tau_x, tau_y, apply_bottom_friction=apply_bottom_friction
                        )

    def _test(
        self,
        tau_x: float = 0.0,
        tau_y: float = 0.0,
        timestep: float = 10.0,
        ntime: int = 360,
        apply_bottom_friction: bool = False,
    ):
        domain = pygetm.domain.create_cartesian(
            500.0 * np.arange(100),
            500.0 * np.arange(30),
            f=0,
            H=50,
            z0=0.01 if apply_bottom_friction else 0.0,
            logger=pygetm.parallel.get_logger(level="ERROR"),
        )
        sim = pygetm.Simulation(
            domain,
            runtype=pygetm.RunType.BAROTROPIC_3D,
            vertical_coordinates=pygetm.vertical_coordinates.Sigma(50),
        )

        # Idealized surface forcing
        tausx = sim.U.array(fill=tau_x)
        tausy = sim.V.array(fill=tau_y)
        sp = sim.T.array(fill=0.0)

        idpdx = sim.U.array(fill=0.0, z=pygetm.CENTERS)
        idpdy = sim.V.array(fill=0.0, z=pygetm.CENTERS)
        viscosity = sim.T.array(fill=0.0, z=pygetm.INTERFACES)

        t = sim.T.array(name="tracer", z=pygetm.CENTERS)
        rng = np.random.default_rng()
        rng.random(t.all_values.shape, out=t.all_values)
        adv = pygetm.operators.Advection(t.grid, scheme=1)

        times = timestep * np.arange(ntime)
        mode_split = 10
        z_sum_ini = sim.T.z.ma.sum()
        pre_tot = (t * sim.T.hn).values.sum()
        for istep, time in enumerate(times):
            sim.update_surface_pressure_gradient(sim.T.z, sp)
            sim.momentum.advance_depth_integrated(
                timestep, tausx, tausy, sim.dpdx, sim.dpdy
            )
            sim.advance_surface_elevation(
                timestep, sim.momentum.U, sim.momentum.V, sim.fwf
            )
            sim.update_depth()

            if istep % mode_split == 0:
                sim.update_depth(True)
                sim.update_surface_pressure_gradient(sim.T.zio, sp)

                sim.momentum.advance(
                    timestep * mode_split,
                    mode_split,
                    tausx,
                    tausy,
                    sim.dpdx,
                    sim.dpdy,
                    idpdx,
                    idpdy,
                    viscosity,
                )

                div = np.zeros(sim.T.hn.shape)
                U1 = (sim.momentum.pk * sim.U.dy).all_values[:, 2:-2, 1:-3]
                U2 = (sim.momentum.pk * sim.U.dy).all_values[:, 2:-2, 2:-2]
                V1 = (sim.momentum.qk * sim.V.dx).all_values[:, 1:-3, 2:-2]
                V2 = (sim.momentum.qk * sim.V.dx).all_values[:, 2:-2, 2:-2]
                W1 = (sim.momentum.ww * sim.T.area).all_values[:-1, 2:-2, 2:-2]
                W2 = (sim.momentum.ww * sim.T.area).all_values[1:, 2:-2, 2:-2]
                dH = ((sim.T.hn - sim.T.ho) * sim.T.area).all_values[:, 2:-2, 2:-2] / (
                    timestep * mode_split
                )
                div = U1 - U2 + V1 - V2 + W1 - W2 - dH
                maxtp = np.zeros_like(div)
                for ar in [U1, U2, V1, V2, W1, W2, dH]:
                    maxtp = np.maximum(maxtp, np.abs(ar))
                reldiv = div / np.where(maxtp > 0.0, maxtp, 1.0)
                self.assertTrue(
                    pygetm.debug.check_zero(
                        "maximum divergence (as missing vertical velocity in m s-1)",
                        div / sim.T.area.values,
                    )
                )
                adv.apply_3d(
                    sim.momentum.uk,
                    sim.momentum.vk,
                    sim.momentum.ww,
                    timestep * mode_split,
                    t,
                )
                new_tot = (t * sim.T.hn).values.sum()
                self.assertTrue(
                    pygetm.debug.check_equal(
                        "layer thicknesses",
                        adv.h[:, 2:-2, 2:-2],
                        sim.T.hn.values,
                        rtol=1e-14,
                        atol=1e-14,
                    )
                )
        self.assertTrue(
            pygetm.debug.check_equal(
                "tracer total before and after simulation", new_tot, pre_tot
            )
        )
        self.assertTrue(
            pygetm.debug.check_equal(
                "total volume before and after simulation",
                z_sum_ini,
                sim.T.z.ma.sum(),
                atol=1e-14,
                rtol=1e-14,
            )
        )


if __name__ == "__main__":
    unittest.main()
