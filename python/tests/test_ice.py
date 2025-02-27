import unittest

import numpy as np

import pygetm


class TestIce(unittest.TestCase):
    def test_coverage(self):
        domain = pygetm.domain.create_cartesian(
            np.linspace(0, 100e3, 50),
            np.linspace(0, 100e3, 51),
            H=50.0,
            interfaces=True,
            f=0.0,
            logger=pygetm.parallel.get_logger(level="ERROR"),
        )
        T = domain.create_grids(nz=30, halox=2, haloy=2)

        airsea = pygetm.airsea.FluxesFromMeteo()
        airsea.initialize(T, domain.root_logger.getChild("airsea"))
        taux = np.random.uniform(0.001, 0.1, airsea.taux.shape)
        tauy = np.random.uniform(0.001, 0.1, airsea.tauy.shape)
        airsea.taux.values[:, :] = taux
        airsea.tauy.values[:, :] = tauy

        ice = pygetm.ice.Ice()
        ice.initialize(T, domain.root_logger.getChild("ice"))
        ct_sf = T.array(fill=10.0)
        sa_sf = T.array(fill=35.0)
        cover = np.random.random_sample(ct_sf.shape) > 0.5
        ct_sf.values[cover] = -10.0
        ice(True, ct_sf, sa_sf, airsea)
        self.assertTrue(ice.has_ice)
        self.assertTrue((ice.ice.values[cover] == 1.0).all())
        self.assertTrue((ice.ice.values[~cover] == 0.0).all())
        self.assertTrue((ct_sf.values[cover] > -10.0).all())
        self.assertTrue((ct_sf.values[~cover] == 10.0).all())
        self.assertTrue((airsea.taux.values[cover] == 0.0).all())
        self.assertTrue((airsea.tauy.values[cover] == 0.0).all())
        self.assertTrue((airsea.taux.values[~cover] == taux[~cover]).all())
        self.assertTrue((airsea.tauy.values[~cover] == tauy[~cover]).all())


if __name__ == "__main__":
    unittest.main()
