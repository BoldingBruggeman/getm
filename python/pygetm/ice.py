import logging

import numpy as np

import pygetm.domain
import pygetm.core
import pygetm.airsea
from pygetm.constants import FILL_VALUE


class Ice:
    def initialize(self, grid: pygetm.core.Grid, logger: logging.Logger):
        self.logger = logger
        self.grid = grid
        self.ice = grid.array(
            name="ice",
            long_name="ice cover",
            units="1",
            fill_value=FILL_VALUE,
            attrs=dict(standard_name="sea_ice_area_fraction"),
            fabm_standard_name="ice_area_fraction",
        )
        self.ice.fill(0.0)
        self.has_ice = False
        self.covered = np.full(grid.H.all_values.shape, False)

    def __call__(
        self,
        macro: bool,
        ct_sf: pygetm.core.Array,
        sa_sf: pygetm.core.Array,
        airsea: pygetm.airsea.Fluxes,
    ):
        if macro:
            # 1st order freezing point approximation based on
            # gsw_mod_freezing_poly_coefficients
            ct_freezing = 0.017947064327968736 - 0.06076099099929818 * sa_sf.all_values
            unmasked = self.grid._water
            np.logical_and(ct_sf.all_values <= ct_freezing, unmasked, out=self.covered)
            self.has_ice = self.covered.any()
            self.ice.all_values[unmasked] = np.where(self.covered[unmasked], 1.0, 0.0)
            if self.has_ice:
                np.putmask(ct_sf.all_values, self.covered, ct_freezing)
                airsea.shf.all_values[self.covered & (airsea.shf.all_values < 0)] = 0.0

        if self.has_ice:
            airsea.taux.all_values[self.covered] = 0.0
            airsea.tauy.all_values[self.covered] = 0.0
