import logging
import numpy as np

from . import domain
from .constants import CENTERS


class Base:
    """Base class. It is responsible for updating all layer thicknesses
    hn for all grids provided to initialize"""

    logger: logging.Logger

    def __init__(self, nz: int):
        if nz <= 0:
            raise Exception("Number of layers nz must be a positive number")
        self.nz = nz

    def initialize(self, ref_grid: domain.Grid, *other_grids: domain.Grid):
        self.logger = ref_grid.domain.root_logger.getChild("vertical_coordinates")

    def update(self):
        """Update layer thicknesses hn all grids"""
        raise NotImplementedError


class PerGrid(Base):
    """Base class for vertical coordinate types that apply the same operation
    to every grid."""

    def initialize(self, *grids: domain.Grid):
        super().initialize(*grids)
        self.grid_info = [self.prepare_update_args(grid) for grid in grids]

    def prepare_update_args(self, grid: domain.Grid):
        """Prepare grid-specific information that will be passed as
        arguments to update_grid"""
        return (grid,)

    def update(self):
        """Update all grids"""
        for gi in self.grid_info:
            self.update_grid(*gi)

    def update_grid(self, *args):
        """Update a single grid"""
        raise NotImplementedError(
            "Classes that inherit from PerGrid must implement update_grid"
        )


def calculate_sigma(nz: int, ddl: float = 0.0, ddu: float = 0.0) -> np.ndarray:
    """Return sigma thicknesses of of layers."""
    if ddl <= 0.0 and ddu <= 0.0:
        return np.broadcast_to(1.0 / nz, (nz,))
    ddl, ddu = max(ddl, 0.0), max(ddu, 0.0)

    # This zooming routine is from Antoine Garapon, ICCH, DK
    ga = np.linspace(0.0, 1.0, nz + 1)
    ga[1:-1] = np.tanh((ddl + ddu) * ga[1:-1] - ddl) + np.tanh(ddl)
    ga[1:-1] /= np.tanh(ddl) + np.tanh(ddu)
    dga = np.diff(ga)
    assert (dga > 0.0).all(), f"ga not monotonically increasing: {ga}"
    assert dga.size == nz
    return dga


class Sigma(PerGrid):
    """Sigma coordinates with optional zooming towards bottom (ddl) and surface (ddu)"""

    def __init__(self, nz: int, ddl: float = 0.0, ddu: float = 0.0):
        super().__init__(nz)
        self.dga = calculate_sigma(nz, ddl, ddu)[:, np.newaxis, np.newaxis]

    def update_grid(self, grid: domain.Grid):
        # From sigma thicknesses as fraction [dga] to layer thicknesses in m [hn]
        np.multiply(
            self.dga, grid.D.all_values, where=grid._water, out=grid.hn.all_values
        )


class GVC(PerGrid):
    """Generalized Vertical Coordinates

    Burchard & Petersen (1997)
    https://doi.org/10.1002/(SICI)1097-0363(19971115)25:9%3C1003::AID-FLD600%3E3.0.CO;2-E
    """

    def __init__(
        self,
        nz: int,
        ddl: float = 0.0,
        ddu: float = 0.0,
        gamma_surf: bool = True,
        Dgamma: float = 0.0,
    ):
        if ddl <= 0.0 and ddu <= 0.0:
            raise Exception("ddl and/or ddu must be a positive number")
        if Dgamma <= 0.0:
            raise Exception("Dgamma must be a positive number")

        super().__init__(nz)

        dsigma = calculate_sigma(nz)
        dbeta = calculate_sigma(nz, ddl, ddu)
        k_ref = -1 if gamma_surf else 0
        self.dsigma_ref = dsigma[k_ref]
        self.dbeta_ref = dbeta[k_ref]

        if self.dbeta_ref > self.dsigma_ref:
            raise Exception(
                "This GVC parameterization would always result in equidistant layers."
                " If this is desired, use Sigma instead."
            )

        self.Dgamma = Dgamma
        self.dsigma = dsigma[:, np.newaxis, np.newaxis]
        self.dbeta = dbeta[:, np.newaxis, np.newaxis]

        # Calculate valid limit for alpha, and from that, the maximum water depth
        # NB the max alpha would be alpha_lim[alpha_lim > 0.0].min()
        # However, where alpha_lim > 0, we know dsigma - dbeta < 0.
        # Since dsigma - dbeta > -dbeta (while both are < 0), the limit must
        # then be >= 1. This limit is irrelevant as alpha <= 1
        # (unless dbeta_ref > dsigma_ref, but that case was already eliminated above)
        alpha_lim = -dbeta / (dsigma - dbeta)
        alpha_min = alpha_lim[alpha_lim < 0.0].max()
        self.D_max = (Dgamma * self.dsigma_ref) / (
            (alpha_min * self.dsigma_ref + (1.0 - alpha_min) * self.dbeta_ref)
        )

    def initialize(self, *grids: domain.Grid):
        super().initialize(*grids)
        self.logger.info(
            f"This GVC parameterization supports water depths up to {self.D_max:.3f} m"
        )

    def update_grid(self, grid: domain.Grid):
        D = grid.D.all_values

        # The aim is to give the reference layer (surface or bottom) a constant
        # thickness of Dgamma / nz = Dgamma * dsigma
        # The final calculation of the reference layer thickness blends
        # fractional thicknesses dsigma and dbeta, giving a thickness in m of
        #   (alpha * dsigma + (1 - alpha) * dbeta) * D
        # Setting this equal to Dgamma * dsigma and rearranging, we obtain
        #   alpha = (Dgamma / D * dsigma - dbeta) / (dsigma - dbeta)
        # If we additionally reduce the target thickness to D * dsigma when
        # the column height drops below Dgamma, we effectively substitute
        # min(Dgamma, D) for Dgamma. That leads to:
        #   alpha = (min(Dgamma / D, 1.0) * dsigma - dbeta) / (dsigma - dbeta)
        alpha = (
            np.minimum(self.Dgamma / D, 1.0) * self.dsigma_ref - self.dbeta_ref
        ) / (self.dsigma_ref - self.dbeta_ref)

        # Blend equal thicknesses (dsigma) with zoomed thicknesses (dbeta)
        dga = alpha * self.dsigma + (1.0 - alpha) * self.dbeta
        assert (dga > 0.0).all(where=grid._water)

        # From sigma thicknesses as fraction [dga] to layer thicknesses in m [hn]
        np.multiply(dga, D, where=grid._water, out=grid.hn.all_values)

        # NB no relaxation here - that can be done at higher level by blending ho and hn


class Adaptive(Base):
    """Adaptive coordinates - placeholder only for now"""

    # add parameters of adaptive coordinates here, after nz [but not model fields]
    def __init__(self, nz: int):
        super().__init__(nz)

    def initialize(self, tgrid: domain.Grid, *other_grids: domain.Grid):
        super().initialize(tgrid, *other_grids)

        self.tgrid = tgrid
        self.other_grids = other_grids
        self.dga_t = tgrid.array(z=CENTERS)
        self.dga_other = tuple(grid.array(z=CENTERS) for grid in other_grids)

        # Here you can obtain any other model field by name, as tgrid.domain.fields[NAME]
        # and store it as attribte of self for later use in update

    def update(self):
        # update sigma thicknesses on tgrid (self.dga_t)

        # Interpolate sigma thicknesses from T grid to other grids
        for dga in self.dga_other:
            self.dga_t.interp(dga)

        # From sigma thicknesses as fraction [dga] to layer thicknesses in m [hn]
        all_grids = (self.tgrid,) + self.other_grids
        all_dga = (self.dga_t,) + self.dga_other
        for grid, dga in zip(all_grids, all_dga):
            np.multiply(
                dga, grid.D.all_values, where=grid._water, out=grid.hn.all_values
            )
