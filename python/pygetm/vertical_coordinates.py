from typing import TYPE_CHECKING
import logging

import numpy as np

if TYPE_CHECKING:
    from . import domain
from .constants import CENTERS
from . import _pygetm


class Base:
    """Base class. It is responsible for updating all layer thicknesses
    hn for all grids provided to initialize"""

    logger: logging.Logger

    def __init__(self, nz: int):
        if nz <= 0:
            raise Exception("Number of layers nz must be a positive number")
        self.nz = nz

    def initialize(self, ref_grid: "domain.Grid", *other_grids: "domain.Grid"):
        self.logger = ref_grid.domain.root_logger.getChild("vertical_coordinates")

    def update(self, timestep: float):
        """Update layer thicknesses hn for all grids"""
        raise NotImplementedError


class PerGrid(Base):
    """Base class for vertical coordinate types that apply the same operation
    to every grid."""

    def initialize(self, *grids: "domain.Grid"):
        super().initialize(*grids)
        self.grid_info = [self.prepare_update_args(grid) for grid in grids]

    def prepare_update_args(self, grid: "domain.Grid"):
        """Prepare grid-specific information that will be passed as
        arguments to __call__"""
        return (grid.D.all_values, grid.hn.all_values, grid.mask.all_values)

    def update(self, timestep: float):
        """Update all grids"""
        for gi in self.grid_info:
            self(*gi)

    def __call__(self, D: np.ndarray, out: np.ndarray, where: np.ndarray = True):
        """Update a single grid"""
        raise NotImplementedError(
            "Classes that inherit from PerGrid must implement __call__"
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

    def __call__(self, D: np.ndarray, out: np.ndarray = None, where: np.ndarray = True):
        # From sigma thicknesses as fraction [dga] to layer thicknesses in m [hn]
        return np.multiply(self.dga, D, out=out, where=np.asarray(where, dtype=bool))


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

        self.dsigma = 1.0 / nz
        self.dbeta = calculate_sigma(nz, ddl, ddu)
        self.k_ref = -1 if gamma_surf else 0

        if self.dbeta[self.k_ref] >= self.dsigma:
            raise Exception(
                "This GVC parameterization would always result in equidistant layers."
                " If this is desired, use Sigma instead."
            )

        self.Dgamma = Dgamma

        # Calculate valid limit for alpha, and from that, the maximum water depth
        # NB the max alpha would be alpha_lim[alpha_lim > 0.0].min()
        # However, where alpha_lim > 0, we know dsigma - dbeta < 0.
        # Since dsigma - dbeta > -dbeta (while both are < 0), the limit must
        # then be >= 1. This limit is irrelevant as alpha <= 1
        # (unless dbeta[k_ref] > dsigma, but that case was already eliminated above)
        alpha_lim = -self.dbeta / (self.dsigma - self.dbeta)
        alpha_min = alpha_lim[alpha_lim < 0.0].max()
        denom = alpha_min * self.dsigma + (1.0 - alpha_min) * self.dbeta[self.k_ref]
        self.D_max = np.inf if abs(denom) < 1e-15 else (Dgamma * self.dsigma) / denom

    def initialize(self, *grids: "domain.Grid"):
        super().initialize(*grids)
        self.logger.info(
            f"This GVC parameterization supports water depths up to {self.D_max:.3f} m"
        )

    def __call__(self, D: np.ndarray, out: np.ndarray = None, where: np.ndarray = None):
        if out is None:
            out = np.empty(self.dbeta.shape + D.shape)
        if where is None:
            where = np.full(D.shape, 1)
        _pygetm.update_gvc(
            self.dsigma, self.dbeta, self.Dgamma, self.k_ref, D, where, out
        )
        return out


class Adaptive(Base):
    """Adaptive coordinates - placeholder only for now"""

    # add parameters of adaptive coordinates here, after nz [but not model fields]
    def __init__(self, nz: int):
        super().__init__(nz)

    def initialize(self, tgrid: "domain.Grid", *other_grids: "domain.Grid"):
        super().initialize(tgrid, *other_grids)

        self.tgrid = tgrid
        self.other_grids = other_grids
        self.dga_t = tgrid.array(z=CENTERS)
        self.dga_other = tuple(grid.array(z=CENTERS) for grid in other_grids)

        # Here you can obtain any other model field by name, as tgrid.domain.fields[NAME]
        # and store it as attribte of self for later use in update

    def update(self, timestep: float):
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
