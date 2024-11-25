from typing import TYPE_CHECKING, Optional, Union
import logging

import numpy as np
import numpy.typing as npt

if TYPE_CHECKING:
    from . import domain
from .constants import CENTERS, INTERFACES
from . import _pygetm


class Base:
    """Base class. It is responsible for updating layer thicknesses
    ``hn`` for all grids provided to initialize"""

    logger: logging.Logger

    def __init__(self, nz: int):
        if nz <= 0:
            raise Exception("Number of layers nz must be a positive number")
        self.nz = nz

    def initialize(
        self, ref_grid: "core.Grid", *other_grids: "core.Grid", logger: logging.Logger
    ):
        self.logger = logger

    def update(self, timestep: float):
        """Update layer thicknesses hn for all grids"""
        raise NotImplementedError


class PerGrid(Base):
    """Base class for vertical coordinate types that apply the same operation
    to every grid."""

    def initialize(self, *grids: "core.Grid", logger: logging.Logger):
        super().initialize(*grids, logger=logger)
        self.grid_info = [self.prepare_update_args(grid) for grid in grids]

    def prepare_update_args(self, grid: "core.Grid"):
        """Prepare grid-specific information that will be passed as
        arguments to __call__"""
        return (grid.D.all_values, grid.hn.all_values, grid.mask.all_values)

    def update(self, timestep: float):
        """Update all grids"""
        for gi in self.grid_info:
            self(*gi)

    def __call__(
        self,
        D: np.ndarray,
        out: Optional[np.ndarray] = None,
        where: Union[bool, npt.ArrayLike] = True,
    ):
        """Calculate layer thicknesses

        Args:
            D: water depths (m)
            out: array to hold layer thicknesses. It must have shape ``(nz,) + D.shape``
            where: locations where to compute thicknesses (typically: water points).
                It must be broadcastable to the shape of ``D``
        """
        raise NotImplementedError(
            "Classes that inherit from PerGrid must implement __call__"
        )


def calculate_sigma(nz: int, ddl: float = 0.0, ddu: float = 0.0) -> np.ndarray:
    """Return sigma thicknesses (fraction of column depth) of all layers,
    using a formulation that allows zooming towards the surface and bottom.

    Args:
        nz: number of layers
        ddl: zoom factor at bottom (0: no zooming, 2: strong zooming)
        ddu: zoom factor at surface (0: no zooming, 2: strong zooming)
    """
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
    """Sigma coordinates with optional zooming towards bottom and surface"""

    def __init__(self, nz: int, *, ddl: float = 0.0, ddu: float = 0.0):
        """
        Args:
            nz: number of layers
            ddl: zoom factor at bottom (0: no zooming, 2: strong zooming)
            ddu: zoom factor at surface (0: no zooming, 2: strong zooming)
        """
        super().__init__(nz)
        self.dga = calculate_sigma(nz, ddl, ddu)[:, np.newaxis, np.newaxis]

    def __call__(
        self,
        D: np.ndarray,
        out: Optional[np.ndarray] = None,
        where: Union[bool, npt.ArrayLike] = True,
    ) -> np.ndarray:
        # From sigma thicknesses as fraction [dga] to layer thicknesses in m [hn]
        return np.multiply(self.dga, D, out=out, where=np.asarray(where, dtype=bool))


class GVC(PerGrid):
    """Generalized Vertical Coordinates

    This blends equidistant and surface/bottom-zoomed coordinates as described in
    `Burchard & Petersen (1997)
    <https://doi.org/10.1002/(SICI)1097-0363(19971115)25%3A9%3C1003%3A%3AAID-FLD600%3E3.0.CO%3B2-E>`_.
    It is designed to keep the thickness of either the surface or bottom layer at a constant
    value, except in shallow water where all layers are assigned equal thickness.
    """

    def __init__(
        self,
        nz: int,
        *,
        ddl: float = 0.0,
        ddu: float = 0.0,
        gamma_surf: bool = True,
        Dgamma: float = 0.0,
    ):
        """
        Args:
            nz: number of layers
            ddl: zoom factor at bottom (0: no zooming, 2: strong zooming)
            ddu: zoom factor at surface (0: no zooming, 2: strong zooming)
            gamma_surf: use layers of constant thickness ``Dgamma/nz`` at surface
                (otherwise, at bottom)
            Dgamma: water depth below which to use equal layer thicknesses
        """
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

    def initialize(self, *grids: "core.Grid", logger: logging.Logger):
        super().initialize(*grids, logger=logger)
        self.logger.info(
            f"This GVC parameterization supports water depths up to {self.D_max:.3f} m"
        )

    def __call__(
        self,
        D: np.ndarray,
        out: Optional[np.ndarray] = None,
        where: Optional[np.ndarray] = None,
    ):
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
    def __init__(
        self,
        nz: int,
        ddl: float = 0.0,
        ddu: float = 0.0,
        gamma_surf: bool = True,
        Dgamma: float = 0.0,
        csigma: float = -1.0,  # 0.01
        cgvc: float = -1.0,  # 0.01
        decay: float = 2.0 / 3.0,
        hpow: int = 3,
        chsurf: float = 0.5,
        hsurf: float = 0.5,
        chmidd: float = 0.2,
        hmidd: float = -4.0,
        chbott: float = 0.3,
        hbott: float = -0.25,
        cneigh: float = 0.1,
        rneigh: float = 0.25,
        cNN: float = -1.0,
        drho: float = 0.3,
        cSS: float = -1.0,
        dvel: float = 0.1,
        chmin: float = -0.1,
        hmin: float = 0.3,
        nvfilter: int = 1,
        vfilter: float = 0.2,
        nhfilter: int = 1,
        hfilter: float = 0.1,
        tgrid: float = 14400.0,
        split: int = 1,
    ):
        """
        Args:
            nz: number of layers
            ddl: zoom factor at bottom (0: no zooming, 2: strong zooming)
            ddu: zoom factor at surface (0: no zooming, 2: strong zooming)
            gamma_surf: use layers of constant thickness `Dgamma/nz` at surface (otherwise, at bottom)
            Dgamma: water depth below which to use equal layer thicknesses
            decay: surface/bottom effect - decay by layer
            hpow: exponent for growth of Dgrid (ramp between 0 and c* tendencies)
            csigma: tendency to uniform sigma
            cgvc: tendency to "standard" gvc (w/ddu,ddl)
            chsurf: tendency to keep surface layer bounded
            hsurf: reference thickness, surface layer (relative to avg cell)
            chmidd: tendency to keep all layers bounded
            hmidd: reference thickness, other layers (relative to avg cell
            chbott: tendency to keep bottom layer bounded
            hbott: reference thickness, bottom layer (relative to avg cell)
            cneigh: tendency to keep neighbors of similar size
            rneigh: reference relative growth between neighbors
            cNN: dependence on NN (density zooming)
            drho: reference value for NN density between neighbor cells
            cSS: dependence on SS (shear zooming)
            dvel: reference value for SS absolute shear between neighbor cells
            chmin: internal nug coeff for shallow-water regions
            hmin: minimum depth
            nvfilter: Number of vertical Dgrid filter iterations [0:]
            vfilter: Strength of vertical filter-of-Dgrid [0:~0.5]
            nhfilter: Number of horizontal Dgrid filter iterations [0:]
            hfilter: Strength of horizontal filter-of-Dgrid [0:~0.5]
            tgrid: Time scale of grid adaptation
            split: Take this many partial-steps for diffusion eq.
        """

        if ddl <= 0.0 and ddu <= 0.0:
            raise Exception("ddl and/or ddu must be a positive number")
        if Dgamma <= 0.0:
            raise Exception("Dgamma must be a positive number")

        super().__init__(nz)
        self.cnp = 1.0 # needed from outside
        self.ddl = ddl
        self.ddu = ddu
        self.Dgamma = Dgamma
        self.decay = decay
        self.hpow = hpow
        self.csigma = csigma
        self.cgvc = cgvc
        self.chsurf = chsurf
        self.chbott = chbott
        self.chmidd = chmidd
        self.hsurf = hsurf
        self.hbott = hbott
        self.hmidd = hmidd
        self.cneigh = cneigh
        self.rneigh = rneigh
        self.cNN = cNN
        self.drho = drho
        self.cSS = cSS
        self.dvel = dvel
        self.chmin = chmin
        self.hmin = hmin
        self.nvfilter = nvfilter
        self.vfilter = vfilter
        self.nhfilter = nhfilter
        self.hfilter = hfilter
        self.tgrid = tgrid
        self.split = split

        # sigma = Sigma(nz, ddl, ddu)
        if self.csigma > 0.0:
            self._sigma_dga = Sigma(nz).dga
        if self.cgvc > 0.0:
            self._gvc_dga = GVC(nz).dga

    def initialize(self, tgrid: "core.Grid", *other_grids: "core.Grid", logger: logging.Logger):
        from pygetm.operators import VerticalDiffusion

        super().initialize(tgrid, *other_grids, logger=logger)
        self.logger.info( f"Initialize Adaptive coordinates")

        self._vertical_diffusion = VerticalDiffusion(tgrid, cnpar=self.cnp)

        self.tgrid = tgrid
        self.nug = tgrid.array(
            name="nug",
            units="m2 s-1",
            long_name="vertical grid diffusivity",
            z=INTERFACES,
            attrs=dict(_require_halos=True, _time_varying=True, _mask_output=True),
        )

        self.ga = tgrid.array(z=INTERFACES, attrs=dict(_require_halos=True, _time_varying=True, _mask_output=True))

        self.other_grids = other_grids
        self.dga_t = tgrid.array(z=CENTERS)
        self.dga_other = tuple(grid.array(z=CENTERS) for grid in other_grids)

        # Here you can obtain any other model field by name, as tgrid.fields[NAME]
        # and store it as attribte of self for later use in update

    def __call__(self, D: np.ndarray, out: np.ndarray = None, where: np.ndarray = None):
        """Calculate dimension less layer thicknesses

        Args:
            D: water depths (m)
            out: array to hold layer thicknesses. It must have shape ``(nz,) + D.shape``
            where: locations where to compute thicknesses (typically: water points).
                It must be broadcastable to the shape of ``D``
        """
        if out is None:
            # out = np.empty(self.dbeta.shape + D.shape)
            out = np.empty(self.nug.shape)
        if where is None:
            where = np.full(D.shape, 1)

        # first add contributions to the grid diffusion field that are
        # handled by python

        if self.csigma > 0:
            self.nug[...] = self.csigma
        else:
            self.nug = 0.0

        # Here we need to have h_gvc - and the scaling has to depend
        # on the value of gamma_surf
        # if self.cgvc > 0:
        #     self.nug(k)=self.nug(k) + cgvc*(hn_gvc(kmax)/h_gvc(k)) !*Hm1

        self.NN = np.empty(self.nug.shape)
        self.SS = np.empty(self.nug.shape)

        dt = 3600. # must come from somewhere
        # then add contributions handled by Fortran
        _pygetm.update_adaptive(
            self.nug,
            self.ga,
            self.NN,
            self.SS,
            self.decay,
            self.hpow,
            self.chsurf,
            self.hsurf,
            self.chmidd,
            self.hmidd,
            self.chbott,
            self.hbott,
            self.cneigh,
            self.rneigh,
            self.cNN,
            self.drho,
            self.cSS,
            self.dvel,
            self.chmin,
            self.hmin,
            dt,
        )
        # all contributions to nug are now added

        # apply vertical filtering from ~/python/src/filters.F90
        if self.nvfilter > 0 and self.vfilter > 0:
            #print("_pygetm.vertical_filter")
            _pygetm.vertical_filter(self.nvfilter, self.nug, self.vfilter)

        # apply horizontal filtering from ~/python/src/filters.F90
        # requires halo updates
        if self.hfilter > 0:
            for _ in range(self.nhfilter):
                #print("_pygetm.horizontal_filter")
                self.nug.update_halos()
                _pygetm.horizontal_filter(self.nug, self.hfilter)


        # now the grid diffusion field is ready to be applied
        # naming of input vectors/arrays to solver
        #KB Bjarne - GETM - pygetm
        #KB mata   -> au    -> a1
        #KB matb   -> bu    -> a2
        #KB matc   -> cu    -> a3
        #KB rhs    -> du    -> a4

        #self.nug.all_values[...] = 0.01
        #self.nug = self.nug/(2.*self.tgrid)
        self._vertical_diffusion.prepare(self.nug, dt)
        self._vertical_diffusion.apply(self.ga)

        # To get dga - that can be used to interpolate to
        # other grids for the calculation of layer heights
        self.dga_t[...] = np.diff(self.ga[...], axis=0)
        self.dga_t.update_halos()
        return out

    def update(self, timestep: float):
        # Interpolate dga from T grid to other grids
        for dga in self.dga_other:
            self.dga_t.interp(dga)
            dga.update_halos()

        # From dga to layer thicknesses in m [hn]
        all_grids = (self.tgrid,) + self.other_grids
        all_dga = (self.dga_t,) + self.dga_other
        for grid, dga in zip(all_grids, all_dga):
            np.multiply(
                dga, grid.D.all_values, where=grid._water, out=grid.hn.all_values
            )
