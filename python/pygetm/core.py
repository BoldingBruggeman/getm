import numbers
import operator
from typing import (
    Optional,
    Union,
    Tuple,
    Literal,
    Mapping,
    List,
    Any,
    Callable,
    TYPE_CHECKING,
)
import logging
import functools

import numpy as np
import numpy.lib.mixins
import numpy.typing as npt
import xarray as xr

from . import _pygetm
from . import parallel
from .constants import CENTERS, INTERFACES, FILL_VALUE, CoordinateType

if TYPE_CHECKING:
    import netCDF4


def _noop(*args, **kwargs):
    pass


class Grid(_pygetm.Grid):
    _coordinate_arrays = "x", "y", "lon", "lat"
    _readonly_arrays = _coordinate_arrays + (
        "dx",
        "dy",
        "idx",
        "idy",
        "dlon",
        "dlat",
        "area",
        "iarea",
        "cor",
    )
    _fortran_arrays = _readonly_arrays + (
        "H",
        "D",
        "mask",
        "z",
        "zo",
        "ho",
        "hn",
        "zc",
        "zf",
        "z0b",
        "z0b_min",
        "zio",
        "zin",
        "alpha",
    )
    _all_arrays = tuple(
        [f"_{n}" for n in _fortran_arrays]
        + [f"_{n}i" for n in _coordinate_arrays]
        + [f"_{n}i_" for n in _coordinate_arrays]
    )
    __slots__ = _all_arrays + (
        "type",
        "ioffset",
        "joffset",
        "postfix",
        "ugrid",
        "vgrid",
        "xgrid",
        "_sin_rot",
        "_cos_rot",
        "rotation",
        "open_boundaries",
        "input_manager",
        "default_output_transforms",
        "input_grid_mappers",
        "rivers",
        "overlap",
        "_interpolators",
        "rotated",
        "_water_contact",
        "_land",
        "_land3d",
        "_water",
        "_water_nohalo",
        "horizontal_coordinates",
        "extra_output_coordinates",
        "_mirrors",
        "tiling",
        "fields",
        "mask3d",
        "bottom_indices",
    )

    _array_args = {
        "x": dict(units="m", attrs=dict(_time_varying=False)),
        "y": dict(units="m", attrs=dict(_time_varying=False)),
        "lon": dict(
            units="degrees_east",
            long_name="longitude",
            attrs=dict(standard_name="longitude", axis="X", _time_varying=False),
        ),
        "lat": dict(
            units="degrees_north",
            long_name="latitude",
            attrs=dict(standard_name="latitude", axis="Y", _time_varying=False),
        ),
        "dx": dict(units="m", attrs=dict(_time_varying=False)),
        "dy": dict(units="m", attrs=dict(_time_varying=False)),
        "idx": dict(units="m-1", attrs=dict(_time_varying=False)),
        "idy": dict(units="m-1", attrs=dict(_time_varying=False)),
        "dlon": dict(units="degrees_east", attrs=dict(_time_varying=False)),
        "dlat": dict(units="degrees_north", attrs=dict(_time_varying=False)),
        "H": dict(
            units="m", long_name="water depth at rest", attrs=dict(_time_varying=False)
        ),
        "D": dict(
            units="m",
            long_name="water depth",
            attrs=dict(standard_name="sea_floor_depth_below_sea_surface"),
        ),
        "mask": dict(attrs=dict(_time_varying=False), fill_value=0),
        "z": dict(units="m", long_name="elevation"),
        "zo": dict(units="m", long_name="elevation at previous microtimestep"),
        "zin": dict(units="m", long_name="elevation at macrotimestep"),
        "zio": dict(units="m", long_name="elevation at previous macrotimestep"),
        "area": dict(
            units="m2",
            long_name="cell area",
            attrs=dict(standard_name="cell_area", _time_varying=False),
        ),
        "iarea": dict(
            units="m-2",
            long_name="inverse of cell area",
            attrs=dict(_time_varying=False),
        ),
        "cor": dict(
            units="1", long_name="Coriolis parameter", attrs=dict(_time_varying=False)
        ),
        "ho": dict(units="m", long_name="cell thickness at previous time step"),
        "hn": dict(
            units="m",
            long_name="cell thickness",
            attrs=dict(standard_name="cell_thickness"),
        ),
        "zc": dict(
            units="m",
            long_name="height",
            attrs=dict(
                axis="Z", positive="up", standard_name="height_above_mean_sea_level"
            ),
        ),
        "zf": dict(
            units="m",
            long_name="interface height",
            attrs=dict(
                axis="Z", positive="up", standard_name="height_above_mean_sea_level"
            ),
        ),
        "z0b": dict(units="m", long_name="hydrodynamic bottom roughness"),
        "z0b_min": dict(
            units="m",
            long_name="minimum hydrodynamic bottom roughness",
            attrs=dict(_time_varying=False),
        ),
        "alpha": dict(units="1", long_name="dampening"),
    }

    def __init__(
        self,
        nx: int,
        ny: int,
        nz: int,
        *,
        halox: int = 0,
        haloy: int = 0,
        postfix: str = "",
        ugrid: Optional["Grid"] = None,
        vgrid: Optional["Grid"] = None,
        xgrid: Optional["Grid"] = None,
        ioffset: int = 1,
        joffset: int = 1,
        overlap: int = 0,
        istart: int = 1,
        jstart: int = 1,
        fields: Optional[Mapping[str, "Array"]] = None,
        tiling: Optional[parallel.Tiling] = None,
    ):
        super().__init__(nx, ny, nz, halox, haloy, istart, jstart)
        self.postfix = postfix
        self.ioffset = ioffset
        self.joffset = joffset
        self.overlap = overlap
        self.ugrid = ugrid
        self.vgrid = vgrid
        self.xgrid = xgrid
        self.fields = {} if fields is None else fields
        self.tiling = tiling

        self._sin_rot: Optional[np.ndarray] = None
        self._cos_rot: Optional[np.ndarray] = None
        self._interpolators = {}
        self._mirrors: Mapping["Grid", Tuple[slice, slice]] = {}
        self.horizontal_coordinates: List["Array"] = []
        self.extra_output_coordinates = []
        for name in self._fortran_arrays:
            self._setup_array(name)

        self.rotation = Array.create(
            grid=self,
            dtype=self.x.dtype,
            name="rotation" + self.postfix,
            units="rad",
            long_name="grid rotation with respect to true North",
            fill_value=np.nan,
            attrs=dict(_time_varying=False),
        )

    def freeze(self):
        with np.errstate(divide="ignore"):
            self._iarea.all_values[...] = 1.0 / self._area.all_values
            self._idx.all_values[...] = 1.0 / self._dx.all_values
            self._idy.all_values[...] = 1.0 / self._dy.all_values

        for name in self._readonly_arrays:
            getattr(self, name).all_values.flags.writeable = False

        self._land = self.mask.all_values == 0
        self._water = ~self._land
        self._water_nohalo = np.full_like(self._water, False)
        interior = (
            slice(self.haloy, self.haloy + self.ny),
            slice(self.halox, self.halox + self.nx),
        )
        self._water_nohalo[interior] = self._water[interior]
        if not hasattr(self, "_water_contact"):
            self._water_contact = self._water

        self.zc.all_values[...] = -self.H.all_values
        self.zf.all_values[...] = -self.H.all_values

        self.H.all_values[self._land] = FILL_VALUE
        self.z0b_min.all_values[self._land] = FILL_VALUE
        self.H.all_values.flags.writeable = False
        self.z0b_min.all_values.flags.writeable = False
        self.z0b.all_values[...] = self.z0b_min.all_values

        # Initialize elevation at all water points to 0
        # The array will have been pre-filled with the correct fill value,
        # so land points are already ok and remain unchanged.
        self.z.all_values[self._water] = 0.0
        self.zo.all_values[...] = self.z.all_values
        self.zio.all_values[...] = self.z.all_values
        self.zin.all_values[...] = self.z.all_values

        # Calculate water depth from bathymetry and elevation
        D = self.H.all_values + self.z.all_values
        self.D.all_values[self._water] = D[self._water]

        # Determine whether any grid points are reotated with respect to true North
        # If that flag is False, it will allow us to skip potentially expensive
        # rotation operations (`rotate` method)
        self.rotated = self.rotation.all_values[self._water].any()

    def close_flux_interfaces(self):
        """Mask U and V points that do not have two bordering wet T points"""
        tmask = self.mask.all_values
        umask = (tmask[:, :-1] == 0) | (tmask[:, 1:] == 0)
        vmask = (tmask[:-1, :] == 0) | (tmask[1:, :] == 0)
        umask &= self.ugrid.mask.all_values[:, :-1] == 1
        vmask &= self.vgrid.mask.all_values[:-1, :] == 1
        self.ugrid.mask.all_values[:, :-1][umask] = 0
        self.vgrid.mask.all_values[:-1, :][vmask] = 0
        self.ugrid.mask.update_halos()
        self.vgrid.mask.update_halos()

    def infer_water_contact(self):
        umask = self.ugrid.mask
        vmask = self.vgrid.mask
        umask_backup = umask.all_values.copy()
        vmask_backup = vmask.all_values.copy()
        tmask = self.mask.all_values
        umask.all_values[:, :-1] = (tmask[:, :-1] != 0) | (tmask[:, 1:] != 0)
        vmask.all_values[:-1, :] = (tmask[:-1, :] != 0) | (tmask[1:, :] != 0)
        umask.update_halos()
        vmask.update_halos()
        self.ugrid._water_contact = umask.all_values != 0
        self.vgrid._water_contact = vmask.all_values != 0
        umask.all_values[:, :] = umask_backup
        vmask.all_values[:, :] = vmask_backup

    def _setup_array(
        self, name: str, array: Optional["Array"] = None, from_supergrid: bool = True
    ) -> "Array":
        if array is None:
            # No array provided, so it must live in Fortran; retrieve it
            kwargs = dict(fill_value=FILL_VALUE)
            kwargs.update(self._array_args[name])
            array = Array(name=name + self.postfix, **kwargs)
            setattr(self, f"_{name}", self.wrap(array, name.encode("ascii")))

        # # Obtain corresponding array on the supergrid.
        # # If this does not exist, we are done
        # source = getattr(self.domain, name + "_", None)
        # if source is None or not from_supergrid:
        #     return array

        # imax, jmax = self.ioffset + 2 * self.nx_, self.joffset + 2 * self.ny_
        # values = source[(slice(self.joffset, jmax, 2), slice(self.ioffset, imax, 2))]
        # slc = (Ellipsis,)
        # if name in ("z0b_min",):
        #     slc = self.mask.all_values[: values.shape[0], : values.shape[1]] > 0
        # array.all_values[: values.shape[0], : values.shape[1]][slc] = values[slc]
        # if values.shape != array.all_values.shape:
        #     # supergrid does not span entire grid; fill remainder by exchanging halos
        #     array.update_halos()

        # has_bounds = (
        #     self.ioffset > 0
        #     and self.joffset > 0
        #     and source.shape[-1] >= imax
        #     and source.shape[-2] >= jmax
        # )
        # if has_bounds and name in self._coordinate_arrays:
        #     # Generate interface coordinates. These are not represented in Fortran as
        #     # they are only needed for plotting. The interface coordinates are slices
        #     # that point to the supergrid data; they thus do not consume additional
        #     # memory.
        #     values_i = source[self.joffset - 1 : jmax : 2, self.ioffset - 1 : imax : 2]
        #     setattr(self, f"_{name}i_", values_i)
        #     setattr(
        #         self,
        #         f"_{name}i",
        #         values_i[self.haloy : -self.haloy, self.halox : -self.halox],
        #     )

        return array

    def interpolator(self, target: "Grid") -> Callable[[np.ndarray, np.ndarray], None]:
        ip = self._interpolators.get(target)
        if ip:
            return ip

        def _assign(x, y, xslice, yslice):
            y[yslice] = x[xslice]

        # assert self.domain is target.domain
        if self.ioffset == target.ioffset + 1 and self.joffset == target.joffset:
            # from U to T
            ip = functools.partial(_pygetm.interp_x, offset=1)
        elif self.ioffset == target.ioffset - 1 and self.joffset == target.joffset:
            # from T to U
            ip = functools.partial(_pygetm.interp_x, offset=0)
        elif self.joffset == target.joffset + 1 and self.ioffset == target.ioffset:
            # from V to T
            ip = functools.partial(_pygetm.interp_y, offset=1)
        elif self.joffset == target.joffset - 1 and self.ioffset == target.ioffset:
            # from T to V
            ip = functools.partial(_pygetm.interp_y, offset=0)
        elif self.ioffset == target.ioffset - 1 and self.joffset == target.joffset - 1:
            # from X to T
            ip = functools.partial(_pygetm.interp_xy, ioffset=0, joffset=0)
        elif self.ioffset == target.ioffset + 1 and self.joffset == target.joffset + 1:
            # from T to X
            ip = functools.partial(_pygetm.interp_xy, ioffset=1, joffset=1)
        elif self.ioffset == target.ioffset - 1 and self.joffset == target.joffset + 1:
            # from V to U (i=-1 and j=0 undefined)
            ip = functools.partial(_pygetm.interp_xy, ioffset=0, joffset=1)
        elif self.ioffset == target.ioffset + 1 and self.joffset == target.joffset - 1:
            # from U to V (i=0 and j=-1 undefined)
            ip = functools.partial(_pygetm.interp_xy, ioffset=1, joffset=0)
        elif self.ioffset == target.ioffset - 2 and self.joffset == target.joffset:
            # from T to UU (no interpolation, just copy slice)
            assert self.nx == target.nx and self.ny == target.ny
            ip = functools.partial(
                _assign,
                xslice=(Ellipsis, slice(1, None)),
                yslice=(Ellipsis, slice(0, -1)),
            )
        elif self.ioffset == target.ioffset and self.joffset == target.joffset - 2:
            # from T to VV (no interpolation, just copy slice)
            assert self.nx == target.nx and self.ny == target.ny
            ip = functools.partial(
                _assign,
                xslice=(Ellipsis, slice(1, None), slice(None)),
                yslice=(Ellipsis, slice(0, -1), slice(None)),
            )
        else:
            raise NotImplementedError(
                f"Cannot interpolate from grid type {self.postfix} "
                f"to grid type {target.postfix}"
            )
        self._interpolators[target] = ip
        return ip

    @property
    def gradient_x_calculator(self):
        target_grid = self.ugrid
        assert (self.ioffset + 1 - target_grid.ioffset) % 2 == 0
        assert (self.joffset - target_grid.joffset) % 2 == 0
        ioffset = (self.ioffset + 1 - target_grid.ioffset) // 2
        joffset = (self.joffset - target_grid.joffset) // 2
        assert ioffset in (0, 1)
        assert joffset in (-1, 0, 1)
        return target_grid, functools.partial(
            _pygetm.gradient_x, self.idx.all_values, ioffset=ioffset, joffset=joffset
        )

    @property
    def gradient_y_calculator(self):
        target_grid = self.vgrid
        assert (self.ioffset - target_grid.ioffset) % 2 == 0
        assert (self.joffset + 1 - target_grid.joffset) % 2 == 0
        ioffset = (self.ioffset - target_grid.ioffset) // 2
        joffset = (self.joffset + 1 - target_grid.joffset) // 2
        assert ioffset in (-1, 0, 1)
        assert joffset in (0, 1)
        return target_grid, functools.partial(
            _pygetm.gradient_y, self.idy.all_values, ioffset=ioffset, joffset=joffset
        )

    def rotate(
        self, u: npt.ArrayLike, v: npt.ArrayLike, to_grid: bool = True
    ) -> Tuple[npt.ArrayLike, npt.ArrayLike]:
        """Rotate a geocentric velocity field to the model coordinate system,
        or a model velocity field to the geocentric coordinate system.

        Args:
            u: velocity in x-direction in source coordinate system
                (Eastward velocity if the source is a geocentric velocity field)
            v: velocity in y-direction in source coordinate system
                (Northward velocity if the source is a geocentric velocity field)
            to_grid: rotate from geocentric to model coordinate system, not vice versa
        """
        if not self.rotated:
            return u, v
        elif self._sin_rot is None:
            self._sin_rot = np.sin(self.rotation.all_values)
            self._cos_rot = np.cos(self.rotation.all_values)

            # hardcode cos(0.5*pi)=0 to increase precision in 90 degree rotation tests
            self._cos_rot[self.rotation.all_values == 0.5 * np.pi] = 0
        sin_rot = -self._sin_rot if to_grid else self._sin_rot
        u_new = u * self._cos_rot - v * sin_rot
        v_new = u * sin_rot + v * self._cos_rot
        return u_new, v_new

    def array(self, *args, **kwargs) -> "Array":
        return Array.create(self, *args, **kwargs)

    def add_to_netcdf(self, nc: "netCDF4.Dataset", postfix: str = ""):
        xdim, ydim = "x" + postfix, "y" + postfix

        def save(name, units="", long_name=None):
            data = getattr(self, name)
            ncvar = nc.createVariable(name + postfix, data.dtype, (ydim, xdim))
            ncvar[...] = data
            ncvar.units = units
            ncvar.long_name = long_name or name

        ny, nx = self.x.shape
        nc.createDimension(xdim, nx)
        nc.createDimension(ydim, ny)
        save("dx", "m")
        save("dy", "m")
        save("H", "m", "undisturbed water depth")
        save("mask")
        save("area", "m2")
        save("cor", "s-1", "Coriolis parameter")

    def nearest_point(
        self,
        x: float,
        y: float,
        mask: Optional[Tuple[int]] = None,
        include_halos: bool = False,
        coordinate_type: Optional[CoordinateType] = None,
    ) -> Optional[Tuple[int, int]]:
        """Return index (i,j) of point nearest to specified coordinate."""
        if coordinate_type is None:
            coordinate_type = self.domain.coordinate_type
        if not self.domain.contains(
            x, y, include_halos=include_halos, coordinate_type=coordinate_type
        ):
            return None
        local_slice, _, _, _ = self.tiling.subdomain2slices(
            halox_sub=self.halox,
            haloy_sub=self.haloy,
            halox_glob=self.halox,
            haloy_glob=self.haloy,
            share=self.overlap,
            exclude_halos=not include_halos,
            exclude_global_halos=True,
        )
        spherical = coordinate_type == CoordinateType.LONLAT
        allx, ally = (self.lon, self.lat) if spherical else (self.x, self.y)
        actx, acty = allx.all_values[local_slice], ally.all_values[local_slice]
        dist = (actx - x) ** 2 + (acty - y) ** 2
        if mask is not None:
            if isinstance(mask, int):
                mask = (mask,)
            invalid = np.ones(dist.shape, dtype=bool)
            for mask_value in mask:
                invalid &= self.mask.all_values[local_slice] != mask_value
            dist[invalid] = np.inf
        idx = np.nanargmin(dist)
        j, i = np.unravel_index(idx, dist.shape)
        return j + local_slice[-2].start, i + local_slice[-1].start


for membername in Grid._all_arrays:
    info = Grid._array_args.get(membername[1:], {})
    long_name = info.get("long_name")
    units = info.get("units")
    doc = ""
    if long_name:
        doc = long_name
        if units:
            doc += f" ({units})"
    setattr(Grid, membername[1:], property(operator.attrgetter(membername), doc=doc))


class Array(_pygetm.Array, numpy.lib.mixins.NDArrayOperatorsMixin):
    __slots__ = (
        "_xarray",
        "_scatter",
        "_gather",
        "_name",
        "attrs",
        "_fill_value",
        "_ma",
        "saved",
        "_shape",
        "_ndim",
        "_size",
        "_dtype",
        "values",
        "update_halos",
        "update_halos_start",
        "update_halos_finish",
        "compare_halos",
        "open_boundaries",
    )
    grid: Grid

    def __init__(
        self,
        name: Optional[str] = None,
        units: Optional[str] = None,
        long_name: Optional[str] = None,
        fill_value: Optional[Union[float, int]] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype: Optional[npt.DTypeLike] = None,
        grid: Grid = None,
        fabm_standard_name: Optional[str] = None,
        attrs: Mapping[str, Any] = {},
    ):
        _pygetm.Array.__init__(self, grid)
        self._xarray: Optional[xr.DataArray] = None
        self._scatter: Optional[parallel.Scatter] = None
        self._gather: Optional[parallel.Gather] = None
        assert (
            fill_value is None or np.ndim(fill_value) == 0
        ), "fill_value must be a scalar value"
        self._name = name
        self.attrs = attrs.copy()
        if units:
            self.attrs["units"] = units
        if long_name:
            self.attrs["long_name"] = long_name
        if fabm_standard_name:
            self.attrs.setdefault("_fabm_standard_names", set()).add(fabm_standard_name)
        self._fill_value = (
            fill_value
            if fill_value is None or dtype is None
            else np.array(fill_value, dtype=dtype)
        )
        self._ma = None
        self.saved = False  #: to be set if this variable is requested for output
        self._shape = shape
        self._ndim = None if shape is None else len(shape)
        self._size = None if shape is None else np.prod(shape)
        self._dtype = dtype
        self.values = None

    def set_fabm_standard_name(self, fabm_standard_name):
        self.attrs.setdefault("_fabm_standard_names", set()).add(fabm_standard_name)

    fabm_standard_name = property(fset=set_fabm_standard_name)

    def mirror(self, target: Optional["Array"] = None):
        target = target or self
        m = self.grid._mirrors.get(target.grid)
        if m is not None:
            source_slice, target_slice = m
            target.all_values[target_slice] = self.all_values[source_slice]

    def finish_initialization(self):
        """This is called by the underlying cython implementation after the array
        receives a value (:attr:`all_values` is valid)
        """
        assert self.grid is not None
        self._dtype = self.all_values.dtype
        self._ndim = self.all_values.ndim
        if self._fill_value is not None:
            # Cast fill value to dtype of the array
            self._fill_value = np.array(self._fill_value, dtype=self._dtype)
        if self.on_boundary or self._ndim == 0:
            # boundary array or scalar
            self.values = self.all_values[...]
        else:
            ny, nx = self.all_values.shape[-2:]
            halox, haloy = self.grid.halox, self.grid.haloy
            self.values = self.all_values[..., haloy : ny - haloy, halox : nx - halox]

        self._shape = self.values.shape
        self._size = self.values.size

        if not self.grid.tiling:
            self.update_halos = _noop
            self.update_halos_start = _noop
            self.update_halos_finish = _noop
            self.compare_halos = _noop
        else:
            self.update_halos = functools.partial(self._distribute, "update_halos")
            self.update_halos_start = functools.partial(
                self._distribute, "update_halos_start"
            )
            self.update_halos_finish = functools.partial(
                self._distribute, "update_halos_finish"
            )
            self.compare_halos = functools.partial(self._distribute, "compare_halos")

    def register(self):
        assert self.grid is not None
        if self._name is not None:
            if self._name in self.grid.fields:
                raise Exception(
                    f"A field with name {self._name!r} has already been registered"
                    " with the field manager."
                )
            self.grid.fields[self._name] = self

    def __repr__(self) -> str:
        return super().__repr__() + self.grid.postfix

    def _distribute(self, method: str, *args, **kwargs) -> parallel.DistributedArray:
        dist = parallel.DistributedArray(
            self.grid.tiling,
            self.all_values,
            self.grid.halox,
            self.grid.haloy,
            overlap=self.grid.overlap,
        )
        self.update_halos = dist.update_halos
        self.update_halos_start = dist.update_halos_start
        self.update_halos_finish = dist.update_halos_finish
        self.compare_halos = dist.compare_halos
        getattr(self, method)(*args, **kwargs)

    def scatter(self, global_data: Optional[np.ndarray]):
        if self.grid.tiling.n == 1:
            if self.grid.tiling.rank == 0:
                self.values[...] = global_data
            return
        if self._scatter is None:
            self._scatter = parallel.Scatter(
                self.grid.tiling,
                self.all_values,
                halox=self.grid.halox,
                haloy=self.grid.haloy,
                share=self.grid.overlap,
                fill_value=self._fill_value,
            )
        self._scatter(global_data)

    def gather(self, out: Optional["Array"] = None, slice_spec=()):
        if self.grid.tiling.n == 1:
            if out is not None:
                out[slice_spec + (Ellipsis,)] = self.values
            return self
        if self._gather is None:
            self._gather = parallel.Gather(
                self.grid.tiling,
                self.values.shape,
                self.dtype,
                fill_value=self._fill_value,
            )
        result = self._gather(
            self.values,
            out.values if isinstance(out, Array) else out,
            slice_spec=slice_spec,
        )
        if result is not None and out is None:
            out = self.grid.domain.glob.grids[self.grid.type].array(dtype=self.dtype)
            out[...] = result
        return out

    def allgather(self) -> np.ndarray:
        if self.grid.tiling.n == 1:
            return self.values
        if self.on_boundary:
            comm = self.grid.tiling.comm
            open_boundaries = self.grid.open_boundaries

            # Gather the number of open boundary points in each subdomain
            np_bdy = np.empty((comm.size,), dtype=int)
            np_local = np.array(self.grid.open_boundaries.np, dtype=int)
            comm.Allgather(np_local, np_bdy)

            # Gather the global indices of open boundary points from each subdomain
            indices = np.empty((np_bdy.sum(),), dtype=int)
            comm.Allgatherv(open_boundaries.local_to_global_indices, (indices, np_bdy))
            assert frozenset(indices) == frozenset(range(open_boundaries.np_glob))

            # Gather the values at the open boundary points from each subdomain
            values = np.empty((np_bdy.sum(),), dtype=self.values.dtype)
            comm.Allgatherv(self.values, (values, np_bdy))

            # Map retrieved values to the appropriate indices in the global array
            all_values = np.empty((open_boundaries.np_glob,), dtype=values.dtype)
            all_values[indices] = values
            return all_values

    def global_sum(
        self,
        reproducible: bool = False,
        where: Optional["Array"] = None,
        to_all: bool = False,
    ) -> Optional[np.ndarray]:
        if reproducible:
            assert not to_all
            all = self.gather()
            if where is not None:
                where = where.gather()
            if all is not None:
                return all.values.sum(
                    where=np._NoValue if where is None else where.values
                )
        else:
            local_sum = self.values.sum(
                where=np._NoValue if where is None else where.values
            )
            tiling = self.grid.tiling
            reduce = tiling.allreduce if to_all else tiling.reduce
            return reduce(local_sum)

    def global_mean(
        self, reproducible: bool = False, where: Optional["Array"] = None
    ) -> Optional[np.ndarray]:
        sum = self.global_sum(reproducible=reproducible, where=where)
        if where is not None:
            count = where.global_sum()
        else:
            count = self.grid.tiling.reduce(self.values.size)
        if sum is not None:
            return sum / count

    @staticmethod
    def create(
        grid: Grid,
        fill: Optional[npt.ArrayLike] = None,
        z: Literal[None, True, False, CENTERS, INTERFACES] = None,
        dtype: npt.DTypeLike = None,
        on_boundary: bool = False,
        register: bool = True,
        **kwargs,
    ) -> "Array":
        """Create a new :class:`Array`

        Args:
            grid: grid associated with the new array
            fill: value to set the new array to
            z: vertical dimension of the new array.
                ``False`` for a 2D array, ``CENTERS`` (or ``True``) for an array
                defined at the layer centers, ``INTERFACES`` for an array defined at
                the layer interfaces. ``None`` to detect from ``fill``.
            dtype: data type
            on_boundary: whether to describe data along the open boundaries (1D),
                instead of the 2D x-y model domain
            register: whether to register the array as field available for output
            **kwargs: additional keyword arguments passed to :class:`Array`
        """
        ar = Array(grid=grid, **kwargs)
        if fill is None and ar.fill_value is not None:
            fill = ar.fill_value
        if fill is not None:
            fill = np.asarray(fill)
            if z is None and not on_boundary:
                if fill.ndim != 3:
                    z = False
                elif fill.shape[0] == grid.nz_ + 1:
                    z = INTERFACES
                else:
                    z = CENTERS
        if dtype is None:
            dtype = float if fill is None else fill.dtype
        shape = [grid.open_boundaries.np] if on_boundary else [grid.ny_, grid.nx_]
        if z:
            nz = grid.nz_ + 1 if z == INTERFACES else grid.nz_
            shape.insert(1 if on_boundary else 0, nz)
        data = ar.allocate(shape, dtype)
        if fill is not None:
            data[...] = fill
        ar.wrap_ndarray(data, on_boundary=on_boundary, register=register)
        return ar

    def fill(self, value):
        """Set array to specified value, while respecting the mask: masked points are
        set to :attr:`fill_value`
        """
        try:
            self.all_values[...] = value
        except ValueError:
            self.values[...] = value
            self.update_halos()
        if self.fill_value is not None and not (self.ndim == 0 or self.on_boundary):
            self.all_values[..., self.grid._land] = self.fill_value

    @property
    def ma(self) -> np.ma.MaskedArray:
        """Masked array representation that combines the data and the mask associated
        with the array's native grid
        """
        if self._ma is None:
            if self.size == 0 or self.on_boundary:
                mask = False
            else:
                mask = np.isin(self.grid.mask.values, (1, 2), invert=True)
            self._ma = np.ma.array(self.values, mask=np.broadcast_to(mask, self._shape))
        return self._ma

    def plot(self, mask: bool = True, **kwargs):
        """Plot the array with :meth:`xarray.DataArray.plot`

        Args:
            **kwargs: additional keyword arguments passed to
                :meth:`xarray.DataArray.plot`
        """
        if "x" not in kwargs and "y" not in kwargs and self.grid.horizontal_coordinates:
            x, y = self.grid.horizontal_coordinates
            kwargs.update(x=x.name, y=y.name)
        if "shading" not in kwargs:
            kwargs["shading"] = "auto"
        return self.as_xarray(mask=mask).plot(**kwargs)

    def interp(
        self,
        target: Union["Array", Grid],
        z: Literal[None, True, False, CENTERS, INTERFACES] = None,
    ) -> "Array":
        """Interpolate the array to another grid.

        Args:
            target: either the :class:`Array` that will hold the interpolated data,
                or the :class:`~pygetm.core.Grid` to interpolate to. If a ``Grid`` is
                provided, a new array will be created to hold the interpolated values.
        """
        if not isinstance(target, Array):
            # Target must be a grid; we need to create the array
            target_z = z if z is not None else self.z
            target = Array.create(target, dtype=self._dtype, z=target_z)
        source_array = self.all_values
        target_array = target.all_values
        if self.grid is target.grid:
            if self.z == INTERFACES and target.z == CENTERS:
                # vertical interpolation from layer interfaces to layer centers
                _pygetm.interp_z(source_array, target_array, offset=0)
            elif self.z == CENTERS and target.z == INTERFACES:
                # vertical interpolation from layer centers to layer interfaces
                # (top and bottom interfaces will be left untouched)
                _pygetm.interp_z(source_array, target_array, offset=1)
        else:
            if self._ndim == 2:
                source_array = source_array[None, ...]
                target_array = target_array[None, ...]
            interpolate = self.grid.interpolator(target.grid)
            interpolate(source_array, target_array)
        return target

    def gradient_x(self, target: Optional["Array"] = None) -> "Array":
        target_grid, calculator = self.grid.gradient_x_calculator
        if target is None:
            target = Array.create(target_grid, dtype=self._dtype, z=self.z)
        calculator(self.all_values, target.all_values)
        return target

    def gradient_y(self, target: Optional["Array"] = None) -> "Array":
        target_grid, calculator = self.grid.gradient_y_calculator
        if target is None:
            target = Array.create(target_grid, dtype=self._dtype, z=self.z)
        calculator(self.all_values, target.all_values)
        return target

    def __array__(self, dtype: Optional[npt.DTypeLike] = None, copy=None) -> np.ndarray:
        """Return interior of the array as a NumPy array.
        No copy will be made unless the requested data type differs from that
        of the underlying array.

        Args:
            dtype: data type
        """
        return np.asarray(self.values, dtype=dtype)

    def isel(self, *, z: int, **kwargs) -> "Array":
        """Select a single depth level. The data in the returned 2D :class:`Array`
        will be a view of the relevant data of the original 3D array. Thus, changes
        in one will affect the other.
        """
        if self._ndim != 3:
            raise NotImplementedError
        if self.units is not None:
            kwargs.setdefault("units", self.units)
        if self.long_name is not None:
            kwargs.setdefault("long_name", f"{self.long_name} @ k={z}")
        kwargs["attrs"] = kwargs.get("attrs", {}).copy()
        for att in ("_mask_output",):
            if att in self.attrs:
                kwargs["attrs"][att] = self.attrs[att]
        ar = Array(grid=self.grid, fill_value=self.fill_value, **kwargs)
        ar.wrap_ndarray(self.all_values[z, ...])
        return ar

    def __getitem__(self, key) -> np.ndarray:
        """Retrieve values from the interior of the array (excluding halos).
        For access to the halos, use :attr:`all_values`.
        """
        return self.values[key]

    def __setitem__(self, key, values):
        """Assign values to the interior of the array (excluding halos).
        For access to the halos, use :attr:`all_values`.
        """
        self.values[key] = values

    @property
    def shape(self) -> Tuple[int, ...]:
        """Shape excluding halos"""
        return self._shape

    @property
    def ndim(self) -> int:
        """Number of dimensions"""
        return self._ndim

    @property
    def z(self):
        """Vertical dimension: ``False`` if the array has no vertical dimension,
        ``CENTERS`` for layer centers, ``INTERFACES`` for layer interfaces.
        """
        if self._ndim != (2 if self.on_boundary else 3):
            return False
        nz = self._shape[1 if self.on_boundary else 0]
        return INTERFACES if nz == self.grid.nz_ + 1 else CENTERS

    @property
    def size(self) -> int:
        """Total number of values, excluding halos"""
        return self._size

    @property
    def dtype(self) -> npt.DTypeLike:
        """Data type"""
        return self._dtype

    @property
    def name(self) -> Optional[str]:
        """Name"""
        return self._name

    @property
    def units(self) -> Optional[str]:
        """Units"""
        return self.attrs.get("units")

    @property
    def long_name(self) -> Optional[str]:
        """Long name"""
        return self.attrs.get("long_name") or self.name

    @property
    def fill_value(self) -> Optional[Union[int, float]]:
        """Fill value"""
        return self._fill_value

    # Below based on https://np.org/devdocs/reference/generated/np.lib.mixins.NDArrayOperatorsMixin.html#np.lib.mixins.NDArrayOperatorsMixin
    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__":
            return NotImplemented

        out = kwargs.get("out", ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, (np.ndarray, numbers.Number, Array)):
                return NotImplemented
            if isinstance(x, Array) and x.grid is not self.grid:
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.all_values if isinstance(x, Array) else x for x in inputs)
        if out:
            kwargs["out"] = tuple(
                x.all_values if isinstance(x, Array) else x for x in out
            )
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(self.create(self.grid, x) for x in result)
        elif method == "at":
            # no return value
            return None
        else:
            # one return value
            return self.create(self.grid, result)

    def set(self, value: Union[float, np.ndarray, xr.DataArray], **kwargs):
        """Link this array to a field or value using
        :attr:`~pygetm.domain.Domain.input_manager`, which will perform temporal and
        spatial interpolation as required.

        Args:
            value: value to assign to this array. If it is time-dependent (if you pass
                an instance of :class:`xarray.DataArray` with a time dimension),
                the array's value will be updated during the simulation whenever
                :meth:`pygetm.input.InputManager.update` is called.
            **kwargs: keyword arguments passed to :meth:`pygetm.input.InputManager.add`
        """
        self.grid.input_manager.add(self, value, **kwargs)

    def require_set(self, logger: Optional[logging.Logger] = None):
        """Assess whether all non-masked cells of this field have been set. If not, an
        error message is written to the log and False is returned.
        """
        valid = True
        if self._fill_value is not None:
            invalid = self.ma == self._fill_value
            if invalid.any():
                (logger or logging.getLogger()).error(
                    f"{self.name} is masked ({self._fill_value})"
                    f" in {invalid.sum()} active grid cells."
                )
                valid = False
        return valid

    def as_xarray(self, mask: bool = False) -> xr.DataArray:
        """Return this array wrapped in an :class:`xarray.DataArray` that includes
        coordinates and can be used for plotting
        """
        if self._xarray is not None and not mask:
            return self._xarray
        attrs = {}
        for key in ("units", "long_name"):
            value = getattr(self, key)
            if value is not None:
                attrs[key] = value
        coords = {}
        if not (
            self is self.grid.x
            or self is self.grid.y
            or self is self.grid.lon
            or self is self.grid.lat
        ):
            coords[f"x{self.grid.postfix}"] = self.grid.x.xarray
            coords[f"y{self.grid.postfix}"] = self.grid.y.xarray
            coords[f"lon{self.grid.postfix}"] = self.grid.lon.xarray
            coords[f"lat{self.grid.postfix}"] = self.grid.lat.xarray
        dims = ("y" + self.grid.postfix, "x" + self.grid.postfix)
        if self.ndim == 3:
            dims = ("zi" if self.z == INTERFACES else "z",) + dims
        values = self.values if not mask else self.ma
        _xarray = xr.DataArray(
            values, coords=coords, dims=dims, attrs=attrs, name=self.name
        )
        if not mask:
            self._xarray = _xarray
        return _xarray

    xarray = property(as_xarray)
