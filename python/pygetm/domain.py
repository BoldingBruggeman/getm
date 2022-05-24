from typing import Optional, Tuple, List, Mapping
import operator
import logging
import os.path
import enum
import collections

import numpy
import numpy.typing
import xarray
import netCDF4

from . import _pygetm
from . import core
from . import parallel
from . import output
from . import input
from .constants import FILL_VALUE, CENTERS, INTERFACES

WEST  = 1
NORTH = 2
EAST  = 3
SOUTH = 4


class VerticalCoordinates(enum.IntEnum):
    SIGMA = 1
#    Z = 2
    GVC = 3
#    HYBRID = 4
#    ADAPTIVE = 5

def find_interfaces(c: numpy.ndarray):
    c_if = numpy.empty((c.size + 1),)
    d = numpy.diff(c)
    c_if[1:-1] = c[:-1] + 0.5 * d
    c_if[0] = c[0] - 0.5 * d[0]
    c_if[-1] = c[-1] + 0.5 * d[-1]
    return c_if

class Grid(_pygetm.Grid):
    _coordinate_arrays = 'x', 'y', 'lon', 'lat'
    _readonly_arrays = _coordinate_arrays + ('dx', 'dy', 'idx', 'idy', 'dlon', 'dlat', 'area', 'iarea', 'cor')
    _fortran_arrays = _readonly_arrays + ('H', 'D', 'mask', 'z', 'zo', 'ho', 'hn', 'zc', 'zf', 'z0b', 'z0b_min', 'zio', 'zin', 'alpha')
    _all_arrays = tuple(['_%s' % n for n in _fortran_arrays] + ['_%si' % n for n in _coordinate_arrays] + ['_%si_' % n for n in _coordinate_arrays])
    __slots__ = _all_arrays + ('halo', 'type', 'ioffset', 'joffset', 'postfix', 'ugrid', 'vgrid', '_sin_rot', '_cos_rot', 'rotation', 'nbdyp', 'overlap')

    _array_args = {
        'x': dict(units='m', constant=True, fill_value=FILL_VALUE),
        'y': dict(units='m', constant=True, fill_value=FILL_VALUE),
        'lon': dict(units='degrees_north', long_name='longitude', constant=True, fill_value=FILL_VALUE),
        'lat': dict(units='degrees_east', long_name='latitude', constant=True, fill_value=FILL_VALUE),
        'dx': dict(units='m', constant=True, fill_value=FILL_VALUE),
        'dy': dict(units='m', constant=True, fill_value=FILL_VALUE),
        'idx': dict(units='m-1', constant=True, fill_value=FILL_VALUE),
        'idy': dict(units='m-1', constant=True, fill_value=FILL_VALUE),
        'dlon': dict(units='degrees_north', constant=True, fill_value=FILL_VALUE),
        'dlat': dict(units='degrees_east', constant=True, fill_value=FILL_VALUE),
        'H': dict(units='m', long_name='water depth at rest', constant=True, fill_value=FILL_VALUE),
        'D': dict(units='m', long_name='water depth', fill_value=FILL_VALUE),
        'mask': dict(constant=True, fill_value=0),
        'z': dict(units='m', long_name='elevation', fill_value=FILL_VALUE),
        'zo': dict(units='m', long_name='elevation at previous microtimestep', fill_value=FILL_VALUE),
        'zin': dict(units='m', long_name='elevation at macrotimestep', fill_value=FILL_VALUE),
        'zio': dict(units='m', long_name='elevation at previous macrotimestep', fill_value=FILL_VALUE),
        'area': dict(units='m2', long_name='grid cell area', constant=True, fill_value=FILL_VALUE),
        'iarea': dict(units='m-2', long_name='inverse of grid cell area', constant=True, fill_value=FILL_VALUE),
        'cor': dict(units='1', long_name='Coriolis parameter', constant=True, fill_value=FILL_VALUE),
        'ho': dict(units='m', long_name='layer heights at previous time step', fill_value=FILL_VALUE),
        'hn': dict(units='m', long_name='layer heights', fill_value=FILL_VALUE),
        'zc': dict(units='m', long_name='depth', fill_value=FILL_VALUE),
        'zf': dict(units='m', long_name='interface depth', fill_value=FILL_VALUE),
        'z0b': dict(units='m', long_name='hydrodynamic bottom roughness', fill_value=FILL_VALUE),
        'z0b_min': dict(units='m', long_name='physical bottom roughness', constant=True, fill_value=FILL_VALUE),
        'alpha': dict(units='1', long_name='dampening', fill_value=FILL_VALUE),
    }

    def __init__(self, domain: 'Domain', grid_type: int, ioffset: int, joffset: int, overlap: int=0, ugrid: Optional['Grid']=None, vgrid: Optional['Grid']=None):
        _pygetm.Grid.__init__(self, domain, grid_type)
        self.halo = domain.halo
        self.type = grid_type
        self.ioffset = ioffset
        self.joffset = joffset
        self.overlap = overlap
        self.postfix = {_pygetm.TGRID: 't', _pygetm.UGRID: 'u', _pygetm.VGRID: 'v', _pygetm.XGRID: 'x', _pygetm.UUGRID: '_uu_adv', _pygetm.VVGRID: '_vv_adv', _pygetm.UVGRID: '_uv_adv', _pygetm.VUGRID: '_vu_adv'}[grid_type]
        self.ugrid: Optional[Grid] = ugrid
        self.vgrid: Optional[Grid] = vgrid
        self._sin_rot: Optional[numpy.ndarray] = None
        self._cos_rot: Optional[numpy.ndarray] = None

        for name in self._readonly_arrays:
            self._setup_array(name, register=False)
        self._iarea.all_values[...] = 1. / self._area.all_values
        self._idx.all_values[...] = 1. / self._dx.all_values
        self._idy.all_values[...] = 1. / self._dy.all_values
        for name in self._readonly_arrays:
            getattr(self, name).all_values.flags.writeable = True

    def initialize(self, nbdyp: int):
        for name in self._fortran_arrays:
            if name in self._readonly_arrays:
                getattr(self, name).register()
            else:
                self._setup_array(name)
        self.rotation = core.Array.create(grid=self, dtype=self.x.dtype, name='rotation' + self.postfix, units='rad', long_name='grid rotation with respect to true North')
        self._setup_array(name, self.rotation)
        self.zc.all_values.fill(0.)
        self.zf.all_values.fill(0.)
        self.z0b.all_values[...] = self.z0b_min.all_values
        self.zo.all_values[...] = self.z.all_values
        self.zio.all_values[...] = self.z.all_values
        self.zin.all_values[...] = self.z.all_values
        self.nbdyp = nbdyp

    def _setup_array(self, name: str, array: Optional[core.Array]=None, register: bool=True):
        if array is None:
            # No array provided, so it must live in Fortran; retrieve it
            array = core.Array(name=name + self.postfix, **self._array_args[name])
            setattr(self, '_%s' % name, self.wrap(array, name.encode('ascii'), register=register))

        # Obtain corresponding array on the supergrid. If this does not exist, we are done
        source = getattr(self.domain, name + '_', None)
        if source is None:
            return

        nj, ni = self.ny_, self.nx_
        has_bounds = self.ioffset > 0 and self.joffset > 0 and self.domain.H_.shape[-1] >= self.ioffset + 2 * ni and self.domain.H_.shape[-2] >= self.joffset + 2 * nj
        valid = self.domain.mask_[self.joffset:self.joffset + 2 * nj:2, self.ioffset:self.ioffset + 2 * ni:2] > 0
        values = source[self.joffset:self.joffset + 2 * nj:2, self.ioffset:self.ioffset + 2 * ni:2]
        array.all_values.fill(numpy.nan)
        slc = valid if name in ('z', 'z0b_min') else (Ellipsis,)
        array.all_values[:values.shape[0], :values.shape[1]][slc] = values[slc]
        if has_bounds and name in self._coordinate_arrays:
            # Generate interface coordinates. These are not represented in Fortran as they are only needed for plotting.
            # The interface coordinates are slices that point to the supergrid data; they thus do not consume additional memory.
            values_i = source[self.joffset - 1:self.joffset + 2 * nj + 1:2, self.ioffset - 1:self.ioffset + 2 * ni + 1:2]
            setattr(self, '_%si_' % name, values_i)
            setattr(self, '_%si' % name, values_i[self.halo:-self.halo, self.halo:-self.halo])

    def rotate(self, u: numpy.typing.ArrayLike, v: numpy.typing.ArrayLike, to_grid: bool= True) -> Tuple[numpy.typing.ArrayLike, numpy.typing.ArrayLike]:
        if self._sin_rot is None:
            self._sin_rot = numpy.sin(self.rotation.all_values)
            self._cos_rot = numpy.cos(self.rotation.all_values)
        sin_rot = -self._sin_rot if to_grid else self._sin_rot
        u_new = u * self._cos_rot - v * sin_rot
        v_new = u * sin_rot + v * self._cos_rot
        return u_new, v_new

    def array(self, *args, **kwargs) -> core.Array:
        return core.Array.create(self,  *args, **kwargs)

    def add_to_netcdf(self, nc: netCDF4.Dataset, postfix: str=''):
        xdim, ydim = 'x' + postfix, 'y' + postfix
        def save(name, units='', long_name=None):
            data = getattr(self, name)
            ncvar = nc.createVariable(name + postfix, data.dtype, (ydim, xdim))
            ncvar[...] = data
            ncvar.units = units
            ncvar.long_name = long_name or name
        ny, nx = self.x.shape
        nc.createDimension(xdim, nx)
        nc.createDimension(ydim, ny)
        save('dx', 'm')
        save('dy', 'm')
        save('H', 'm', 'undisturbed water depth')
        save('mask')
        save('area', 'm2')
        save('cor', 's-1', 'Coriolis parameter')

    def nearest_point(self, x: float, y: float, mask: Optional[Tuple[int]]=None, include_halos: bool=False) -> Optional[Tuple[int, int]]:
        """Return index (i,j) of point nearest to specified coordinate."""
        if not self.domain.contains(x, y, include_halos=include_halos):
            return None
        local_slice, _, _, _ = self.domain.tiling.subdomain2slices(halo_sub=self.halo, halo_glob=self.halo, share=self.overlap, exclude_halos=not include_halos, exclude_global_halos=True)
        allx, ally = (self.lon, self.lat) if self.domain.spherical else (self.x, self.y)
        dist = (allx.all_values[local_slice] - x)**2 + (ally.all_values[local_slice] - y)**2
        if mask is not None:
            if isinstance(mask, int):
                mask = (mask,)
            invalid = numpy.ones(dist.shape, dtype=bool)
            for mask_value in mask:
                invalid = numpy.logical_and(invalid, self.mask.all_values[local_slice] != mask_value)
            dist[invalid] = numpy.inf
        idx = numpy.nanargmin(dist)
        j, i = numpy.unravel_index(idx, dist.shape)
        return j + local_slice[-2].start, i + local_slice[-1].start

for membername in Grid._all_arrays:
    info = Grid._array_args.get(membername[1:], {})
    long_name = info.get('long_name')
    units = info.get('units')
    doc = ''
    if long_name:
        doc = long_name
        if units:
            doc += ' (%s)' % units
    setattr(Grid, membername[1:], property(operator.attrgetter(membername), doc=doc))

def read_centers_to_supergrid(ncvar, ioffset: int, joffset: int, nx: int, ny: int, dtype=None):
    if dtype is None:
        dtype = ncvar.dtype
    data = numpy.ma.masked_all(ncvar.shape[:-2] + (ny * 2 + 1, nx * 2 + 1), dtype=dtype)

    # Create an array to data at centers (T points),
    # with strips of size 1 on all sides to support interpolation to interfaces
    data_centers = numpy.ma.masked_all(ncvar.shape[:-2] + (ny + 2, nx + 2), dtype=dtype)

    masked_values = []
    if hasattr(ncvar, 'missing_value'):
        masked_values.append(numpy.array(ncvar.missing_value, ncvar.dtype))

    # Extend the read domain (T grid) by 1 each side, where possible
    # That will allow us to interpolate (rater than extrapolate) to values at the interfaces
    ex_imin = 0 if ioffset == 0 else 1
    ex_imax = 0 if ioffset + nx == ncvar.shape[-1] else 1
    ex_jmin = 0 if joffset == 0 else 1
    ex_jmax = 0 if joffset + ny == ncvar.shape[-2] else 1
    data_centers[..., 1 - ex_jmin:1 + ny + ex_jmax, 1 - ex_imin:1 + nx + ex_imax] = ncvar[..., joffset - ex_jmin:joffset + ny + ex_jmax, ioffset - ex_imin:ioffset + nx + ex_imax]
    for value in masked_values:
        data_centers = numpy.ma.masked_equal(data_centers, value, copy=False)

    data_if_ip = numpy.ma.masked_all((4,) + ncvar.shape[:-2] + (ny + 1, nx + 1), dtype=dtype)
    data_if_ip[0, ...] = data_centers[...,  :-1,  :-1]
    data_if_ip[1, ...] = data_centers[..., 1:,    :-1]
    data_if_ip[2, ...] = data_centers[...,  :-1, 1:  ]
    data_if_ip[3, ...] = data_centers[..., 1:,   1:  ]
    data[..., 1::2, 1::2] = data_centers[1:-1, 1:-1]
    data[..., ::2, ::2] = data_if_ip.mean(axis=0)

    data_if_ip[0, ..., :-1] = data_centers[...,  :-1,  1:-1]
    data_if_ip[1, ..., :-1] = data_centers[..., 1:,    1:-1]
    data[..., ::2, 1::2] = data_if_ip[:2, ..., :-1].mean(axis=0)

    data_if_ip[0, ..., :-1, :] = data_centers[..., 1:-1,  :-1]
    data_if_ip[1, ..., :-1, :] = data_centers[..., 1:-1, 1:, ]
    data[..., 1::2, ::2] = data_if_ip[:2, ..., :-1, :].mean(axis=0)

    if len(masked_values) > 0:
        data.set_fill_value(masked_values[0])

    return data

deg2rad = numpy.pi / 180        # degree to radian conversion
rearth = 6378815.               # radius of the earth (m)
omega = 2. * numpy.pi / 86164.  # rotation rate of the earth (rad/s), 86164 is number of seconds in a sidereal day

def center_to_supergrid_1d(data) -> numpy.ndarray:
    assert data.ndim == 1, 'data must be one-dimensional'
    assert data.size > 1, 'data must have at least 2 elements'
    data_sup = numpy.empty((data.size * 2 + 1,))
    data_sup[1::2] = data
    data_sup[2:-2:2] = 0.5 * (data[1:] + data[:-1])
    data_sup[0] = 2 * data_sup[1] - data_sup[2]
    data_sup[-1] = 2 * data_sup[-2] - data_sup[-3]
    return data_sup

def interfaces_to_supergrid_1d(data, out: Optional[numpy.ndarray]=None) -> numpy.ndarray:
    assert data.ndim == 1, 'data must be one-dimensional'
    assert data.size > 1, 'data must have at least 2 elements'
    if out is None:
        out = numpy.empty((data.size * 2 - 1,))
    out[0::2] = data
    out[1::2] = 0.5 * (data[1:] + data[:-1])
    return out

def interfaces_to_supergrid_2d(data, out: Optional[numpy.ndarray]=None) -> numpy.ndarray:
    assert data.ndim == 2, 'data must be two-dimensional'
    assert data.shape[0] > 1 and data.shape[1] > 1, 'data must have at least 2 elements'
    if out is None:
        out = numpy.empty((data.shape[1] * 2 - 1, data.shape[0] * 2 - 1))
    out[0::2, 0::2] = data
    out[1::2, 0::2] = 0.5 * (data[:-1, :] + data[1:, :])
    out[0::2, 1::2] = 0.5 * (data[:,:-1] + data[:,1:])
    out[1::2, 1::2] = 0.25 * (data[:-1,:-1] + data[:-1,1:] + data[1:,:-1] + data[1:,1:])
    return out

def create_cartesian(x, y, nz: int, interfaces=False, **kwargs) -> 'Domain':
    """Create Cartesian domain from 1D arrays with x coordinates and  y coordinates."""
    assert x.ndim == 1, 'x coordinate must be one-dimensional'
    assert y.ndim == 1, 'y coordinate must be one-dimensional'

    if interfaces:
        nx, ny = x.size - 1, y.size - 1
        x, y = interfaces_to_supergrid_1d(x), interfaces_to_supergrid_1d(y)
    else:
        nx, ny = x.size, y.size
        x, y = center_to_supergrid_1d(x), center_to_supergrid_1d(y)
    return Domain.create(nx, ny, nz, x=x, y=y[:, numpy.newaxis], **kwargs)

def create_spherical(lon, lat, nz: int, interfaces=False, **kwargs) -> 'Domain':
    """Create spherical domain from 1D arrays with longitudes and latitudes."""
    assert lon.ndim == 1, 'longitude coordinate must be one-dimensional'
    assert lat.ndim == 1, 'latitude coordinate must be one-dimensional'

    if interfaces:
        nx, ny = lon.size - 1, lat.size - 1
        lon, lat = interfaces_to_supergrid_1d(lon), interfaces_to_supergrid_1d(lat)
    else:
        nx, ny = lon.size, lat.size
        lon, lat = center_to_supergrid_1d(lon), center_to_supergrid_1d(lat)
    return Domain.create(nx, ny, nz, lon=lon, lat=lat[:, numpy.newaxis], spherical=True, **kwargs)

def create_spherical_at_resolution(minlon: float, maxlon: float, minlat: float, maxlat: float, resolution: float, nz: int, **kwargs) -> 'Domain':
    """Create spherical domain encompassing the specified longitude range and latitude range and desired resolution in m."""
    assert maxlon > minlon, 'Maximum longitude %s must be greater than minimum longitude %s' % (maxlon, minlon)
    assert maxlat > minlat, 'Maximum latitude %s must be greater than minimum latitude %s' % (maxlat, minlat)
    assert resolution > 0, 'Desired resolution must be greater than 0, but is %s m' % (resolution,)
    dlat = resolution / (deg2rad * rearth)
    minabslat = min(abs(minlat), abs(maxlat))
    dlon = resolution / (deg2rad * rearth) / numpy.cos(deg2rad * minabslat)
    nx = int(numpy.ceil((maxlon - minlon) / dlon)) + 1
    ny = int(numpy.ceil((maxlat - minlat) / dlat)) + 1
    return create_spherical(numpy.linspace(minlon, maxlon, nx), numpy.linspace(minlat, maxlat, ny), nz=nz, interfaces=True, **kwargs)

def load(path: str, nz: int, runtype: int):
    """Load domain from file.
    
    Args:
        path: NetCDF file to load from
        nz: number of vertical layers
        runtype: run type
    """
    vars = {}
    with netCDF4.Dataset(path) as nc:
        for name in ('lon', 'lat', 'x', 'y', 'H', 'mask', 'z0b_min', 'cor'):
            if name in nc.variables:
                vars[name] = nc.variables[name][...]
    spherical = 'lon' in vars
    ny, nx = vars['lon' if spherical else 'x'].shape
    nx = (nx - 1) // 2
    ny = (ny - 1) // 2
    return Domain.create(nx, ny, nz, spherical=spherical, **vars)

class RiverTracer(core.Array):
    __slots__ = ('_follow',)
    def __init__(self, grid, river_name: str, tracer_name: str, value: numpy.ndarray, follow: numpy.ndarray, **kwargs):
        super().__init__(grid=grid, name=tracer_name + '_in_river_' + river_name, long_name='%s in river %s' % (tracer_name, river_name), **kwargs)
        self.wrap_ndarray(value)
        self._follow = follow

    @property
    def follow_target_cell(self) -> bool:
        return self._follow

    @follow_target_cell.setter
    def follow_target_cell(self, value: bool):
        self._follow[...] = value

class River:
    def __init__(self, name: str, i: int, j: int, zl: Optional[float]=None, zu: Optional[float]=None, x: Optional[float]=None, y: Optional[float]=None):
        self.name = name
        self.i = i
        self.j = j
        self.x = x
        self.y = y
        self.zl = zl
        self.zu = zu
        self.active = True
        self._tracers: Mapping[str, RiverTracer] = {}

    def locate(self, grid: Grid):
        if self.x is not None:
            ind = grid.nearest_point(self.x, self.y, mask=1, include_halos=True)
            if ind is None:
                grid.domain.logger.info('River %s at x=%s, y=%s not present in this subdomain' % (self.name, self.x, self.y))
                if grid.domain is grid.domain.glob:
                    raise Exception('River %s is located at x=%s, y=%s, which does not fall within the global model domain' % (self.name, self.x, self.y))
                self.active = False
            else:
                self.j, self.i = ind
                grid.domain.logger.info('River %s at x=%s, y=%s is located at i=%i, j=%i in this subdomain' % (self.name, self.x, self.y, self.i, self.j))
        return self.active

    def initialize(self, grid: Grid, flow: numpy.ndarray):
        assert self.active
        self.flow = core.Array(grid=grid, name='river_' + self.name + '_flow', units='m3 s-1', long_name='inflow from %s' % self.name)
        self.flow.wrap_ndarray(flow)

    def __getitem__(self, key) -> RiverTracer:
        return self._tracers[key]

    def __len__(self):
        return len(self._tracers)

    def __iter__(self):
        return iter(self._tracers)

class Rivers(Mapping[str, River]):
    def __init__(self, grid: Grid):
        self.grid = grid
        self._rivers: List[River] = []
        self._tracers = []
        self._frozen = False

    def add_by_index(self, name: str, i: int, j: int, **kwargs):
        """Add a river at a location specified by the indices of a tracer point"""
        assert not self._frozen, 'The river collection has already been initialized and can no longer be modified.'

        i_loc = i - self.grid.domain.tiling.xoffset + self.grid.domain.halox
        j_loc = j - self.grid.domain.tiling.yoffset + self.grid.domain.haloy

        if self.grid.domain.glob is not None and self.grid.domain.glob is not self.grid.domain:
            self.grid.domain.glob.rivers.add_by_index(name, i, j, **kwargs)

        if i_loc >= 0 and j_loc >= 0 and i_loc < self.grid.nx_ and j_loc < self.grid.ny_:
            river = River(name, i_loc, j_loc, **kwargs)
            self._rivers.append(river)
            return river

    def add_by_location(self, name: str, x: float, y: float, **kwargs):
        """Add a river at a location specified by the nearest coordinates (longitude and latitude on a spherical grid)"""
        if self.grid.domain.glob is not None and self.grid.domain.glob is not self.grid.domain:
            self.grid.domain.glob.rivers.add_by_location(name, x, y, **kwargs)
        river = River(name, None, None, x=x, y=y, **kwargs)
        self._rivers.append(river)
        return river

    def initialize(self):
        """Freeze the river collection. Drop those outside the current subdomain and verify the remaining ones are on unmasked T points."""
        assert not self._frozen, 'The river collection has already been initialized'
        self._frozen = True
        self._rivers = [river for river in self._rivers if river.locate(self.grid)]
        self.flow = numpy.zeros((len(self._rivers),))
        self.i = numpy.empty((len(self._rivers),), dtype=int)
        self.j = numpy.empty((len(self._rivers),), dtype=int)
        self.iarea = numpy.empty((len(self._rivers),))
        for iriver, river in enumerate(self._rivers):
            if self.grid.mask.all_values[river.j, river.i] != 1:
                raise Exception('River %s is located at i=%i, j=%i, which is not water (it has mask value %i).' % (river.name, river.i, river.j, self.grid.mask.all_values[river.j, river.i]))
            river.initialize(self.grid, self.flow[..., iriver])
            self.i[iriver] = river.i
            self.j[iriver] = river.j
            self.iarea[iriver] = self.grid.iarea.all_values[river.j, river.i]

    def __getitem__(self, key) -> River:
        for river in self._rivers:
            if key == river.name:
                return river
        raise KeyError()

    def __len__(self) -> int:
        return len(self._rivers)

    def __iter__(self):
        return map(operator.attrgetter('name'), self._rivers)

class OpenBoundary:
    def __init__(self, side: int, l: int, mstart: int, mstop: int, mstart_: int, mstop_: int, type_2d: int, type_3d: int):
        self.side = side
        self.l = l
        self.mstart = mstart
        self.mstop = mstop
        self.mstart_ = mstart_
        self.mstop_ = mstop_
        self.type_2d = type_2d
        self.type_3d = type_3d

class OpenBoundaries(collections.Mapping):
    __slots__ = ('domain', 'np', 'np_glob', 'i', 'j', 'i_glob', 'j_glob', 'z', 'u', 'v', 'lon', 'lat', 'zc', 'zf', 'local_to_global', '_boundaries', '_frozen')

    def __init__(self, domain: 'Domain'):
        self.domain = domain
        self._boundaries: List[OpenBoundary] = []
        self._frozen = False

    def add_by_index(self, side: int, l: int, mstart: int, mstop: int, type_2d: int, type_3d: int):
        """Note that l, mstart, mstop are 0-based indices of a T point in the global domain.
        mstop indicates the upper limit of the boundary - it is the first index that is EXcluded."""
        assert not self._frozen, 'The open boundary collection has already been initialized'
        # NB below we convert to indices in the T grid of the current subdomain INCLUDING halos
        # We also limit the indices to the range valid for the current subdomain.
        xoffset, yoffset = self.domain.tiling.xoffset - self.domain.halox, self.domain.tiling.yoffset - self.domain.haloy
        if side in (WEST, EAST):
            l_offset, m_offset, l_max, m_max = xoffset, yoffset, self.domain.T.nx_, self.domain.T.ny_
        else:
            l_offset, m_offset, l_max, m_max = yoffset, xoffset, self.domain.T.ny_, self.domain.T.nx_
        l_loc = l - l_offset
        mstart_loc_ = mstart - m_offset
        mstop_loc_ = mstop - m_offset
        mstart_loc = min(max(0, mstart_loc_), m_max)
        mstop_loc = min(max(0, mstop_loc_), m_max)
        if l_loc < 0 or l_loc >= l_max or mstop_loc <= mstart_loc:
            # Boundary lies completely outside current subdomain. Record it anyway, so we can later set up a
            # global -> local map of open boundary points
            l_loc, mstart_loc, mstop_loc = None, None, None
        self._boundaries.append(OpenBoundary(side, l_loc, mstart_loc, mstop_loc, mstart_loc_, mstop_loc_, type_2d, type_3d))

        if self.domain.glob is not None and self.domain.glob is not self.domain:
            self.domain.glob.open_boundaries.add_by_index(side, l, mstart, mstop, type_2d, type_3d)

    def initialize(self):
        """Freeze the open boundary collection. Drop those outside the current subdomain."""
        assert not self._frozen, 'The open boundary collection has already been initialized'

        HALO = 2
        nbdyp = 0
        nbdyp_glob = 0
        bdyinfo, bdy_i, bdy_j = [], [], []
        side2count = {}
        self.local_to_global = []
        for side in (WEST, NORTH, EAST, SOUTH):
            n = 0
            for boundary in [b for b in self._boundaries if b.side == side]:
                if boundary.l is not None:
                    mskip = boundary.mstart - boundary.mstart_
                    assert mskip >= 0
                    # Note that bdyinfo needs indices into the T grid EXCLUDING halos
                    bdyinfo.append(numpy.array((boundary.l - HALO, boundary.mstart - HALO, boundary.mstop - HALO, boundary.type_2d, boundary.type_3d, nbdyp), dtype=numpy.intc))
                    nbdyp += boundary.mstop - boundary.mstart

                    # In the mask assignment below, mask=3 points are always in between mask=2 points.
                    # This will not be correct if the boundary only partially falls within this subdomain (i.e., it starts outside),
                    # but as this only affects points at the outer edge of the halo zone, it will be solved by the halo exchange of the mask later on.
                    if side in (WEST, EAST):
                        t_mask = self.domain.mask_[1 + 2 * boundary.mstart:2 * boundary.mstop:2, 1 + boundary.l * 2]
                        vel_mask = self.domain.mask_[2 + 2 * boundary.mstart:2 * boundary.mstop:2, 1 + boundary.l * 2]
                        boundary.i = numpy.repeat(boundary.l, boundary.mstop - boundary.mstart)
                        boundary.j = numpy.arange(boundary.mstart, boundary.mstop)
                    else:
                        t_mask = self.domain.mask_[1 + boundary.l * 2, 1 + 2 * boundary.mstart:2 * boundary.mstop:2]
                        vel_mask = self.domain.mask_[1 + boundary.l * 2, 2 + 2 * boundary.mstart:2 * boundary.mstop:2]
                        boundary.i = numpy.arange(boundary.mstart, boundary.mstop)
                        boundary.j = numpy.repeat(boundary.l, boundary.mstop - boundary.mstart)
                    if (t_mask == 0).any():
                        self.domain.logger.error('%i of %i points of this open boundary are on land' % ((t_mask == 0).sum(), boundary.mstop - boundary.mstart))
                        raise Exception()
                    t_mask[...] = 2
                    vel_mask[...] = 3
                    bdy_i.append(boundary.i)
                    bdy_j.append(boundary.j)

                    if self.local_to_global and self.local_to_global[-1][1] == nbdyp_glob + mskip:
                        # attach to previous boundary
                        self.local_to_global[-1][1] += boundary.mstop - boundary.mstart
                    else:
                        # gap; add new slice
                        self.local_to_global.append([nbdyp_glob + mskip, nbdyp_glob + mskip + boundary.mstop - boundary.mstart])
                    n += 1
                else:
                    self._boundaries.remove(boundary)
                nbdyp_glob += boundary.mstop_ - boundary.mstart_
            side2count[side] = n
        self.np = nbdyp
        self.np_glob = nbdyp_glob
        self.i = numpy.empty((0,), dtype=numpy.intc) if self.np == 0 else numpy.concatenate(bdy_i, dtype=numpy.intc)
        self.j = numpy.empty((0,), dtype=numpy.intc) if self.np == 0 else numpy.concatenate(bdy_j, dtype=numpy.intc)
        self.i_glob = self.i - self.domain.halox + self.domain.tiling.xoffset
        self.j_glob = self.j - self.domain.haloy + self.domain.tiling.yoffset
        self.domain.logger.info('%i open boundaries (%i West, %i North, %i East, %i South)' % (len(bdyinfo), side2count[WEST], side2count[NORTH], side2count[EAST], side2count[SOUTH]))
        if self.np > 0:
            if self.np == self.np_glob:
                assert len(self.local_to_global) == 1 and self.local_to_global[0][0] == 0 and self.local_to_global[0][1] == self.np_glob
                self.local_to_global = None
            else:
                self.domain.logger.info('global-to-local open boundary map: %s' % (self.local_to_global,))
            bdyinfo = numpy.stack(bdyinfo, axis=-1)
            self.domain.initialize_open_boundaries(nwb=side2count[WEST], nnb=side2count[NORTH], neb=side2count[EAST], nsb=side2count[SOUTH], nbdyp=self.np, bdy_i=self.i - HALO, bdy_j=self.j - HALO, bdy_info=bdyinfo)

        # Coordinates of open boundary points
        self.zc = self.domain.T.array(z=CENTERS, on_boundary=True)
        self.zf = self.domain.T.array(z=INTERFACES, on_boundary=True)
        if self.domain.lon is not None:
            self.lon = self.domain.T.array(on_boundary=True, fill=self.domain.T.lon.all_values[self.j, self.i])
        if self.domain.lat is not None:
            self.lat = self.domain.T.array(on_boundary=True, fill=self.domain.T.lat.all_values[self.j, self.i])

        # The arrays below are placeholders that will be assigned data (from momemntum/sealevel Fortran modules) when linked to the Simulation
        self.z = core.Array(grid=self.domain.T, name='z_bdy', units='m', long_name='surface elevation at open boundaries')
        self.u = core.Array(grid=self.domain.T, name='u_bdy', long_name='Eastward velocity or transport at open boundaries')
        self.v = core.Array(grid=self.domain.T, name='v_bdy', long_name='Northward velocity or transport open boundaries')

        self._frozen = True

    def __getitem__(self, key) -> OpenBoundary:
        return self._boundaries[key]

    def __len__(self) -> int:
        return len(self._boundaries)

    def __iter__(self):
        return iter(self._boundaries)

class Domain(_pygetm.Domain):
    @staticmethod
    def partition(tiling: parallel.Tiling, nx: int, ny: int, nz: int, global_domain: Optional['Domain'], halo: int=2, has_xy: bool=True, has_lonlat: bool=True, logger: Optional[logging.Logger]=None, **kwargs):
        assert nx == tiling.nx_glob and ny == tiling.ny_glob, 'Extent of global domain (%i, %i) does not match that of tiling (%i, %i).' % (ny, nx, tiling.ny_glob, tiling.nx_glob)
        assert tiling.n == tiling.comm.Get_size(), 'Number of active cores in subdomain decompositon (%i) does not match available number of cores (%i).' % (tiling.n, tiling.comm.Get_size())

        halo = 4   # coordinates are scattered without their halo - Domain object will update halos upon creation
        share = 1  # one X point overlap in both directions between subdomains for variables on the supergrid
        local_slice, _, _, _ = tiling.subdomain2slices(halo_sub=4, halo_glob=4, scale=2, share=1, exclude_halos=False, exclude_global_halos=True)

        coordinates = {'f': 'cor'}
        if has_xy:
            coordinates['x'] = 'x'
            coordinates['y'] = 'y'
        if has_lonlat:
            coordinates['lon'] = 'lon'
            coordinates['lat'] = 'lat'
        for name, att in coordinates.items():
            c = numpy.empty((2 * tiling.ny_sub + 2 * halo + share, 2 * tiling.nx_sub + 2 * halo + share))
            scatterer = parallel.Scatter(tiling, c, halo=halo, share=share, scale=2, fill_value=numpy.nan)
            scatterer(None if global_domain is None else getattr(global_domain, att + '_'))
            assert not numpy.isnan(c[local_slice]).any(), 'Subdomain %s contains NaN after initial scatter'
            kwargs[name] = c

        domain = Domain(tiling.nx_sub, tiling.ny_sub, nz, tiling=tiling, logger=logger, **kwargs)

        halo = 4
        parallel.Scatter(tiling, domain.mask_, halo=halo, share=share, scale=2)(None if global_domain is None else global_domain.mask_)
        parallel.Scatter(tiling, domain.H_, halo=halo, share=share, scale=2)(None if global_domain is None else global_domain.H_)
        parallel.Scatter(tiling, domain.z0b_min_, halo=halo, share=share, scale=2)(None if global_domain is None else global_domain.z0b_min_)
        parallel.Scatter(tiling, domain.z_, halo=halo, share=share, scale=2)(None if global_domain is None else global_domain.z_)

        return domain

    def _exchange_metric(self, data, relative_in_x: bool=False, relative_in_y: bool=False, fill_value=numpy.nan):
        if not self.tiling:
            return

        expected_shape = (1 + 2 * (self.ny + 2 * self.haloy), 1 + 2 * (self.nx + 2 * self.halox))
        assert data.shape == expected_shape, 'Wrong shape: got %s, expected %s.' % (data.shape, expected_shape)

        halo = 2
        superhalo = 2 * halo
        valid_before = numpy.logical_not(numpy.isnan(data))

        # Expand the data array one each side
        data_ext = numpy.full((data.shape[0] + 2, data.shape[1] + 2), fill_value, dtype=data.dtype)
        data_ext[1:-1, 1:-1] = data
        data_ext[             :superhalo,                    : superhalo    ] = data[:superhalo,              : superhalo]
        data_ext[             :superhalo,       superhalo + 1:-superhalo - 1] = data[:superhalo,   superhalo  :-superhalo]
        data_ext[             :superhalo,      -superhalo    :              ] = data[:superhalo,  -superhalo  :          ]
        data_ext[superhalo + 1:-superhalo - 1,               : superhalo    ] = data[ superhalo:  -superhalo, :superhalo]
        data_ext[superhalo + 1:-superhalo - 1, -superhalo    :              ] = data[ superhalo:  -superhalo, -superhalo:]
        data_ext[-superhalo:,                                : superhalo    ] = data[-superhalo:,             : superhalo]
        data_ext[-superhalo:,                   superhalo + 1:-superhalo - 1] = data[-superhalo:,  superhalo  :-superhalo]
        data_ext[-superhalo:,                      -superhalo:              ] = data[-superhalo:, -superhalo  :          ]
        self.tiling.wrap(data_ext, superhalo + 1).update_halos()

        # For values in the halo, compute their difference with the outer boundary of the subdomain we exchanged with (now the innermost halo point).
        # Then use that difference plus the value on our own boundary as values inside the halo.
        # This ensures coordinate variables are monotonically increasing in interior AND halos, even if periodic boundary conditions are used.
        if relative_in_x:
            data_ext[:, :superhalo + 1] += data_ext[:, superhalo + 1:superhalo + 2] - data_ext[:, superhalo:superhalo + 1]
            data_ext[:, -superhalo - 1:] += data_ext[:, -superhalo - 2:-superhalo - 1] - data_ext[:, -superhalo - 1:-superhalo]
        if relative_in_y:
            data_ext[:superhalo + 1, :] += data_ext[superhalo + 1:superhalo + 2, :] - data_ext[superhalo:superhalo + 1, :]
            data_ext[-superhalo - 1:, :] += data_ext[-superhalo - 2:-superhalo - 1, :] - data_ext[-superhalo - 1:-superhalo, :]

        # Since subdomains share the outer boundary, that boundary will be replicated in the outermost interior point and in the innermost halo point
        # We move the outer part of the halos (all but their innermost point) one point inwards to eliminate that overlapping point
        # Where we do not have a subdomain neighbor, we keep the original values.
        if self.tiling.bottomleft  != -1: data[:superhalo,              : superhalo] = data_ext[             :superhalo,                    : superhalo    ]
        if self.tiling.bottom      != -1: data[:superhalo,   superhalo  :-superhalo] = data_ext[             :superhalo,       superhalo + 1:-superhalo - 1]
        if self.tiling.bottomright != -1: data[:superhalo,  -superhalo  :          ] = data_ext[             :superhalo,      -superhalo    :              ]
        if self.tiling.left        != -1: data[ superhalo:  -superhalo, :superhalo]  = data_ext[superhalo + 1:-superhalo - 1,               : superhalo    ]
        if self.tiling.right       != -1: data[ superhalo:  -superhalo, -superhalo:] = data_ext[superhalo + 1:-superhalo - 1, -superhalo    :              ]
        if self.tiling.topleft     != -1: data[-superhalo:,             : superhalo] = data_ext[-superhalo:,                                : superhalo    ]
        if self.tiling.top         != -1: data[-superhalo:,  superhalo  :-superhalo] = data_ext[-superhalo:,                   superhalo + 1:-superhalo - 1]
        if self.tiling.topright    != -1: data[-superhalo:, -superhalo  :          ] = data_ext[-superhalo:,                      -superhalo:              ]

        valid_after = numpy.logical_not(numpy.isnan(data))
        still_ok = numpy.where(valid_before, valid_after, True)
        assert still_ok.all(), 'Rank %i: _exchange_metric corrupted %i values: %s.' % (self.tiling.rank, still_ok.size - still_ok.sum(), still_ok)

    @staticmethod
    def create(nx: int, ny: int, nz: int, runtype: int=1, lon: Optional[numpy.ndarray]=None, lat: Optional[numpy.ndarray]=None, x: Optional[numpy.ndarray]=None, y: Optional[numpy.ndarray]=None, spherical: bool=False, mask: Optional[numpy.ndarray]=1, H: Optional[numpy.ndarray]=None, z0: Optional[numpy.ndarray]=0., f: Optional[numpy.ndarray]=None, tiling: Optional[parallel.Tiling]=None, periodic_x: bool=False, periodic_y: bool=False, z: Optional[numpy.ndarray]=0., logger: Optional[logging.Logger]=None, **kwargs):
        global_domain = None
        logger = logger or parallel.getLogger()
        parlogger = logger.getChild('parallel')

        # Determine subdomain division
        if tiling is None:
            # No tiling provided - autodetect
            mask = numpy.broadcast_to(mask, (1 + 2 * ny, 1 + 2 * nx))
            tiling = parallel.Tiling.autodetect(mask=mask[1::2, 1::2], logger=parlogger, periodic_x=periodic_x, periodic_y=periodic_y)
        elif isinstance(tiling, str):
            # Path to dumped Tiling object provided
            if not os.path.isfile(tiling):
                logger.critical('Cannot find file %s. If tiling is a string, it must be the path to an existing file with a pickled tiling object.' % tiling)
                raise Exception()
            tiling = parallel.Tiling.load(tiling)
        else:
            # Existing tiling object provided - transfer extent of global domain to determine subdomain sizes
            if isinstance(tiling, tuple):
                tiling = parallel.Tiling(nrow=tiling[0], ncol=tiling[1], periodic_x=periodic_x, periodic_y=periodic_y)
            tiling.set_extent(nx, ny)
        tiling.report(parlogger)

        global_tiling = tiling
        if tiling.n > 1:
            # The global tiling object is a simple 1x1 partition
            global_tiling = parallel.Tiling(nrow=1, ncol=1, ncpus=1, periodic_x=periodic_x, periodic_y=periodic_y)
            global_tiling.set_extent(nx, ny)

        # If on master node (possibly only node), create global domain object
        if tiling.rank == 0:
            global_domain = Domain(nx, ny, nz, lon, lat, x, y, spherical, tiling=global_tiling, mask=mask, H=H, z0=z0, f=f, z=z, logger=logger, **kwargs)

        # If there is only one node, return the global domain immediately
        if tiling.n == 1:
            return global_domain

        # Create the subdomain, and (if on root) attach a pointer to the global domain
        subdomain = Domain.partition(tiling, nx, ny, nz, global_domain, runtype=runtype, has_xy=x is not None, has_lonlat=lon is not None, spherical=spherical, logger=logger, **kwargs)
        subdomain.glob = global_domain

        return subdomain

    def __init__(self, nx: int, ny: int, nz: int,
        lon: Optional[numpy.ndarray]=None, lat: Optional[numpy.ndarray]=None, x: Optional[numpy.ndarray]=None, y: Optional[numpy.ndarray]=None,
        spherical: bool=False, mask: Optional[numpy.ndarray]=1, H: Optional[numpy.ndarray]=None, z0: Optional[numpy.ndarray]=0.,
        f: Optional[numpy.ndarray]=None, tiling: Optional[parallel.Tiling]=None, z: Optional[numpy.ndarray]=0.,
        logger: Optional[logging.Logger]=None, Dmin: float=1., Dcrit: float=2.,
        vertical_coordinate_method: VerticalCoordinates=VerticalCoordinates.SIGMA, ddl: float=0., ddu: float=0., Dgamma: float=0., gamma_surf: bool=True,
        **kwargs):
        """Create domain with coordinates, bathymetry, mask defined on the supergrid.

        Args:
            nx: number of tracer points in x direction
            ny: number of tracer points in y direction
            nz: number of vertical layers
            lon: longitude (degreees East)
            lat: latitude (degreees North)
            x: x coordinate (m)
            y: y coordinate (m)
            spherical: grid is spherical (as opposed to Cartesian). If True, at least `lon` and `lat` must be provided. Otherwise at least `x` and `y` must be provided.
            mask: initial mask (0: land, 1: water)
            H: initial distance between bottom depth and some arbitrary depth reference (m, positive if bottom lies below the depth reference). Typically the depth reference is mean sea level.
            z0: initial bottom roughness (m)
            f: Coriolis parameter. By default this is calculated from latitude `lat` if provided.
            tiling: subdomain decomposition
            Dmin: minimum depth (m) for wet points. At this depth, all hydrodynamic terms except the pressure gradient and bottom friction are switched off.
            Dcrit: depth (m) at which tapering of processes (all except pressure gradient and bottom friction) begins.
            vertical_coordinate_method: type of vertical coordinate to use
            ddl: dimensionless factor for zooming towards the bottom (0: no zooming, > 2: strong zooming)
            ddl: dimensionless factor for zooming towards the surface (0: no zooming, > 2: strong zooming)
            Dgamma: depth (m) range over which z-like coordinates should be used
            gamma_surf: use z-like coordinates in surface layer (as opposed to bottom layer)
        """
        assert nx > 0, 'Number of x points is %i but must be > 0' % nx
        assert ny > 0, 'Number of y points is %i but must be > 0' % ny
        assert nz > 0, 'Number of z points is %i but must be > 0' % nz
        assert lat is not None or f is not None, 'Either lat of f must be provided to determine the Coriolis parameter.'

        self.root_logger: logging.Logger = logger if logger is not None else parallel.getLogger()
        self.logger: logging.Logger = self.root_logger.getChild('domain')
        self.field_manager: Optional[output.FieldManager] = None
        self.input_manager: input.InputManager = input.InputManager()    #: input manager responsible for reading from NetCDF and for spatial and temporal interpolation
        self.glob: Optional['Domain'] = self

        self.logger.info('Domain size (T grid): %i x %i (%i cells)' % (nx, ny, nx * ny))

        self.imin, self.imax = 1, nx
        self.jmin, self.jmax = 1, ny
        self.kmin, self.kmax = 1, nz

        super().__init__(self.imin, self.imax, self.jmin, self.jmax, self.kmin, self.kmax)

        halo = 2

        shape = (2 * ny + 1, 2 * nx + 1)
        superhalo = 2 * halo
        shape_ = (shape[0] + 2 * superhalo, shape[1] + 2 * superhalo)

        # Set up subdomain partition information to enable halo exchanges
        if tiling is None:
            tiling = parallel.Tiling(**kwargs)
        self.tiling = tiling

        def setup_metric(source: Optional[numpy.ndarray]=None, optional: bool=False, fill_value=numpy.nan, relative_in_x: bool=False, relative_in_y: bool=False, dtype: numpy.typing.DTypeLike=float, writeable: bool=True) -> Tuple[Optional[numpy.ndarray], Optional[numpy.ndarray]]:
            if optional and source is None:
                return None, None
            data = numpy.full(shape_, fill_value, dtype)
            data_int = data[superhalo:-superhalo, superhalo:-superhalo]
            if source is not None:
                if numpy.shape(source) == data.shape:
                    data[...] = source
                else:
                    try:
                        # First try if data has been provided on the supergrid
                        data_int[...] = source
                    except ValueError:
                        try:
                            # Now try if data has been provided on the X (corners) grid
                            source_on_X = numpy.broadcast_to(source, (ny + 1, nx + 1))
                            interfaces_to_supergrid_2d(source_on_X, out=data_int)
                        except ValueError:
                            raise Exception('Cannot array broadcast to supergrid (%i x %i) or X grid (%i x %i)' % ((ny * 2 + 1, nx * 2 + 1, ny + 1, nx + 1)))
                self._exchange_metric(data, relative_in_x, relative_in_y, fill_value=fill_value)
            data.flags.writeable = data_int.flags.writeable = writeable
            return data_int, data

        # Supergrid metrics (without underscore=interior only, with underscore=including halos)
        self.x, self.x_ = setup_metric(x, optional=True, relative_in_x=True, writeable=False)
        self.y, self.y_ = setup_metric(y, optional=True, relative_in_y=True, writeable=False)
        self.lon, self.lon_ = setup_metric(lon, optional=True, relative_in_x=True, writeable=False)
        self.lat, self.lat_ = setup_metric(lat, optional=True, relative_in_y=True, writeable=False)
        self.H, self.H_ = setup_metric(H)
        self.z, self.z_ = setup_metric(z)       # elevation
        self.z0b_min, self.z0b_min_ = setup_metric(z0)
        self.mask, self.mask_ = setup_metric(mask, dtype=numpy.intc, fill_value=0)

        cor = f if f is not None else 2. * omega * numpy.sin(deg2rad * lat)
        self.cor, self.cor_ = setup_metric(cor, writeable=False)

        # Compute dx, dy from Cartesian or spherical coordinates
        # These have had their halo exchanges as part of setup_metric and are therefore defined inside the halos too.
        self.dx, self.dx_ = setup_metric()
        self.dy, self.dy_ = setup_metric()
        if spherical:
            dlon_ = self.lon_[:, 2:] - self.lon_[:, :-2]
            dlat_ = self.lat_[2:, :] - self.lat_[:-2, :]
            self.dx_[:, 1:-1] = deg2rad * dlon_ * rearth * numpy.cos(deg2rad * self.lat_[:, 1:-1])
            self.dy_[1:-1, :] = deg2rad * dlat_ * rearth
        else:
            self.dx_[:, 1:-1] = self.x_[:, 2:] - self.x_[:, :-2]
            self.dy_[1:-1, :] = self.y_[2:, :] - self.y_[:-2, :]

        # Halo exchange for dx, dy, needed to ensure the outer strips of the halos are valid
        # Those outermost strips could not be computed by central-differencing the coordinates as that would require points outside the domain.
        self._exchange_metric(self.dx_)
        self._exchange_metric(self.dy_)

        self.dx_.flags.writeable = self.dx.flags.writeable = False
        self.dy_.flags.writeable = self.dy.flags.writeable = False

        self.rotation, self.rotation_ = setup_metric()
        def supergrid_rotation(x, y):
            # For each point, draw lines to the nearest neighbor (1/2 a grid cell) on the left, right, top and bottom.
            # Determinethe angle bey
            rotation_left = numpy.arctan2(y[:, 1:-1] - y[:, :-2], x[:, 1:-1] - x[:, :-2])
            rotation_right = numpy.arctan2(y[:, 2:] - y[:, 1:-1], x[:, 2:] - x[:, 1:-1])
            rotation_bot = numpy.arctan2(y[1:-1, :] - y[:-2, :], x[1:-1, :] - x[:-2, :]) - 0.5 * numpy.pi
            rotation_top = numpy.arctan2(y[2:, :] - y[1:-1, :], x[2:, :] - x[1:-1, :]) - 0.5 * numpy.pi
            x_dum = numpy.cos(rotation_left[1:-1,:]) + numpy.cos(rotation_right[1:-1,:]) + numpy.cos(rotation_bot[:,1:-1]) + numpy.cos(rotation_top[:,1:-1])
            y_dum = numpy.sin(rotation_left[1:-1,:]) + numpy.sin(rotation_right[1:-1,:]) + numpy.sin(rotation_bot[:,1:-1]) + numpy.sin(rotation_top[:,1:-1])
            return numpy.arctan2(y_dum, x_dum)
        if self.lon_ is not None and self.lat_ is not None:
            # Proper rotation with respect to true North
            self.rotation_[1:-1,1:-1] = supergrid_rotation(self.lon_, self.lat_)
        else:
            # Rotation with respect to y axis - assumes y axis always point to true North (can be valid on for infinitesimally small domain)
            self.rotation_[1:-1,1:-1] = supergrid_rotation(self.x_, self.y_)
        self._exchange_metric(self.rotation_)
        self.rotation.flags.writeable = False

        self.area, self.area_ = setup_metric(self.dx * self.dy, writeable=False)

        self.spherical = spherical

        # Determine if we have simple coordinates (useful for xarray and plotting in general)
        self.lon_is_1d = self.lon is not None and (self.lon[:1, :] == self.lon[:, :]).all()
        self.lat_is_1d = self.lat is not None and (self.lat[:, :1] == self.lat[:, :]).all()
        self.x_is_1d = self.x is not None and (self.x[:1, :] == self.x[:, :]).all()
        self.y_is_1d = self.y is not None and (self.y[:, :1] == self.y[:, :]).all()

        self.halo = self.halox
        self.shape = (nz, ny + 2 * self.halo, nx + 2 * self.halo)

        # Advection grids (two letters: first for advected quantity, second for advection direction)
        self.UU = Grid(self, _pygetm.UUGRID, ioffset=3, joffset=1)
        self.VV = Grid(self, _pygetm.VVGRID, ioffset=1, joffset=3)
        self.UV = Grid(self, _pygetm.UVGRID, ioffset=2, joffset=2)
        self.VU = Grid(self, _pygetm.VUGRID, ioffset=2, joffset=2)

        # Create grids
        self.U = Grid(self, _pygetm.UGRID, ioffset=2, joffset=1, ugrid=self.UU, vgrid=self.UV)
        self.V = Grid(self, _pygetm.VGRID, ioffset=1, joffset=2, ugrid=self.VU, vgrid=self.VV)
        self.T = Grid(self, _pygetm.TGRID, ioffset=1, joffset=1, ugrid=self.U, vgrid=self.V)
        self.X = Grid(self, _pygetm.XGRID, ioffset=0, joffset=0, overlap=1)

        self.Dmin = Dmin
        self.Dcrit = Dcrit
        self.ddl = ddl
        self.ddu = ddu
        self.Dgamma = Dgamma
        self.gamma_surf = gamma_surf
        self.vertical_coordinate_method = vertical_coordinate_method

        self._initialized = False
        self.open_boundaries = OpenBoundaries(self)
        self.rivers = Rivers(self.T)

    def initialize(self, runtype: int, field_manager: Optional[output.FieldManager]=None):
        """Initialize the domain. This updates the mask in order for it to be consistent across T, U, V, X grids.
        Values for the mask, bathymetry, and bottom roughness are subsequently read-only."""
        assert not self._initialized, 'Domain has already been initialized'
        if self.glob is not None and self.glob is not self:
            self.glob.initialize(runtype)

        # Mask U,V,X points without any valid T neighbor - this mask will be maintained by the domain to be used for e.g. plotting
        tmask = self.mask_[1::2, 1::2]
        self.mask_[2:-2:2, 1::2][numpy.logical_and(tmask[1:, :] == 0, tmask[:-1, :] == 0)] = 0
        self.mask_[1::2, 2:-2:2][numpy.logical_and(tmask[:, 1:] == 0, tmask[:, :-1] == 0)] = 0
        self.mask_[2:-2:2, 2:-2:2][numpy.logical_and(numpy.logical_and(tmask[1:, 1:] == 0, tmask[:-1, 1:] == 0), numpy.logical_and(tmask[1:, :-1] == 0, tmask[:-1, :-1] == 0))] = 0
        self._exchange_metric(self.mask_, fill_value=0)

        if field_manager is not None:
            self.field_manager = field_manager
        self.field_manager = self.field_manager or output.FieldManager()

        self.open_boundaries.initialize()

        # Mask U,V,X points unless all their T neighbors are valid - this mask will be sent to Fortran and determine which points are computed
        mask_ = numpy.array(self.mask_, copy=True)
        tmask = mask_[1::2, 1::2]
        mask_[2:-2:2, 1::2][numpy.logical_or(tmask[1:, :] == 0, tmask[:-1, :] == 0)] = 0
        mask_[1::2, 2:-2:2][numpy.logical_or(tmask[:, 1:] == 0, tmask[:, :-1] == 0)] = 0
        mask_[2:-2:2, 2:-2:2][numpy.logical_or(numpy.logical_or(tmask[1:, 1:] == 0, tmask[:-1, 1:] == 0), numpy.logical_or(tmask[1:, :-1] == 0, tmask[:-1, :-1] == 0))] = 0
        self._exchange_metric(mask_, fill_value=0)
        self.mask_[...] = mask_

        for grid in self.grids.values():
            grid.initialize(self.open_boundaries.np)
        self.UU.mask.all_values.fill(0)
        self.UV.mask.all_values.fill(0)
        self.VU.mask.all_values.fill(0)
        self.VV.mask.all_values.fill(0)
        self.UU.mask.all_values[:, :-1][numpy.logical_and(self.U.mask.all_values[:, :-1], self.U.mask.all_values[:,1:])] = 1
        self.UV.mask.all_values[:-1, :][numpy.logical_and(self.U.mask.all_values[:-1, :], self.U.mask.all_values[1:,:])] = 1
        self.VU.mask.all_values[:, :-1][numpy.logical_and(self.V.mask.all_values[:, :-1], self.V.mask.all_values[:,1:])] = 1
        self.VV.mask.all_values[:-1, :][numpy.logical_and(self.V.mask.all_values[:-1, :], self.V.mask.all_values[1:,:])] = 1

        self.logger.info('Number of unmasked points excluding halos: %i on T grid, %i on U grid, %i on V grid, %i on X grid' % ((self.T.mask.values > 0).sum(), (self.U.mask.values > 0).sum(), (self.V.mask.values > 0).sum(), (self.X.mask.values > 0).sum()))

        self.H_.flags.writeable = self.H.flags.writeable = False
        self.z0b_min_.flags.writeable = self.z0b_min.flags.writeable = False
        self.mask_.flags.writeable = self.mask.flags.writeable = False
        self.z_.flags.writeable = self.z.flags.writeable = False

        super().initialize(runtype, Dmin=self.Dmin, method_vertical_coordinates=self.vertical_coordinate_method, ddl=self.ddl, ddu=self.ddu, Dgamma=self.Dgamma, gamma_surf=self.gamma_surf)

        self.rivers.initialize()

        # Water depth and thicknesses on T grid that lag 1/2 time step behind tracer (i.e., they are in sync with U,V,X grids)
        self.D_T_half = self.T.array(fill=numpy.nan)
        self.h_T_half = self.T.array(fill=numpy.nan, z=CENTERS)
        self.depth = self.T.array(z=CENTERS, name='pres', units='dbar', long_name='pressure', fabm_standard_name='depth', fill_value=FILL_VALUE)

        self._initialized = True

    def set_bathymetry(self, depth, scale_factor=None, periodic_lon: bool=False):
        """Set bathymetric depth on supergrid. The bathymetric depth is the distance between some arbitrary depth reference
        (often mean sea level) and the bottom, positive for greater depth.
        
        Args:
            scale_factor: apply scale factor to provided depths
            periodic_lon: depth source spans entire globe in longitude
        """
        assert not self._initialized, 'set_bathymetry cannot be called after the domain has been initialized.'
        if not isinstance(depth, xarray.DataArray):
            # Depth is provided as raw data and therefore must be already on the supergrid
            self.H[...] = depth
        else:
            # Depth is provided as xarray object that includes coordinates (we require CF compliant longitude, latitude)
            # Interpolate to target grid.
            depth = input.limit_region(depth, self.lon.min(), self.lon.max(), self.lat.min(), self.lat.max(), periodic_lon=periodic_lon)
            depth = input.horizontal_interpolation(depth, self.lon, self.lat)
            self.H[...] = depth.values
        if scale_factor is not None:
            self.H *= scale_factor

    def mask_shallow(self, minimum_depth: float):
        self.mask[self.H < minimum_depth] = 0

    def limit_velocity_depth(self, critical_depth: Optional[float]=None):
        """Decrease bathymetric depth of velocity (U, V) points to the minimum of the bathymetric depth of both neighboring T points,
        wherever one of these two points is shallower than the specified critical depth.
        
        Args:
            critical_depth: neighbor depth at which the limiting starts. If neighbor are shallower than this value,
                the depth of velocity points is restricted. If not provided, ``self.Dcrit`` is used.
        """
        assert not self._initialized, 'limit_velocity_depth cannot be called after the domain has been initialized.'
        if critical_depth is None:
            critical_depth = self.Dcrit
        tdepth = self.H_[1::2, 1::2]
        Vchange = numpy.logical_or(tdepth[1:, :] <= critical_depth, tdepth[:-1, :] <= critical_depth)
        self.H_[2:-2:2, 1::2][Vchange] = numpy.minimum(tdepth[1:, :], tdepth[:-1, :])[Vchange]
        Uchange = numpy.logical_or(tdepth[:, 1:] <= critical_depth, tdepth[:, :-1] <= critical_depth)
        self.H_[1::2, 2:-2:2][Uchange] = numpy.minimum(tdepth[:, 1:], tdepth[:, :-1])[Uchange]
        self.logger.info('limit_velocity_depth has decreased depth in %i U points (%i currently unmasked), %i V points (%i currently unmasked).' % (Uchange.sum(), Uchange.sum(where=self.mask_[1::2, 2:-2:2] != 0), Vchange.sum(), Vchange.sum(where=self.mask_[2:-2:2, 1::2] != 0)))

    def mask_rectangle(self, xmin: Optional[float]=None, xmax: Optional[float]=None, ymin: Optional[float]=None, ymax: Optional[float]=None, value: int=0):
        assert not self._initialized, 'adjust_mask cannot be called after the domain has been initialized.'
        selected = numpy.ones(self.mask.shape, dtype=bool)
        x, y = (self.lon, self.lat) if self.spherical else (self.x, self.y)
        if xmin is not None: selected = numpy.logical_and(selected, x >= xmin)
        if xmax is not None: selected = numpy.logical_and(selected, x <= xmax)
        if ymin is not None: selected = numpy.logical_and(selected, y >= ymin)
        if ymax is not None: selected = numpy.logical_and(selected, y <= ymax)
        self.mask[selected] = value

    def plot(self, fig=None, show_H: bool=True, show_mesh: bool=True, show_rivers: bool=True, editable: bool=False, sub: bool=False):
        """Plot the domain, optionally including bathymetric depth, mesh and river positions.
        
        Args:
            fig: :class:`matplotlib.figure.Figure` instance to plot to. If not provided, a new figure is created.
            show_H: show bathymetry as color map
            show_mesh: show model grid
            show_rivers: show rivers with position and name
            editable: allow interactive selection of rectangular regions in the domain plot that are subsequently masked out
            sub: plot the subdomain, not the global domain

        Returns:
            :class:`matplotlib.figure.Figure` instance for processes with rank 0 or if ``sub`` is ``True``, otherwise ``None``
        """
        import matplotlib.pyplot
        import matplotlib.collections
        import matplotlib.widgets
        if self.glob is not self and not sub:
            # We need to plot the global domain; not the current subdomain.
            # If we are the root, divert the plot command to the global domain. Otherwise just ignore this and return.
            if self.glob:
                return self.glob.plot(fig, show_H, show_mesh, show_rivers, editable)
            return

        if fig is None:
            fig, ax = matplotlib.pyplot.subplots(figsize=(0.15 * self.nx, 0.15 * self.ny))
        else:
            ax = fig.gca()

        x, y = (self.lon, self.lat) if self.spherical else (self.x, self.y)

        local_slice, _, _, _ = self.tiling.subdomain2slices(halo_sub=0, halo_glob=4, scale=2, share=1, exclude_global_halos=True)
        if show_H:
            import cmocean
            cm = cmocean.cm.deep
            cm.set_bad('gray')
            c = ax.pcolormesh(x[local_slice], y[local_slice], numpy.ma.array(self.H[local_slice], mask=self.mask[local_slice]==0), alpha=0.5 if show_mesh else 1, shading='auto', cmap=cm)
            #c = ax.contourf(x, y, numpy.ma.array(self.H, mask=self.mask==0), 20, alpha=0.5 if show_mesh else 1)
            cb = fig.colorbar(c)
            cb.set_label('undisturbed water depth (m)')

        if show_rivers:
            for river in self.rivers.values():
                iloc, jloc = 1 + river.i * 2, 1 + river.j * 2
                lon = self.lon_[jloc, iloc]
                lat = self.lat_[jloc, iloc]
                ax.plot([lon], [lat], '.r')
                ax.text(lon, lat, river.name, color='r')

        def plot_mesh(ax, x, y, **kwargs):
            segs1 = numpy.stack((x, y), axis=2)
            segs2 = segs1.transpose(1, 0, 2)
            ax.add_collection(matplotlib.collections.LineCollection(segs1, **kwargs))
            ax.add_collection(matplotlib.collections.LineCollection(segs2, **kwargs))

        if show_mesh:
            plot_mesh(ax, x[::2, ::2], y[::2, ::2], colors='k', linestyle='-', linewidth=.3)
            #ax.pcolor(x[1::2, 1::2], y[1::2, 1::2], numpy.ma.array(x[1::2, 1::2], mask=True), edgecolors='k', linestyles='--', linewidth=.2)
            #pc = ax.pcolormesh(x[1::2, 1::2], y[1::2, 1::2],  numpy.ma.array(x[1::2, 1::2], mask=True), edgecolor='gray', linestyles='--', linewidth=.2)
            ax.plot(x[::2, ::2], y[::2, ::2], '.k', markersize=3.)
            ax.plot(x[1::2, 1::2], y[1::2, 1::2], 'xk', markersize=2.5)
        ax.set_xlabel('longitude (degrees East)' if self.spherical else 'x (m)')
        ax.set_ylabel('latitude (degrees North)' if self.spherical else 'y (m)')
        if not self.spherical:
            ax.axis('equal')
        xmin, xmax = numpy.nanmin(x), numpy.nanmax(x)
        ymin, ymax = numpy.nanmin(y), numpy.nanmax(y)
        xmargin = 0.05 * (xmax - xmin)
        ymargin = 0.05 * (ymax - ymin)
        ax.set_xlim(xmin - xmargin, xmax + xmargin)
        ax.set_ylim(ymin - ymargin, ymax + ymargin)

        def on_select(eclick, erelease):
            xmin, xmax = min(eclick.xdata, erelease.xdata), max(eclick.xdata, erelease.xdata)
            ymin, ymax = min(eclick.ydata, erelease.ydata), max(eclick.ydata, erelease.ydata)
            self.mask_rectangle(xmin, xmax, ymin, ymax)
            c.set_array(numpy.ma.array(self.H, mask=self.mask==0).ravel())
            fig.canvas.draw()
            #self.sel.set_active(False)
            #self.sel = None
            #ax.draw()
            #fig.clf()
            #self.plot(fig=fig, show_mesh=show_mesh)
        if editable:
            self.sel = matplotlib.widgets.RectangleSelector(
                    ax, on_select,
                    useblit=True,
                    button=[1],
                    interactive=False)
        return fig

    def save(self, path: str, full: bool=False, sub: bool=False):
        """Save grid to a NetCDF file that can be interpreted by :func:`load`.
        
        Args:
            path: NetCDF file to save to
        """
        if self.glob is not self and not sub:
            # We need to save the global domain; not the current subdomain.
            # If we are the root, divert the plot command to the global domain. Otherwise just ignore this and return.
            if self.glob:
                self.glob.save(path, full)
            return

        with netCDF4.Dataset(path, 'w') as nc:
            def create(name, units, long_name, values, coordinates: str, dimensions=('y', 'x')):
                fill_value = None
                if numpy.ma.getmask(values) is not numpy.ma.nomask:
                    fill_value = values.fill_value
                ncvar = nc.createVariable(name, values.dtype, dimensions, fill_value=fill_value)
                ncvar.units = units
                ncvar.long_name = long_name
                #ncvar.coordinates = coordinates
                ncvar[...] = values

            def create_var(name, units, long_name, values, values_):
                if values is None:
                    return
                create(name, units, long_name, values, coordinates='lon lat' if self.spherical else 'x y')
                if full: create(name + '_', units, long_name, values_, dimensions=('y_', 'x_'), coordinates='lon_ lat_' if self.spherical else 'x_ y_')

            nc.createDimension('x', self.H.shape[1])
            nc.createDimension('y', self.H.shape[0])
            if full:
                nc.createDimension('x_', self.H_.shape[1])
                nc.createDimension('y_', self.H_.shape[0])
                create_var('dx', 'm', 'dx', self.dx, self.dx_)
                create_var('dy', 'm', 'dy', self.dy, self.dy_)
                create_var('area', 'm2', 'area', self.dx * self.dy, self.dx_ * self.dy_)
            create_var('lat', 'degrees_north', 'latitude', self.lat, self.lat_)
            create_var('lon', 'degrees_east', 'longitude', self.lon, self.lon_)
            create_var('x', 'm', 'x', self.x, self.x_)
            create_var('y', 'm', 'y', self.y, self.y_)
            create_var('H', 'm', 'undisturbed water depth', self.H, self.H_)
            create_var('mask', '', 'mask', self.mask, self.mask_)
            create_var('z0b_min', 'm', 'bottom roughness', self.z0b_min, self.z0b_min_)
            create_var('cor', '', 'Coriolis parameter', self.cor, self.cor_)

    def save_grids(self, path: str):
        with netCDF4.Dataset(path, 'w') as nc:
            self.T.add_to_netcdf(nc)
            self.U.add_to_netcdf(nc, postfix='u')
            self.V.add_to_netcdf(nc, postfix='v')
            self.X.add_to_netcdf(nc, postfix='x')

    def contains(self, x: float, y: float, include_halos: bool=False) -> bool:
        """Determine whether the domain contains the specified point.
        
        Args:
            x: native x coordinate (longitude for spherical grids, Cartesian coordinate in m otherwise)
            y: native y coordinate (latitude for spherical grids, Cartesian coordinate in m otherwise)
            include_halos: whether to also search the halos

        Returns:
            True if the point falls within the domain, False otherwise
        """
        local_slice, _, _, _ = self.tiling.subdomain2slices(halo_sub=4, halo_glob=4, scale=2, share=1, exclude_halos=not include_halos, exclude_global_halos=True)
        allx, ally = (self.lon_, self.lat_) if self.spherical else (self.x_, self.y_)
        allx, ally = allx[local_slice], ally[local_slice]
        ny, nx = allx.shape

        # Determine whether point falls within current subdomain
        # based on https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
        x_bnd = numpy.concatenate((allx[0, :-1], allx[:-1, -1], allx[-1, nx-1:0:-1], allx[ny-1:0:-1, 0]))
        y_bnd = numpy.concatenate((ally[0, :-1], ally[:-1, -1], ally[-1, nx-1:0:-1], ally[ny-1:0:-1, 0]))
        assert not numpy.isnan(x_bnd).any(), 'Invalid x boundary: %s.' % (x_bnd,)
        assert not numpy.isnan(y_bnd).any(), 'Invalid y boundary: %s.' % (y_bnd,)
        assert x_bnd.size == 2 * ny + 2 * nx - 4
        inside = False
        for i, (vertxi, vertyi) in enumerate(zip(x_bnd, y_bnd)):
            vertxj, vertyj = x_bnd[i - 1], y_bnd[i - 1]
            if (vertyi > y) != (vertyj > y) and x < (vertxj - vertxi) * (y - vertyi) / (vertyj - vertyi) + vertxi:
                inside = not inside
        return inside

    def update_depth(self, _3d: bool=False):
        """Use old and new surface elevation on T grid to update elevations on U, V, X grids
        and subsequently update total water depth ``D`` on all grids.

        Args:
            _3d: update elevations of the macrotimestep (``zin``) rather than elevations of the microtimestep (``z``).
                This first synchronizes the elevations of the macrotimestep on the T grid (``self.T.zin``)
                with those of the microtimestep (``self.T.z``). It also updates layer thicknesses ``hn``,
                layer center depths ``zc`` and interface depths ``zf`` on all grids.

        This routine will ensure values are up to date in the domain interior and in the halos,
        but that this requires that ``self.T.z`` (and old elevations ``self.T.zo`` or ``self.T.zio``)
        are already up to date in halos.
        """
        if _3d:
            # Store current elevations as previous elevations (on the 3D time step)
            self.T.zio.all_values[...] = self.T.zin.all_values
            self.U.zio.all_values[...] = self.U.zin.all_values
            self.V.zio.all_values[...] = self.V.zin.all_values
            self.X.zio.all_values[...] = self.X.zin.all_values

            # Synchronize new elevations on the 3D time step to those of the 2D time step that has just completed.
            self.T.zin.all_values[...] = self.T.z.all_values

            z_T, z_U, z_V, z_X, zo_T = self.T.zin, self.U.zin, self.V.zin, self.X.zin, self.T.zio
        else:
            z_T, z_U, z_V, z_X, zo_T = self.T.z, self.U.z, self.V.z, self.X.z, self.T.zo

        # Compute surface elevation on U, V, X grids.
        # These must lag 1/2 a timestep behind the T grid.
        # They are therefore calculated from the average of old and new elevations on the T grid.
        z_T_half = 0.5 * (zo_T + z_T)
        z_T_half.interp(z_U)
        z_T_half.interp(z_V)
        z_T_half.interp(z_X)
        _pygetm.clip_z(z_U, self.Dmin)
        _pygetm.clip_z(z_V, self.Dmin)
        _pygetm.clip_z(z_X, self.Dmin)

        # Halo exchange for elevation on U, V grids, needed because the very last points in the halos
        # (x=-1 for U, y=-1 for V) are not valid after interpolating from the T grid above.
        # These elevations are needed to later compute velocities from transports
        # (by dividing by layer thicknesses, which are computed from elevation)
        # These velocities will be advected, and therefore need to be valid througout the halos.
        # We do not need to halo-exchange elevation on the X grid, since that needs to be be valid
        # at the innermost halo point only, which is ensured by z_T exchange.
        z_U.update_halos(parallel.Neighbor.RIGHT)
        z_V.update_halos(parallel.Neighbor.TOP)

        # Update total water depth D on T, U, V, X grids
        # This also processes the halos; no further halo exchange needed.
        numpy.add(self.T.H.all_values, z_T.all_values, where=self.T.mask.all_values > 0, out=self.T.D.all_values)
        numpy.add(self.U.H.all_values, z_U.all_values, where=self.U.mask.all_values > 0, out=self.U.D.all_values)
        numpy.add(self.V.H.all_values, z_V.all_values, where=self.V.mask.all_values > 0, out=self.V.D.all_values)
        numpy.add(self.X.H.all_values, z_X.all_values, where=self.X.mask.all_values > 0, out=self.X.D.all_values)

        # Update dampening factor (0-1) for shallow water
        _pygetm.alpha(self.U.D, self.Dmin, self.Dcrit, self.U.alpha)
        _pygetm.alpha(self.V.D, self.Dmin, self.Dcrit, self.V.alpha)

        # Update total water depth on advection grids. These must be 1/2 timestep behind the T grid.
        # That's already the case for the X grid, but for the T grid we explicitly compute and use the average of old and new D.
        numpy.add(self.T.H.all_values, z_T_half.all_values, out=self.D_T_half.all_values)
        self.UU.D.all_values[:, :-1] = self.D_T_half.all_values[:, 1:]
        self.VV.D.all_values[:-1, :] = self.D_T_half.all_values[1:, :]
        self.UV.D.all_values[:, :] = self.VU.D.all_values[:, :] = self.X.D.all_values[1:, 1:]

        if _3d:
            # Store previous layer thicknesses
            self.T.ho.all_values[...] = self.T.hn.all_values
            self.U.ho.all_values[...] = self.U.hn.all_values
            self.V.ho.all_values[...] = self.V.hn.all_values
            self.X.ho.all_values[...] = self.X.hn.all_values

            # Update layer thicknesses (hn) on all grids, using bathymetry H and new elevations zin (on the 3D timestep)
            self.do_vertical()

            # Update vertical coordinates, used for e.g., output, internal pressure, vertical interpolation of open boundary forcing of tracers
            for grid in (self.T, self.U, self.V):
                _pygetm.thickness2vertical_coordinates(grid.mask, grid.H, grid.hn, grid.zc, grid.zf)

            # Update thicknesses on advection grids. These must be at time=n+1/2
            # That's already the case for the X grid, but for the T grid (now at t=n+1) we explicitly compute thicknesses at time=n+1/2.
            # Note that UU.hn and VV.hn will miss the x=-1 and y=-1 strips, respectively (the last strip of values within their halos);
            # fortunately these values are not needed for advection.
            self.h_T_half.all_values[...] = 0.5 * (self.T.ho.all_values + self.T.hn.all_values)
            self.UU.hn.all_values[:, :, :-1] = self.h_T_half.all_values[:, :, 1:]
            self.VV.hn.all_values[:, :-1, :] = self.h_T_half.all_values[:, 1:, :]
            self.UV.hn.all_values[:, :, :] = self.VU.hn.all_values[:, :, :] = self.X.hn.all_values[:, 1:, 1:]

            if self.depth.saved:
                # Update pressure (dbar) at layer centers, assuming it is equal to depth in m
                _pygetm.thickness2center_depth(self.T.mask, self.T.hn, self.depth)

            if self.open_boundaries.zc.saved:
                # Update vertical coordinate at open boundary, used to interpolate inputs on z grid to dynamic model depths
                self.open_boundaries.zc.all_values[...] = self.T.zc.all_values[:, self.open_boundaries.j, self.open_boundaries.i].T
