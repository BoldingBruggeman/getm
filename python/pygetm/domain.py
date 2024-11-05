from typing import Mapping, Optional, Tuple, Union
import enum
import functools
import logging

import numpy as np
import numpy.typing as npt
import xarray as xr
import netCDF4

from . import core
from . import parallel
from . import rivers
from . import open_boundaries
from .constants import CoordinateType, GRAVITY


# class VerticalCoordinates(enum.IntEnum):
#     SIGMA = 1
#     #    Z = 2
#     GVC = 3
#     #    HYBRID = 4
#     #    ADAPTIVE = 5


class EdgeTreatment(enum.Enum):
    MISSING = enum.auto()
    CLAMP = enum.auto()
    PERIODIC = enum.auto()
    EXTRAPOLATE = enum.auto()
    EXTRAPOLATE_PERIODIC = enum.auto()


def find_interfaces(c: npt.ArrayLike) -> np.ndarray:
    c_if = np.empty((c.size + 1))
    d = np.diff(c)
    c_if[1:-1] = c[:-1] + 0.5 * d
    c_if[0] = c[0] - 0.5 * d[0]
    c_if[-1] = c[-1] + 0.5 * d[-1]
    return c_if


DEG2RAD = np.pi / 180  # degree to radian conversion
RAD2DEG = 180 / np.pi  # radian to degree conversion
R_EARTH = 6378815.0  # radius of the earth (m)
OMEGA = (
    2.0 * np.pi / 86164.0
)  # rotation rate of the earth (rad/s), 86164 is number of seconds in a sidereal day


def coriolis(lat: npt.ArrayLike) -> np.ndarray:
    """Calculate Coriolis parameter f for the given latitude.

    Args:
        lat: latitude in degrees North

    Returns:
        Coriolis parameter f
    """
    return (2.0 * OMEGA) * np.sin(DEG2RAD * np.asarray(lat, dtype=float))


def centers_to_supergrid_1d(
    data: npt.ArrayLike,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    edges: EdgeTreatment = EdgeTreatment.MISSING,
    missing_value=np.nan,
) -> np.ndarray:
    assert data.ndim == 1, "data must be one-dimensional"
    assert data.size > 1, "data must have at least 2 elements"
    edges = EdgeTreatment(edges)
    data_sup = np.empty((data.size * 2 + 1,), dtype=dtype)
    data_sup[1::2] = data
    data_sup[2:-2:2] = 0.5 * (data[1:] + data[:-1])
    if edges == EdgeTreatment.PERIODIC:
        # Reconstruct first and last interface by averaging very first and last values
        data_sup[0] = data_sup[-1] = 0.5 * (data[0] + data[-1])
    elif edges == EdgeTreatment.EXTRAPOLATE:
        # Reconstruct first and last interface through linear extrapolation,
        # using nearest interior difference
        data_sup[0] = 2 * data_sup[1] - data_sup[2]
        data_sup[-1] = 2 * data_sup[-2] - data_sup[-3]
    elif edges == EdgeTreatment.EXTRAPOLATE_PERIODIC:
        # Reconstruct first and last interface through linear extrapolation,
        # using oppposite interior difference
        data_sup[0] = data_sup[1] + (data_sup[-3] - data_sup[-2])
        data_sup[-1] = data_sup[-2] + (data_sup[2] - data_sup[1])
    elif edges == EdgeTreatment.CLAMP:
        data_sup[0] = data_sup[1]
        data_sup[-1] = data_sup[-2]
    else:
        data_sup[0] = data_sup[-1] = missing_value
    return data_sup


def expand_2d(
    source: np.ndarray,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    edges_x: EdgeTreatment = EdgeTreatment.MISSING,
    edges_y: EdgeTreatment = EdgeTreatment.MISSING,
    missing_value=np.nan,
) -> np.ndarray:
    edges_x = EdgeTreatment(edges_x)
    edges_y = EdgeTreatment(edges_y)

    # Create an array to hold data at centers (T points),
    # with strips of size 1 on all sides to support interpolation to interfaces
    out_shape = source.shape[:-2] + (source.shape[-2] + 2, source.shape[-1] + 2)
    out = np.empty_like(source, shape=out_shape, dtype=dtype)
    out[..., 1:-1, 1:-1] = source

    if edges_x == EdgeTreatment.EXTRAPOLATE:
        out[..., 0] = 2 * out[..., 1] - out[..., 2]
        out[..., -1] = 2 * out[..., -2] - out[..., -3]
    elif edges_x == EdgeTreatment.EXTRAPOLATE_PERIODIC:
        out[..., 0] = out[..., 1] + (out[..., -3] - out[..., -2])
        out[..., -1] = out[..., -2] + (out[..., 2] - out[..., 1])
    elif edges_x == EdgeTreatment.CLAMP:
        out[..., 0] = out[..., 1]
        out[..., -1] = out[..., -2]
    elif edges_x == EdgeTreatment.PERIODIC:
        out[..., 0] = out[..., -2]
        out[..., -1] = out[..., 1]
    elif edges_x == EdgeTreatment.MISSING:
        out[..., 0] = missing_value
        out[..., -1] = missing_value

    if edges_y == EdgeTreatment.EXTRAPOLATE:
        out[..., 0, :] = 2 * out[..., 1, :] - out[..., 2, :]
        out[..., -1, :] = 2 * out[..., -2, :] - out[..., -3, :]
    elif edges_y == EdgeTreatment.EXTRAPOLATE_PERIODIC:
        out[..., 0, :] = out[..., 1, :] + (out[..., -3, :] - out[..., -2, :])
        out[..., -1, :] = out[..., -2, :] + (out[..., 2, :] - out[..., 1, :])
    elif edges_y == EdgeTreatment.CLAMP:
        out[..., 0, :] = out[..., 1, :]
        out[..., -1, :] = out[..., -2, :]
    elif edges_y == EdgeTreatment.PERIODIC:
        out[..., 0, :] = out[..., -2, :]
        out[..., -1, :] = out[..., 1, :]
    elif edges_y == EdgeTreatment.MISSING:
        out[..., 0, :] = missing_value
        out[..., -1, :] = missing_value

    return out


def centers_to_supergrid_2d(
    source: npt.ArrayLike,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    edges_x: EdgeTreatment = EdgeTreatment.MISSING,
    edges_y: EdgeTreatment = EdgeTreatment.MISSING,
    missing_value=np.nan,
) -> np.ndarray:
    source = np.asanyarray(source)
    ny, nx = source.shape

    source_ex = expand_2d(
        source,
        dtype=dtype,
        edges_x=edges_x,
        edges_y=edges_y,
        missing_value=missing_value,
    )

    out_shape = source.shape[:-2] + (ny * 2 + 1, nx * 2 + 1)
    out = np.empty_like(source_ex, shape=out_shape)

    if_ip_shape = (4,) + source.shape[:-2] + (ny + 1, nx + 1)
    data_if_ip = np.empty_like(source_ex, shape=if_ip_shape)
    data_if_ip[0, ...] = source_ex[..., :-1, :-1]
    data_if_ip[1, ...] = source_ex[..., 1:, :-1]
    data_if_ip[2, ...] = source_ex[..., :-1, 1:]
    data_if_ip[3, ...] = source_ex[..., 1:, 1:]
    out[..., 1::2, 1::2] = source_ex[1:-1, 1:-1]  # T points
    out[..., ::2, ::2] = data_if_ip.mean(axis=0)  # X points

    data_if_ip[0, ..., :-1] = source_ex[..., :-1, 1:-1]
    data_if_ip[1, ..., :-1] = source_ex[..., 1:, 1:-1]
    out[..., ::2, 1::2] = data_if_ip[:2, ..., :-1].mean(axis=0)  # V points

    data_if_ip[0, ..., :-1, :] = source_ex[..., 1:-1, :-1]
    data_if_ip[1, ..., :-1, :] = source_ex[..., 1:-1, 1:]
    out[..., 1::2, ::2] = data_if_ip[:2, ..., :-1, :].mean(axis=0)  # U points

    return np.ma.filled(out, missing_value)


def interfaces_to_supergrid_1d(
    data: npt.ArrayLike,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    assert data.ndim == 1, "data must be one-dimensional"
    assert data.size > 1, "data must have at least 2 elements"
    if out is None:
        out = np.empty((data.size * 2 - 1,), dtype=dtype)
    out[0::2] = data
    out[1::2] = 0.5 * (data[1:] + data[:-1])
    return out


def interfaces_to_supergrid_2d(
    data: npt.ArrayLike,
    *,
    dtype: Optional[npt.DTypeLike] = None,
    out: Optional[np.ndarray] = None,
) -> np.ndarray:
    assert data.ndim == 2, "data must be two-dimensional"
    assert data.shape[0] > 1 and data.shape[1] > 1, "dimensions must have length >= 2"
    if out is None:
        out = np.empty((data.shape[0] * 2 - 1, data.shape[1] * 2 - 1), dtype=dtype)
    out[0::2, 0::2] = data
    out[1::2, 0::2] = 0.5 * (data[:-1, :] + data[1:, :])
    out[0::2, 1::2] = 0.5 * (data[:, :-1] + data[:, 1:])
    out[1::2, 1::2] = 0.25 * (
        data[:-1, :-1] + data[:-1, 1:] + data[1:, :-1] + data[1:, 1:]
    )
    return out


def create_cartesian(
    x: npt.ArrayLike, y: npt.ArrayLike, *, interfaces=False, **kwargs
) -> "Domain":
    """Create Cartesian domain from 1D arrays with x coordinates and  y coordinates.

    Args:
        x: array with x coordinates (1d or 2d)
            (at cell interfaces if `interfaces=True`, else at cell centers)
        y: array with y coordinates (1d or 2d)
            (at cell interfaces if `interfaces=True`, else at cell centers)
        interfaces: coordinates are given at cell interfaces, rather than cell centers.
        **kwargs: additional arguments passed to :class:`Domain`
    """
    x = np.asarray(x)
    y = np.asarray(y)
    if y.ndim == 1:
        y = y[:, np.newaxis]
    if interfaces:
        nx, ny = x.shape[-1] - 1, y.shape[0] - 1
    else:
        nx, ny = x.shape[-1], y.shape[0]
    return Domain(nx, ny, x=x, y=y, coordinate_type=CoordinateType.XY, **kwargs)


def create_spherical(
    lon: npt.ArrayLike, lat: npt.ArrayLike, *, interfaces=False, **kwargs
) -> "Domain":
    """Create spherical domain from 1D arrays with longitudes and latitudes.

    Args:
        lon: array with longitude coordinates (1d or 2d)
            (at cell interfaces if `interfaces=True`, else at cell centers)
        lat: array with latitude coordinates (1d or 2d)
            (at cell interfaces if `interfaces=True`, else at cell centers)
        interfaces: coordinates are given at cell interfaces, rather than cell centers.
        **kwargs: additional arguments passed to :class:`Domain`
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    if lat.ndim == 1:
        lat = lat[:, np.newaxis]

    if interfaces:
        nx, ny = lon.shape[-1] - 1, lat.shape[0] - 1
    else:
        nx, ny = lon.shape[-1], lat.shape[0]
    return Domain(nx, ny, lon=lon, lat=lat, **kwargs)


def create_spherical_at_resolution(
    minlon: float,
    maxlon: float,
    minlat: float,
    maxlat: float,
    resolution: float,
    **kwargs,
) -> "Domain":
    """Create spherical domain encompassing the specified longitude range and latitude
    range and desired resolution in m.

    Args:
        minlon: minimum longitude
        maxlon: maximum longitude
        minlat: minimum latitude
        maxlat: maximum latitude
        resolution: maximum grid cell length and width (m)
        **kwargs: additional arguments passed to :class:`Domain`
    """
    if maxlon <= minlon:
        raise Exception(
            f"Maximum longitude {maxlon} must exceed minimum longitude {minlon}"
        )
    if maxlat <= minlat:
        raise Exception(
            f"Maximum latitude {maxlat} must exceed minimum latitude {minlat}"
        )
    if resolution <= 0.0:
        raise Exception(f"Desired resolution must exceed 0, but is {resolution} m")
    dlat = resolution / (DEG2RAD * R_EARTH)
    minabslat = min(abs(minlat), abs(maxlat))
    dlon = resolution / (DEG2RAD * R_EARTH) / np.cos(DEG2RAD * minabslat)
    nx = int(np.ceil((maxlon - minlon) / dlon)) + 1
    ny = int(np.ceil((maxlat - minlat) / dlat)) + 1
    return create_spherical(
        np.linspace(minlon, maxlon, nx),
        np.linspace(minlat, maxlat, ny),
        interfaces=True,
        **kwargs,
    )


class DomainArray:
    def __init__(self, writeable: bool = True, **kwargs):
        self.writable = writeable
        self.kwargs = kwargs

    def __set_name__(self, owner, name):
        self.private_name = f"_{name}"

    def __get__(self, domain: "Domain", objtype=None) -> Optional[np.ndarray]:
        values = getattr(domain, self.private_name, None)
        if values is not None:
            if values.flags.writeable and not self.writable:
                values = values.view()
                values.flags.writeable = False
            elif not values.flags.writeable and self.writable:
                values = values.copy()
                setattr(domain, self.private_name, values)
        return values

    def __set__(self, domain: "Domain", values):
        assert self.writable
        values = getattr(domain, self.private_name, None)
        values[...] = domain._map_array(values, **self.kwargs)


def calculate_and_bcast(method):
    @functools.wraps(method)
    def wrapper(self: "Domain", *args, **kwargs):
        if self.comm.rank == 0:
            result = method(self, *args, **kwargs)
        else:
            result = None
        return self.comm.bcast(result)

    return wrapper


def _rotation(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # For each point, draw lines to the nearest neighbor (1/2 a grid cell) on
    # the left, right, top and bottom.
    rot_left = np.arctan2(y[:, 1:-1] - y[:, :-2], x[:, 1:-1] - x[:, :-2])
    rot_right = np.arctan2(y[:, 2:] - y[:, 1:-1], x[:, 2:] - x[:, 1:-1])
    rot_bot = np.arctan2(y[1:-1, :] - y[:-2, :], x[1:-1, :] - x[:-2, :]) - 0.5 * np.pi
    rot_top = np.arctan2(y[2:, :] - y[1:-1, :], x[2:, :] - x[1:-1, :]) - 0.5 * np.pi
    x_dum = (
        np.cos(rot_left[1:-1, :])
        + np.cos(rot_right[1:-1, :])
        + np.cos(rot_bot[:, 1:-1])
        + np.cos(rot_top[:, 1:-1])
    )
    y_dum = (
        np.sin(rot_left[1:-1, :])
        + np.sin(rot_right[1:-1, :])
        + np.sin(rot_bot[:, 1:-1])
        + np.sin(rot_top[:, 1:-1])
    )
    return np.arctan2(y_dum, x_dum)


class Domain:
    # Grid metrics frozen at initialization
    x = DomainArray(writeable=False, xcoordinate=True)
    y = DomainArray(writeable=False, ycoordinate=True)
    lon = DomainArray(writeable=False, xcoordinate=True)
    lat = DomainArray(writeable=False, ycoordinate=True)
    f = DomainArray(writeable=False)
    dx = DomainArray(writeable=False)
    dy = DomainArray(writeable=False)
    rotation = DomainArray(writeable=False)
    area = DomainArray(writeable=False)

    # Grid metrics that can be manipulated after the domain is created
    mask = DomainArray()
    H = DomainArray()
    z0 = DomainArray()

    def __init__(
        self,
        nx: int,
        ny: int,
        *,
        lon: Optional[np.ndarray] = None,
        lat: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        y: Optional[np.ndarray] = None,
        coordinate_type: Optional[CoordinateType] = None,
        mask: Optional[np.ndarray] = 1,
        H: Optional[np.ndarray] = None,
        z0: Optional[np.ndarray] = 0.0,
        f: Optional[np.ndarray] = None,
        periodic_x: bool = False,
        periodic_y: bool = False,
        comm: parallel.MPI.Comm = parallel.MPI.COMM_WORLD,
        logger: Optional[logging.Logger] = None,
    ):
        """Create new domain.

        Args:
            nx: number of tracer points in x-direction
            ny: number of tracer points in y-direction
            lon: longitude (degrees East)
            lat: latitude (degrees North)
            x: x coordinate (m)
            y: y coordinate (m)
            coordinate_type: preferred coordinate type for plots and output
            mask: initial mask (0: land, 1: water)
            H: initial bathymetric depth. This is the distance between the bottom and
                some arbitrary depth reference (m, positive if bottom lies below the
                depth reference). Typically the depth reference is mean sea level.
            z0: minimum hydrodynamic bottom roughness (m)
            f: Coriolis parameter. By default this is calculated from latitude ``lat``
                if provided.
            periodic_x: use periodic boundary in x-direction (left == right)
            periodic_y: use periodic boundary in y-direction (top == bottom)
        """
        if nx <= 0:
            raise Exception(f"Number of x points is {nx} but must be > 0")
        if ny <= 0:
            raise Exception(f"Number of y points is {ny} but must be > 0")
        has_xy = x is not None and y is not None
        has_lonlat = lon is not None and lat is not None
        if not (has_xy or has_lonlat):
            raise Exception(f"Either x and y, or lon and lat, must be provided")
        if lat is None and f is None:
            raise Exception(
                "Either lat of f must be provided to determine the Coriolis parameter."
            )
        if coordinate_type is None:
            coordinate_type = CoordinateType.XY if has_xy else CoordinateType.LONLAT
        assert (coordinate_type == CoordinateType.XY and has_xy) or (
            coordinate_type == CoordinateType.LONLAT and has_lonlat
        )

        self.nx = nx
        self.ny = ny
        self.periodic_x = periodic_x
        self.periodic_y = periodic_y
        self.comm = comm
        self.coordinate_type = coordinate_type
        self.root_logger = logger or parallel.get_logger()
        self.logger = self.root_logger.getChild("domain")

        self.open_boundaries = open_boundaries.OpenBoundaries(
            nx, ny, self.logger.getChild("open_boundaries")
        )
        self.rivers = rivers.Rivers(
            nx, ny, coordinate_type, self.logger.getChild("rivers")
        )
        self.default_output_transforms = []
        self.input_grid_mappers = []

        if comm.rank != 0:
            return

        self._x = self._map_array(x, edges=EdgeTreatment.EXTRAPOLATE)
        self._y = self._map_array(y, edges=EdgeTreatment.EXTRAPOLATE)
        if lon is not None and np.shape(lon) != (1 + ny * 2, 1 + nx * 2):
            # Interpolate longitude in cos-sin space to handle periodic boundary
            # condition, but skip this if longitude is already provided on supergrid
            # (no interpolation needed) to improve accuracy of rotation tests.
            lon_rad = DEG2RAD * np.asarray(lon, dtype=float)
            coslon = np.cos(lon_rad)
            sinlon = np.sin(lon_rad)
            coslon = self._map_array(coslon, edges=EdgeTreatment.EXTRAPOLATE)
            sinlon = self._map_array(sinlon, edges=EdgeTreatment.EXTRAPOLATE)
            self._lon = np.arctan2(sinlon, coslon) * RAD2DEG
        else:
            self._lon = self._map_array(lon)
        self._lat = self._map_array(lat, edges=EdgeTreatment.EXTRAPOLATE)
        self._mask = self._map_array(mask, missing_value=0, dtype=int)
        self._H = self._map_array(H)
        self._z0 = self._map_array(z0)
        if f is None:
            # Calculate Coriolis parameter from latitude
            f = coriolis(lat)
        self._f = self._map_array(f)

        kwargs_expand = {}
        if self.periodic_x:
            kwargs_expand["edges_x"] = EdgeTreatment.EXTRAPOLATE_PERIODIC
        if self.periodic_y:
            kwargs_expand["edges_y"] = EdgeTreatment.EXTRAPOLATE_PERIODIC
        if has_xy:
            # Expand x, y by 1 in each direction to calculate dx, dy, rotation
            x_ex = expand_2d(self._x, **kwargs_expand)
            y_ex = expand_2d(self._y, **kwargs_expand)
        if has_lonlat:
            # Expand lon, lat by 1 in each direction to calculate dx, dy, rotation
            lon_rad_ex = DEG2RAD * expand_2d(self._lon, **kwargs_expand)
            lat_rad_ex = DEG2RAD * expand_2d(self._lat, **kwargs_expand)

        if has_xy:
            dx_x = x_ex[1:-1, 2:] - x_ex[1:-1, :-2]
            dy_x = y_ex[1:-1, 2:] - y_ex[1:-1, :-2]
            dx_y = x_ex[2:, 1:-1] - x_ex[:-2, 1:-1]
            dy_y = y_ex[2:, 1:-1] - y_ex[:-2, 1:-1]
            scale = 1.0
        else:
            dlon_rad_x = lon_rad_ex[1:-1, 2:] - lon_rad_ex[1:-1, :-2]
            dlat_rad_x = lat_rad_ex[1:-1, 2:] - lat_rad_ex[1:-1, :-2]
            coslat_x = np.cos(0.5 * (lat_rad_ex[1:-1, 2:] + lat_rad_ex[1:-1, :-2]))
            dx_x = coslat_x * np.sin(0.5 * dlon_rad_x)
            dy_x = np.sin(0.5 * dlat_rad_x) * np.cos(0.5 * dlon_rad_x)
            dlon_rad_y = lon_rad_ex[2:, 1:-1] - lon_rad_ex[:-2, 1:-1]
            dlat_rad_y = lat_rad_ex[2:, 1:-1] - lat_rad_ex[:-2, 1:-1]
            coslat_y = np.cos(0.5 * (lat_rad_ex[2:, 1:-1] + lat_rad_ex[:-2, 1:-1]))
            dx_y = coslat_y * np.sin(0.5 * dlon_rad_y)
            dy_y = np.sin(0.5 * dlat_rad_y) * np.cos(0.5 * dlon_rad_y)
            scale = R_EARTH * 2.0
        self._dx = scale * np.hypot(dx_x, dy_x)
        self._dy = scale * np.hypot(dx_y, dy_y)

        if has_lonlat and (
            (self._lat != self._lat[0, 0]).any() or (self._lon != self._lon[0, 0]).any()
        ):
            # Proper rotation with respect to true North
            self._rotation = _rotation(lon_rad_ex, lat_rad_ex)
        else:
            # Rotation with respect to y axis - assumes y axis always points to
            # true North (can be valid only for infinitesimally small domain)
            self._rotation = _rotation(x_ex, y_ex)

        self._area = self._dx * self._dy

    def _map_array(
        self,
        values: Optional[npt.ArrayLike],
        *,
        dtype: npt.DTypeLike = float,
        edges: EdgeTreatment = EdgeTreatment.MISSING,
        missing_value=np.nan,
    ) -> Optional[np.ndarray]:
        if self.comm.rank != 0:
            return

        if values is None:
            return

        source_shape = np.shape(values)
        source_shape = (1,) * (2 - len(source_shape)) + source_shape  # broadcast
        target_shape = (self.ny * 2 + 1, self.nx * 2 + 1)

        def can_cast(*target_shape: int) -> bool:
            assert len(target_shape) == len(source_shape)
            return all((l == 1 or l == lr) for l, lr in zip(source_shape, target_shape))

        if source_shape[0] == 1 and source_shape[1] == 1:
            # scalar value
            mapped_values = np.array(values, dtype=dtype)
        elif can_cast(self.ny, self.nx):
            # values provided at cell centers
            edges_x = edges_y = edges
            if self.periodic_x and edges_x == EdgeTreatment.MISSING:
                edges_x = EdgeTreatment.PERIODIC
            if self.periodic_y and edges_y == EdgeTreatment.MISSING:
                edges_y = EdgeTreatment.PERIODIC
            if source_shape[0] == 1:
                values_sup = centers_to_supergrid_1d(
                    np.ravel(values),
                    edges=edges_x,
                    missing_value=missing_value,
                    dtype=dtype,
                )
                mapped_values = values_sup[np.newaxis, :]
            elif source_shape[1] == 1:
                values_sup = centers_to_supergrid_1d(
                    np.ravel(values),
                    edges=edges_y,
                    missing_value=missing_value,
                    dtype=dtype,
                )
                mapped_values = values_sup[:, np.newaxis]
            else:
                mapped_values = centers_to_supergrid_2d(
                    values,
                    edges_x=edges_x,
                    edges_y=edges_y,
                    missing_value=missing_value,
                    dtype=dtype,
                )
        elif can_cast(self.ny + 1, self.nx + 1):
            # values provided at cell corners
            if source_shape[0] == 1:
                values_sup = interfaces_to_supergrid_1d(np.ravel(values), dtype=dtype)
                mapped_values = values_sup[np.newaxis, :]
            elif source_shape[1] == 1:
                values_sup = interfaces_to_supergrid_1d(np.ravel(values), dtype=dtype)
                mapped_values = values_sup[:, np.newaxis]
            else:
                mapped_values = interfaces_to_supergrid_2d(values)
        else:
            # values provided on supergrid
            assert can_cast(
                *target_shape
            ), f"Cannot map array with shape {values.shape} to supergrid with shape {target_shape}"
            mapped_values = np.array(values, dtype=dtype)
        if mapped_values.shape != target_shape:
            mapped_values = np.broadcast_to(mapped_values, target_shape)
        return mapped_values

    def create_tiling(self) -> parallel.Tiling:
        mask = None if self.comm.rank != 0 else self._mask[1::2, 1::2]
        mask = self.comm.bcast(mask)
        return parallel.Tiling.autodetect(
            mask,
            periodic_x=self.periodic_x,
            periodic_y=self.periodic_y,
            logger=self.logger.getChild("subdomain_decomposition"),
        )

    def create_grids(
        self,
        nz: int,
        halox: int,
        haloy: int,
        fields: Optional[Mapping[str, core.Array]] = None,
        tiling: Optional[parallel.Tiling] = None,
        input_manager=None,
        velocity_grids: int = 0,
    ) -> core.Grid:
        if self.comm.rank == 0:
            # We are the root node - update the global mask
            self.infer_UVX_masks2()
            self.open_boundaries.adjust_mask(self._mask)

        if tiling is None:
            tiling = self.create_tiling()
        elif tiling.nx_glob is None:
            tiling.set_extent(self.nx, self.ny)

        def create_grid(
            postfix: str,
            ioffset: int,
            joffset: int,
            *,
            overlap: int = 0,
            **kwargs,
        ) -> core.Grid:
            grid = core.Grid(
                tiling.nx_sub + overlap,
                tiling.ny_sub + overlap,
                nz,
                halox=halox,
                haloy=haloy,
                postfix=postfix,
                fields={} if fields is None else fields,
                tiling=tiling,
                ioffset=ioffset,
                joffset=joffset,
                overlap=overlap,
                **kwargs,
            )
            self._populate_grid(grid)
            grid.input_manager = input_manager
            grid.default_output_transforms = self.default_output_transforms
            grid.input_grid_mappers = self.input_grid_mappers
            return grid

        U = V = X = UU = UV = VU = VV = None

        if velocity_grids > 1:
            UU = create_grid("_uu_adv", 3, 1)
            UV = create_grid("_uv_adv", 2, 2)
            VU = create_grid("_vu_adv", 2, 2)
            VV = create_grid("_vv_adv", 1, 3)

        if velocity_grids > 0:
            U = create_grid("u", 2, 1, ugrid=UU, vgrid=UV)
            V = create_grid("v", 1, 2, ugrid=VU, vgrid=VV)
            X = create_grid("x", 0, 0, overlap=1)

        T = create_grid("t", 1, 1, ugrid=U, vgrid=V, xgrid=X)
        T.rivers = self.rivers

        if velocity_grids > 0:
            T.infer_water_contact()
            T.close_flux_interfaces()

        if velocity_grids > 1:
            U.close_flux_interfaces()
            V.close_flux_interfaces()

            # No transport between velocity points along an open boundary (just outside)
            # This is done to state that no valid values (in e.g. h and D) are required
            # in these points.
            UV.mask.all_values[:-1, :][
                (U.mask.all_values[:-1, :] == 4) & (U.mask.all_values[1:, :] == 4)
            ] = 0
            VU.mask.all_values[:, :-1][
                (V.mask.all_values[:, :-1] == 4) & (V.mask.all_values[:, 1:] == 4)
            ] = 0

        for grid in (T, U, V, X, UU, UV, VU, VV):
            if grid is not None:
                grid.freeze()

        self.open_boundaries.initialize(T, tiling)
        T.z.open_boundaries = open_boundaries.ArrayOpenBoundaries(T.z)
        self.open_boundaries.z = T.z.open_boundaries.values

        self.rivers.initialize(T)

        return T

    def _populate_grid(self, grid: core.Grid):
        NAMEMAP = dict(z0="_z0b_min", f="_cor", rotation="rotation")
        is_root = grid.tiling.rank == 0

        edges_x = EdgeTreatment.PERIODIC if self.periodic_x else EdgeTreatment.MISSING
        edges_y = EdgeTreatment.PERIODIC if self.periodic_y else EdgeTreatment.MISSING

        def scatter(source: np.ndarray, target: np.ndarray, fill_value):
            s = parallel.Scatter(
                grid.tiling, target, grid.halox, grid.haloy, grid.overlap
            )
            all_data = None
            if is_root:
                if grid.joffset > 1 or grid.ioffset > 1:
                    # UU or VV grid that needs one more strip of 1 cell beyond the end
                    # of the supergrid. Normally that is
                    source = expand_2d(
                        source,
                        edges_x=edges_x,
                        edges_y=edges_y,
                        missing_value=fill_value,
                    )[1:, 1:]
                data = source[grid.joffset :: 2, grid.ioffset :: 2]
                all_data = np.full(global_shape, fill_value, dtype=target.dtype)
                all_data[...] = data[: global_shape[0], : global_shape[1]]
            s(all_data)
            return all_data

        global_shape = (
            grid.tiling.ny_glob + grid.overlap,
            grid.tiling.nx_glob + grid.overlap,
        )
        for name in (
            "x",
            "y",
            "lon",
            "lat",
            "f",
            "dx",
            "dy",
            "rotation",
            "area",
            "mask",
            "H",
            "z0",
        ):
            source = getattr(self, f"_{name}", None)
            available = grid.tiling.comm.bcast(source is not None)
            target = getattr(grid, NAMEMAP.get(name, f"_{name}"), None)
            if available and target is not None:
                source = scatter(source, target.all_values, target.fill_value)
                if is_root:
                    # On the root node, keep a pointer to the full global field
                    # This will be used preferentially for full-domain output
                    target.attrs["_global_values"] = source
                if self.periodic_x or self.periodic_y:
                    target.update_halos()

    @calculate_and_bcast
    def cfl_check(
        self, z: float = 0.0, return_location: bool = False
    ) -> Union[float, Tuple[float, int, int, float]]:
        """Determine maximum time step for depth-integrated equations

        Args:
            z: surface elevation (m) at rest
            return_location: whether to also return the location
                and depth that determined the maximum step

        Note: this returns global indices for the T grid, not the supergrid
        """
        mask = self.mask[1::2, 1::2] > 0
        dx = self.dx[1::2, 1::2]
        dy = self.dy[1::2, 1::2]
        H = self.H[1::2, 1::2]
        denom2 = (2.0 * GRAVITY) * (H + z) * (dx**2 + dy**2)
        maxdts = dx * dy / np.sqrt(denom2, where=mask, out=np.ones_like(H))
        maxdts[~mask] = np.inf
        maxdt = maxdts.min()
        if return_location:
            j, i = np.unravel_index(np.argmin(maxdts), maxdts.shape)
            return (maxdt, i, j, self.H[1 + 2 * j, 1 + 2 * i])
        return maxdt

    @property
    def maxdt(self) -> float:
        return self.cfl_check()

    # @calculate_and_bcast
    def infer_UVX_masks2(self):
        if not self._mask.flags.writeable:
            self._mask = self._mask.copy()

        tmask = self._mask[1::2, 1::2]
        umask = self._mask[1::2, 2::2]
        vmask = self._mask[2::2, 1::2]
        xmask = self._mask[::2, ::2]

        edges_x = EdgeTreatment.PERIODIC if self.periodic_x else EdgeTreatment.MISSING
        edges_y = EdgeTreatment.PERIODIC if self.periodic_y else EdgeTreatment.MISSING
        tmask_ex = expand_2d(tmask, edges_x=edges_x, edges_y=edges_y, missing_value=0)

        # Now mask U,V,X points unless all their T neighbors are valid - this mask will
        # be sent to Fortran and determine which points are computed
        bad = (tmask_ex[1:-1, 2:] == 0) | (tmask_ex[1:-1, 1:-1] == 0)
        umask[bad & (umask == 1)] = 0
        bad = (tmask_ex[2:, 1:-1] == 0) | (tmask_ex[1:-1, 1:-1] == 0)
        vmask[bad & (vmask == 1)] = 0
        bad = (
            (tmask_ex[1:, 1:] == 0)
            | (tmask_ex[:-1, 1:] == 0)
            | (tmask_ex[1:, :-1] == 0)
            | (tmask_ex[:-1, :-1] == 0)
        )
        xmask[bad] = 0

    @calculate_and_bcast
    def mask_shallow(self, minimum_depth: float):
        """Mask all points shallower less the specified value.

        Args:
            minimum_depth: minimum bathmetric depth :attr:`H`; points that are
                shallower will be masked
        """
        self.mask[self.H < minimum_depth] = 0

    @calculate_and_bcast
    def limit_velocity_depth(self, critical_depth: float = np.inf):
        """Decrease bathymetric depth of velocity (U, V) points to the minimum of the
        bathymetric depth of both neighboring T points, wherever one of these two
        points is shallower than the specified critical depth.

        Args:
            critical_depth: neighbor depth at which the limiting starts. If either
                neighbor (T grid) is shallower than this value, the depth of velocity
                point (U or V grid) is restricted. If not provided, ``self.Dcrit`` is
                used.
        """
        tdepth = self.H_[1::2, 1::2]
        Vsel = (tdepth[1:, :] <= critical_depth) | (tdepth[:-1, :] <= critical_depth)
        self.H_[2:-2:2, 1::2][Vsel] = np.minimum(tdepth[1:, :], tdepth[:-1, :])[Vsel]
        Usel = (tdepth[:, 1:] <= critical_depth) | (tdepth[:, :-1] <= critical_depth)
        self.H_[1::2, 2:-2:2][Usel] = np.minimum(tdepth[:, 1:], tdepth[:, :-1])[Usel]
        self.logger.info(
            f"limit_velocity_depth has decreased depth in {Usel.sum()} U points"
            f" ({Usel.sum(where=self.mask_[1::2, 2:-2:2] > 0)} currently unmasked),"
            f" {Vsel.sum()} V points ({Vsel.sum(where=self.mask_[2:-2:2, 1::2] > 0)}"
            " currently unmasked)."
        )

    @calculate_and_bcast
    def mask_rectangle(
        self,
        xmin: Optional[float] = None,
        xmax: Optional[float] = None,
        ymin: Optional[float] = None,
        ymax: Optional[float] = None,
        mask_value: int = 0,
        coordinate_type: Optional[CoordinateType] = None,
    ):
        """Mask all points that fall within the specified rectangle.

        Args:
            xmin: lower native x coordinate of the rectangle to mask
                (default: left boundary of the domain)
            xmax: upper native x coordinate of the rectangle to mask
                (default: right boundary of the domain)
            ymin: lower native y coordinate of the rectangle to mask
                (default: bottom boundary of the domain)
            ymax: upper native y coordinate of the rectangle to mask
                (default: top boundary of the domain)

        Coordinates will be interpreted as longitude, latitude if the domain is
        configured as spherical; otherwise they will be interpreted as Cartesian
        x and y (m).
        """
        selected = np.ones(self.mask.shape, dtype=bool)
        coordinate_type = coordinate_type or self.coordinate_type
        if coordinate_type == CoordinateType.LONLAT:
            x, y = (self.lon, self.lat)
        elif coordinate_type == CoordinateType.XY:
            x, y = (self.x, self.y)
        else:
            x = np.linspace(-0.5, self.nx + 0.5, 1 + 2 * self.nx)
            y = np.linspace(-0.5, self.ny + 0.5, 1 + 2 * self.ny)
            x, y = np.broadcast_arrays(x[np.newaxis, :], y[:, np.newaxis])
        if xmin is not None:
            selected &= x >= xmin
        if xmax is not None:
            selected &= x <= xmax
        if ymin is not None:
            selected &= y >= ymin
        if ymax is not None:
            selected &= y <= ymax
        self.mask[selected] = mask_value

    @calculate_and_bcast
    def mask_indices(self, istart, istop, jstart, jstop, value: int = 0):
        """Mask all points that fall within the specified rectangle.

        Args:
            istart: lower x index (first that is included)
            istop: upper x index (first that is EXcluded)
            jstart: lower y index (first that is included)
            jstop: upper y index (first that is EXcluded)
        """
        istart_T = 1 + 2 * istart
        istop_T = 1 + 2 * istop
        jstart_T = 1 + 2 * jstart
        jstop_T = 1 + 2 * jstop
        self.mask_[jstart_T:jstop_T, istart_T:istop_T] = value

    def rotate(self) -> "Domain":
        def tp(array):
            return None if array is None else np.transpose(array)[::-1, :]

        kwargs = dict(
            coordinate_type=self.coordinate_type,
            comm=self.comm,
            logger=self.root_logger,
        )
        if self.comm.rank == 0:
            kwargs.update(
                lon=tp(self.lon),
                lat=tp(self.lat),
                x=tp(self.x),
                y=tp(self.y),
                mask=tp(self.mask),
                H=tp(self.H),
                z0=tp(self.z0),
                f=tp(self.f),
            )
        return Domain(self.ny, self.nx, **kwargs)

    @calculate_and_bcast
    def plot(
        self,
        field: Optional[np.ndarray] = None,
        *,
        fig: Optional["matplotlib.figure.Figure"] = None,
        show_bathymetry: bool = True,
        show_mask: bool = False,
        show_mesh: bool = True,
        show_rivers: bool = True,
        show_subdomains: bool = False,
        editable: bool = False,
        coordinate_type: Optional[CoordinateType] = None,
        tiling: Optional[parallel.Tiling] = None,
        label: Optional[str] = None,
        cmap: Union[None, "matplotlib.colors.Colormap", str] = None,
    ):
        """Plot the domain, optionally including bathymetric depth, mesh and
        river positions.

        Args:
            fig: :class:`matplotlib.figure.Figure` instance to plot to. If not provided,
                a new figure is created.
            show_bathymetry: show bathymetry as color map
            show_mask: show mask as color map (this disables ``show_bathymetry``)
            show_mesh: show model grid
            show_rivers: show rivers with position and name
            editable: allow interactive selection of rectangular regions in the domain
                plot that are subsequently masked out
            sub: plot the subdomain, not the global domain

        Returns:
            :class:`matplotlib.figure.Figure` instance for processes with rank 0 or if
                ``sub`` is ``True``, otherwise ``None``
        """
        import matplotlib.pyplot
        import matplotlib.collections
        import matplotlib.widgets

        if fig is None:
            fig, ax = matplotlib.pyplot.subplots(
                figsize=(0.15 * self.nx, 0.15 * self.ny)
            )
        else:
            ax = fig.gca()

        if coordinate_type is None:
            coordinate_type = self.coordinate_type
        if coordinate_type == CoordinateType.LONLAT:
            x, y = (self._lon, self._lat)
            xlabel, ylabel = "longitude (°East)", "latitude (°North)"
        elif coordinate_type == CoordinateType.XY:
            x, y = (self._x, self._y)
            xlabel, ylabel = "x (m)", "y (m)"
        else:
            x = 0.5 * np.arange(1 + self.nx * 2)
            y = 0.5 * np.arange(1 + self.ny * 2)[:, np.newaxis]
            x, y = np.broadcast_arrays(x, y)
            xlabel, ylabel = "cell index", "cell index"

        if field is None:
            if show_mask:
                field = self._mask
                label = "mask value"
            elif show_bathymetry:
                import cmocean

                cmap = cmocean.cm.deep
                cmap.set_bad("gray")
                field = np.ma.array(self._H, mask=self._mask == 0)
                label = "undisturbed water depth (m)"

        c = ax.pcolormesh(
            x,
            y,
            field,
            alpha=0.5 if show_mesh else 1,
            shading="auto",
            cmap=cmap,
        )
        # c = ax.contourf(x, y, np.ma.array(self.H, mask=self.mask==0), 20, alpha=0.5 if show_mesh else 1)
        cb = fig.colorbar(c)
        if label is not None:
            cb.set_label(label)

        if show_rivers and self.rivers:
            x_ = None if self._x is None else self._x[1::2, 1::2]
            y_ = None if self._y is None else self._y[1::2, 1::2]
            lon_ = None if self._lon is None else self._lon[1::2, 1::2]
            lat_ = None if self._lat is None else self._lat[1::2, 1::2]
            self.rivers.map_to_grid(self._mask[1::2, 1::2], x_, y_, lon_, lat_)
            for river in self.rivers.values():
                i_sup, j_sup = 1 + river.i_glob * 2, 1 + river.j_glob * 2
                river_x, river_y = x[j_sup, i_sup], y[j_sup, i_sup]
                ax.plot([river_x], [river_y], ".r")
                ax.text(river_x, river_y, river.name, color="r")

        def plot_mesh(ax, x, y, **kwargs):
            segs1 = np.stack((x, y), axis=2)
            segs2 = segs1.transpose(1, 0, 2)
            ax.add_collection(matplotlib.collections.LineCollection(segs1, **kwargs))
            ax.add_collection(matplotlib.collections.LineCollection(segs2, **kwargs))

        if show_mesh:
            plot_mesh(
                ax, x[::2, ::2], y[::2, ::2], colors="k", linestyle="-", linewidth=0.3
            )
            # ax.pcolor(x[1::2, 1::2], y[1::2, 1::2], np.ma.array(x[1::2, 1::2], mask=True), edgecolors='k', linestyles='--', linewidth=.2)
            # pc = ax.pcolormesh(x[1::2, 1::2], y[1::2, 1::2],  np.ma.array(x[1::2, 1::2], mask=True), edgecolor='gray', linestyles='--', linewidth=.2)
            ax.plot(x[::2, ::2], y[::2, ::2], "xk", markersize=3.0)
            ax.plot(x[1::2, 1::2], y[1::2, 1::2], ".k", markersize=2.5)

        if show_subdomains:
            assert tiling is not None
            for icol in range(tiling.ncol + 1):
                i = 2 * (tiling.xoffset_global + icol * tiling.nx_sub)
                if i >= 0 and i < x.shape[-1]:
                    ax.plot(x[:, i], y[:, i], "-k", linewidth=2.0)
                    ax.plot(x[:, i], y[:, i], "--w", linewidth=2.0)
            for irow in range(tiling.nrow + 1):
                j = 2 * (tiling.yoffset_global + irow * tiling.ny_sub)
                if j >= 0 and j < x.shape[-2]:
                    ax.plot(x[j, :], y[j, :], "-k", linewidth=2.0)
                    ax.plot(x[j, :], y[j, :], "--w", linewidth=2.0)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if coordinate_type != CoordinateType.LONLAT:
            ax.axis("equal")
        xmin, xmax = np.nanmin(x), np.nanmax(x)
        ymin, ymax = np.nanmin(y), np.nanmax(y)
        xmargin = 0.05 * (xmax - xmin)
        ymargin = 0.05 * (ymax - ymin)
        ax.set_xlim(xmin - xmargin, xmax + xmargin)
        ax.set_ylim(ymin - ymargin, ymax + ymargin)

        def on_select(eclick, erelease):
            xmin, xmax = (
                min(eclick.xdata, erelease.xdata),
                max(eclick.xdata, erelease.xdata),
            )
            ymin, ymax = (
                min(eclick.ydata, erelease.ydata),
                max(eclick.ydata, erelease.ydata),
            )
            self.mask_rectangle(xmin, xmax, ymin, ymax)
            c.set_array(np.ma.array(self.H, mask=self.mask == 0).ravel())
            fig.canvas.draw()
            # self.sel.set_active(False)
            # self.sel = None
            # ax.draw()
            # fig.clf()
            # self.plot(fig=fig, show_mesh=show_mesh)

        if editable:
            self.sel = matplotlib.widgets.RectangleSelector(
                ax, on_select, useblit=True, button=[1], interactive=False
            )
        return fig


# def load(path: str, nz: int, **kwargs) -> "Domain":
#     """Load domain from file. Typically this is a file created by :meth:`Domain.save`.

#     Args:
#         path: NetCDF file to load from
#         nz: number of vertical layers
#         **kwargs: additional keyword arguments to pass to :class:`Domain`
#     """
#     name2kwarg = dict(z0b_min="z0", cor="f")
#     with netCDF4.Dataset(path) as nc:
#         for name in ("lon", "lat", "x", "y", "H", "mask", "z0b_min", "cor", "z0", "f"):
#             if name in nc.variables and name not in kwargs:
#                 kwargs[name2kwarg.get(name, name)] = nc.variables[name][...]
#     spherical = kwargs.get("spherical", "lon" in kwargs)
#     ny, nx = kwargs["lon" if spherical else "x"].shape
#     nx = (nx - 1) // 2
#     ny = (ny - 1) // 2
#     return create(nx, ny, nz, spherical=spherical, **kwargs)

#     def save(self, path: str, full: bool = False, sub: bool = False):
#         """Save grid to a NetCDF file that can be interpreted by :func:`load`.

#         Args:
#             path: NetCDF file to save to
#             full: also save every field including its halos separately
#                 This can be useful for debugging.
#             sub: save the local subdomain, not the global domain
#         """
#         if self.glob is not self and not sub:
#             # We need to save the global domain; not the current subdomain.
#             # If we are the root, divert the plot command to the global domain.
#             # Otherwise just ignore this and return.
#             if self.glob:
#                 self.glob.save(path, full)
#             return

#         with netCDF4.Dataset(path, "w") as nc:

#             def create(
#                 name, units, long_name, values, coordinates: str, dimensions=("y", "x")
#             ):
#                 fill_value = None
#                 if np.ma.getmask(values) is not np.ma.nomask:
#                     fill_value = values.fill_value
#                 ncvar = nc.createVariable(
#                     name, values.dtype, dimensions, fill_value=fill_value
#                 )
#                 ncvar.units = units
#                 ncvar.long_name = long_name
#                 # ncvar.coordinates = coordinates
#                 ncvar[...] = values

#             def create_var(name, units, long_name, values, values_):
#                 if values is None:
#                     return
#                 create(
#                     name,
#                     units,
#                     long_name,
#                     values,
#                     coordinates="lon lat" if self.spherical else "x y",
#                 )
#                 if full:
#                     create(
#                         name + "_",
#                         units,
#                         long_name,
#                         values_,
#                         dimensions=("y_", "x_"),
#                         coordinates="lon_ lat_" if self.spherical else "x_ y_",
#                     )

#             nc.createDimension("x", self.H.shape[1])
#             nc.createDimension("y", self.H.shape[0])
#             if full:
#                 nc.createDimension("x_", self.H_.shape[1])
#                 nc.createDimension("y_", self.H_.shape[0])
#                 create_var("dx", "m", "dx", self.dx, self.dx_)
#                 create_var("dy", "m", "dy", self.dy, self.dy_)
#                 create_var("area", "m2", "area", self.dx * self.dy, self.dx_ * self.dy_)
#             create_var("lat", "degrees_north", "latitude", self.lat, self.lat_)
#             create_var("lon", "degrees_east", "longitude", self.lon, self.lon_)
#             create_var("x", "m", "x", self.x, self.x_)
#             create_var("y", "m", "y", self.y, self.y_)
#             create_var("H", "m", "undisturbed water depth", self.H, self.H_)
#             create_var("mask", "", "mask", self.mask, self.mask_)
#             create_var("z0", "m", "bottom roughness", self.z0b_min, self.z0b_min_)
#             create_var("f", "", "Coriolis parameter", self.cor, self.cor_)

#     def contains(
#         self,
#         x: float,
#         y: float,
#         include_halos: bool = False,
#         spherical: Optional[bool] = None,
#     ) -> bool:
#         """Determine whether the domain contains the specified point.

#         Args:
#             x: native x coordinate (longitude for spherical grids, Cartesian coordinate
#                 in m otherwise)
#             y: native y coordinate (latitude for spherical grids, Cartesian coordinate
#                 in m otherwise)
#             include_halos: whether to also search the halos

#         Returns:
#             True if the point falls within the domain, False otherwise
#         """
#         if spherical is None:
#             spherical = self.spherical
#         local_slice, _, _, _ = self.tiling.subdomain2slices(
#             halox_sub=2 * self.halox,
#             haloy_sub=2 * self.haloy,
#             halox_glob=2 * self.halox,
#             haloy_glob=2 * self.haloy,
#             scale=2,
#             share=1,
#             exclude_halos=not include_halos,
#             exclude_global_halos=True,
#         )
#         allx, ally = (self.lon_, self.lat_) if spherical else (self.x_, self.y_)
#         allx, ally = allx[local_slice], ally[local_slice]
#         ny, nx = allx.shape

#         # Determine whether point falls within current subdomain
#         # based on https://wrf.ecse.rpi.edu/Research/Short_Notes/pnpoly.html
#         x_bnd = np.concatenate(
#             (
#                 allx[0, :-1],
#                 allx[:-1, -1],
#                 allx[-1, nx - 1 : 0 : -1],
#                 allx[ny - 1 : 0 : -1, 0],
#             )
#         )
#         y_bnd = np.concatenate(
#             (
#                 ally[0, :-1],
#                 ally[:-1, -1],
#                 ally[-1, nx - 1 : 0 : -1],
#                 ally[ny - 1 : 0 : -1, 0],
#             )
#         )
#         assert not np.isnan(x_bnd).any(), f"Invalid x boundary: {x_bnd}."
#         assert not np.isnan(y_bnd).any(), f"Invalid y boundary: {y_bnd}."
#         assert x_bnd.size == 2 * ny + 2 * nx - 4
#         inside = False
#         for i, (vertxi, vertyi) in enumerate(zip(x_bnd, y_bnd)):
#             vertxj, vertyj = x_bnd[i - 1], y_bnd[i - 1]
#             if (vertyi > y) != (vertyj > y) and x < (vertxj - vertxi) * (y - vertyi) / (
#                 vertyj - vertyi
#             ) + vertxi:
#                 inside = not inside
#         return inside
