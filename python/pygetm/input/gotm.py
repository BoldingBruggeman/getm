import argparse
import datetime
from typing import Iterable, Mapping, Union
import os

import cftime
import numpy as np
import xarray as xr


METEO_COLUMNS = ("u10", "v10", "sp", "t2m", "hum", "tcc")
METEO_LONG_NAMES = dict(
    u10="eastward wind speed",
    v10="northward wind speed",
    sp="surface pressure",
    t2m="air temperature @ 2 m",
    hum="humidity @ 2 m",
    tcc="total cloud cover",
)
METEO_UNITS = dict(u10="m s-1", v10="m s-1", tcc="1")


def get_timeseries(
    path: str,
    columns: Union[Mapping[int, str], Iterable[str]],
    units: Mapping[str, str] = {},
    long_names: Mapping[str, str] = {},
) -> xr.Dataset:
    dts = []
    all_values = []
    ncol = None
    if not isinstance(columns, Mapping):
        columns = dict(enumerate(columns))
        ncol = max(columns) + 1
    with open(path) as f:
        last_dt = None
        for line in f:
            if line.startswith("#"):
                continue
            dt = datetime.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            assert last_dt is None or last_dt <= dt
            items = line[20:].rstrip("\n").split()
            assert ncol is None or len(items) == ncol
            dts.append(
                cftime.datetime(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
                )
            )
            all_values.append(list(map(float, items)))

    all_values = np.asarray(all_values)
    timevar = xr.DataArray(np.array(dts), dims=("time",))
    name2var = dict(time=timevar)
    for icol, name in columns.items():
        values = all_values[:, icol]
        attrs = {}
        if name in units:
            attrs["units"] = units[name]
        if name in long_names:
            attrs["long_name"] = long_names[name]
        ar = xr.DataArray(
            values, coords={"time": timevar}, dims=("time",), name=name, attrs=attrs
        )
        ar.encoding["source"] = os.path.normpath(path)
        name2var[name] = ar
    return xr.Dataset(name2var)


def get_profiles(
    path: str,
    columns: Union[Mapping[int, str], Iterable[str]],
    units: Mapping[str, str] = {},
    long_names: Mapping[str, str] = {},
) -> xr.Dataset:
    dts = []
    all_values = []
    z = None
    ncol = None
    if not isinstance(columns, Mapping):
        columns = dict(enumerate(columns))
        ncol = max(columns) + 1
    with open(path) as f:
        last_dt = None
        while 1:
            line = f.readline()
            if line.startswith("#"):
                continue
            if not line:
                break
            dt = datetime.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
            assert last_dt is None or last_dt <= dt
            items = line[20:].rstrip("\n").split()
            n = int(items[0])
            up = int(items[1]) == 1
            values = []
            current_z = []
            for i in range(n):
                line = f.readline()
                if line.startswith("#"):
                    continue
                items = list(map(float, line.rstrip("\n").split()))
                current_z.append(items.pop(0))
                assert ncol is None or len(items) == ncol
                values.append(items)
            current_z = np.asarray(current_z)
            if z is None:
                z = current_z
            else:
                assert (current_z == z).all()
            dts.append(
                cftime.datetime(
                    dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second
                )
            )
            all_values.append(values)

    all_values = np.asarray(all_values)

    timevar = xr.DataArray(np.array(dts), dims=("time",))
    zvar = xr.DataArray(z, dims=("z",), attrs={"positive": "up", "units": "m"})
    name2var = dict(time=timevar)
    for icol, name in columns.items():
        values = all_values[..., icol]
        attrs = {}
        if name in units:
            attrs["units"] = units[name]
        if name in long_names:
            attrs["long_name"] = long_names[name]
        ar = xr.DataArray(
            values,
            coords={"time": timevar, "z": zvar},
            dims=("time", "z"),
            name=name,
            attrs=attrs,
        )
        ar.encoding["source"] = os.path.normpath(path)
        name2var[name] = ar
    return xr.Dataset(name2var)


def get_meteo(path: str) -> xr.Dataset:
    return get_timeseries(path, METEO_COLUMNS, METEO_UNITS, METEO_LONG_NAMES)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file")
    args = parser.parse_args()
    ds = get_meteo(args.file)
    print(ds)
