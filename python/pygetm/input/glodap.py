import tarfile
import shutil
import glob
import tempfile
import os.path
import netCDF4
import argparse
import logging
import urllib.request

import numpy as np

from pygetm import pygsw
from pygetm.util.nctools import copy_variable
from pygetm.util.fill import Filler

URL = "https://www.nodc.noaa.gov/archive/arc0107/0162565/1.1/data/0-data/mapped/GLODAPv2.2016b_MappedClimatologies.tar.gz"


def download(outfile: str, logger: logging.Logger, source: str = URL):
    filename = os.path.basename(source)
    with tempfile.TemporaryDirectory() as root:
        target = os.path.join(root, filename)

        if source.startswith("http"):
            logger.info(f"Downloading {source}...")
            with urllib.request.urlopen(source) as response, open(target, "wb") as fout:
                shutil.copyfileobj(response, fout)
            source = target

        logger.info(f"Extracting {filename}...")
        with tarfile.open(source) as tar:
            tar.extractall(root)

        logger.info("Collecting variables...")
        with netCDF4.Dataset(outfile, "w", format="NETCDF4") as ncout:
            ncout.set_fill_off()
            for path in glob.glob(os.path.join(root, "**/*.nc"), recursive=True):
                with netCDF4.Dataset(path) as nc:
                    nc.set_auto_maskandscale(False)
                    varname = os.path.basename(path).rsplit(".", 2)[-2]
                    ncvar = nc[varname]
                    logger.info(f"  - {varname}: {ncvar.long_name} ({ncvar.units})...")
                    if not ncout.variables:
                        copy_variable(nc["Depth"], ncout)
                        copy_variable(nc["lon"], ncout).units = "degrees_east"
                        copy_variable(nc["lat"], ncout).units = "degrees_north"
                    ncvarout = copy_variable(ncvar, ncout)
                    ncvarout.coordinates = "lon lat Depth"


def fill(path: str, logger: logging.Logger):
    with netCDF4.Dataset(path, "r+") as nc:
        nc.set_auto_maskandscale(False)
        for name in nc.variables:
            if name not in ("lon", "lat", "Depth"):
                ncvar = nc[name]
                data = ncvar[:, :, :]
                mask = data == ncvar._FillValue
                logger.info(f"Filling {mask.sum()} masked points for {name}...")
                filler = Filler(mask, dim_weights=(0.9, 1.0, 1.0))
                filler(data)
                ncvar[:, :, :] = data


def add_density(path: str, logger: logging.Logger):
    with netCDF4.Dataset(path, "r+") as nc:
        lon, lat, z = nc["lon"][:] % 360, nc["lat"][:], nc["Depth"][:]
        lon, lat, z = np.broadcast_arrays(
            lon[np.newaxis, np.newaxis, :],
            lat[np.newaxis, :, np.newaxis],
            z[:, np.newaxis, np.newaxis],
        )
        salt = nc["salinity"][...]
        temp = nc["temperature"][...]
        unmasked = ~(np.ma.getmaskarray(salt) | np.ma.getmaskarray(temp))

        z = z[unmasked]

        logger.info("Calculating absolute salinity from practical salinity...")
        sa = pygsw.sa_from_sp(lon[unmasked], lat[unmasked], z, salt[unmasked])

        logger.info("Calculating conservative temperature...")
        pt = pygsw.pt0_from_t(sa, temp[unmasked], z)
        ct = pygsw.ct_from_pt(sa, pt)

        logger.info("Calculating density...")
        rho = pygsw.rho(sa, ct, z)

        res = np.full_like(nc["salinity"], nc["salinity"]._FillValue)

        res[unmasked] = sa
        ncvar = copy_variable(nc["salinity"], nc, name="sa", copy_data=False)
        ncvar.long_name = "absolute salinity"
        ncvar.units = "g kg-1"
        ncvar[:, :, :] = res

        res[unmasked] = ct
        ncvar = copy_variable(nc["temperature"], nc, name="ct", copy_data=False)
        ncvar.long_name = "conservative temperature"
        ncvar.units = "degrees_Celsius"
        ncvar[:, :, :] = res

        res[unmasked] = rho
        ncvar = copy_variable(nc["salinity"], nc, name="density", copy_data=False)
        ncvar.long_name = "in situ density"
        ncvar.units = "kg m-3"
        ncvar[:, :, :] = res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "outfile", nargs="?", default="glodap.nc", help="NetCDF file to write output to"
    )
    parser.add_argument(
        "--source", help="Source of the original GLODAP file (tar.gz).", default=URL
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()

    download(args.outfile, logger=logger, source=args.source)
    add_density(args.outfile, logger=logger)
    fill(args.outfile, logger=logger)
