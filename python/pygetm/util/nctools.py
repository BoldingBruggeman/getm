from typing import Optional, Iterable

import netCDF4


def copy_variable(
    ncvar: netCDF4.Variable,
    nctarget: netCDF4.Dataset,
    dimensions: Optional[Iterable[str]] = None,
    copy_data: bool = True,
    chunksizes=None,
    name: str = None,
    zlib: bool = False,
) -> netCDF4.Variable:
    if name is None:
        name = ncvar.name
    if dimensions is None:
        dimensions = ncvar.dimensions
    for dim in dimensions:
        if dim not in nctarget.dimensions:
            length = ncvar.shape[ncvar.dimensions.index(dim)]
            nctarget.createDimension(dim, length)
    fill_value = None if not hasattr(ncvar, "_FillValue") else ncvar._FillValue
    ncvarnew = nctarget.createVariable(
        name,
        ncvar.dtype,
        dimensions,
        fill_value=fill_value,
        chunksizes=chunksizes,
        zlib=zlib,
    )
    ncvarnew.setncatts(
        {att: getattr(ncvar, att) for att in ncvar.ncattrs() if att != "_FillValue"}
    )
    ncvarnew.set_auto_maskandscale(False)
    if copy_data:
        ncvarnew[...] = ncvar[...]
    return ncvarnew
