import xarray
import cftime


def replace_calendar(da: xarray.DataArray, calendar: str) -> xarray.DataArray:
    tmcoord = da.getm.time
    new_time = [
        cftime.datetime(
            dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, calendar=calendar
        )
        for dt in tmcoord.values
    ]
    return da.assign_coords({tmcoord.name: new_time})
