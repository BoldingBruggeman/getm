import logging
from typing import MutableMapping, Optional, List, Union
import datetime

import numpy.typing
import cftime

from .. import core
from . import operators

class FieldManager:
    def __init__(self):
        self.fields: MutableMapping[str, core.Array] = {}

    def register(self, array: core.Array):
        assert array.name is not None, 'Cannot register field without name.'
        assert array.name not in self.fields, 'A field with name "%s" has already been registered.' % array.name
        self.fields[array.name] = array

class File(operators.FieldCollection):
    def __init__(self, field_manager: FieldManager, logger: logging.Logger, interval: Union[int, datetime.timedelta]=1, path: Optional[str]=None, default_dtype: Optional[numpy.typing.DTypeLike]=None):
        super().__init__(field_manager, default_dtype=default_dtype)
        self._logger = logger
        self.next = None
        self.interval_in_dt = isinstance(interval, int)
        self.interval = interval if self.interval_in_dt else interval.total_seconds()
        self.save_on_close_only = self.interval_in_dt and self.interval == -1
        self.path = path

    def start(self, itimestep: int, time: Optional[cftime.datetime], save: bool, default_time_reference: Optional[cftime.datetime]):
        if save and not self.save_on_close_only:
            self._logger.debug('Saving initial state')
            self.save_now(0., time)
        now = itimestep if self.interval_in_dt else 0.
        self.next = now + self.interval

    def save(self, seconds_passed: float, itimestep: int, time: Optional[cftime.datetime]):
        for field in self._updatable:
            field.update()
        if self.save_on_close_only:
            return
        now = itimestep if self.interval_in_dt else seconds_passed
        if now >= self.next:
            self._logger.debug('Saving')
            self.save_now(seconds_passed, time)
            self.next += self.interval

    def close(self, seconds_passed: float, time: Optional[cftime.datetime]):
        if self.save_on_close_only:
            self.save_now(seconds_passed, time)
        self._logger.debug('Closing')
        self.close_now(seconds_passed, time)

    def save_now(self, seconds_passed: float, time: Optional[cftime.datetime]):
        raise NotImplementedError

    def close_now(self, seconds_passed: float, time: Optional[cftime.datetime]):
        pass

# Note: netcdf cannot be imported before FieldManager and File are defined, because both are imported by netcdf itself.
# If this import statement came earlier, it would cause a circular dependency error.
from . import netcdf

class OutputManager(FieldManager):
    def __init__(self, rank: int, logger: Optional[logging.Logger]=None):
        FieldManager.__init__(self)
        self.rank = rank
        self.files: List[File] = []
        self._logger = logger or logging.getLogger()

    def add_netcdf_file(self, path: str, **kwargs) -> netcdf.NetCDFFile:
        self._logger.debug('Adding NetCDF file %s' % path)
        file = netcdf.NetCDFFile(self, self._logger.getChild(path), path, rank=self.rank, **kwargs)
        self.files.append(file)
        return file

    def add_restart(self, path: str, **kwargs) -> netcdf.NetCDFFile:
        kwargs.setdefault('interval', -1)
        file = self.add_netcdf_file(path, **kwargs)
        for field in self.fields.values():
            if field.attrs.get('_part_of_state'):
                file.request(field)
        return file

    def start(self, itimestep: int=0, time: Optional[cftime.datetime]=None, save: bool=True, default_time_reference: Optional[cftime.datetime]=None):
        for file in self.files:
            file.start(itimestep, time, save, default_time_reference or time)

    def save(self, seconds_passed: float, itimestep: int, time: Optional[cftime.datetime]=None):
        for file in self.files:
            file.save(seconds_passed, itimestep, time)

    def close(self, seconds_passed: float, time: Optional[cftime.datetime]=None):
        for file in self.files:
            self._logger.debug('Closing %s' % file.path)
            file.close(seconds_passed, time)
