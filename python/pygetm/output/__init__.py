import logging
from typing import Mapping, Optional, List, Union
import datetime
import enum

from numpy.typing import DTypeLike
import cftime

from .. import core
from . import operators


class TimeUnit(enum.Enum):
    TIMESTEPS = enum.auto()
    SECONDS = enum.auto()
    MINUTES = enum.auto()
    HOURS = enum.auto()
    DAYS = enum.auto()
    MONTHS = enum.auto()
    YEARS = enum.auto()


time_unit2seconds = {
    TimeUnit.SECONDS: 1,
    TimeUnit.MINUTES: 60,
    TimeUnit.HOURS: 3600,
    TimeUnit.DAYS: 86400,
}


class File(operators.FieldCollection):
    def __init__(
        self,
        available_fields: Mapping[str, core.Array],
        logger: logging.Logger,
        interval: Union[datetime.timedelta, int] = 1,
        interval_units: TimeUnit = TimeUnit.TIMESTEPS,
        start: Optional[cftime.datetime] = None,
        stop: Optional[cftime.datetime] = None,
        path: Optional[str] = None,
        default_dtype: Optional[DTypeLike] = None,
        save_initial: bool = True,
        sub: bool = False,
    ):
        """
        Args:
            interval: time interval to save at
            interval_units: units for time interval
                (not used if ``interval`` is given as :class:`datetime.timedelta`)
            start: simulation time at which to start saving.
                This defaults to the start of the simulation
            stop: simulation time at which to stop saving.
                This defaults to the end of the simulation
            default_dtype: default data type for real-valued variables
            sub: whether to save separate files per subdomain
            save_initial: whether to save at the very start of the simulation
        """
        super().__init__(available_fields, default_dtype=default_dtype, sub=sub)
        self._logger = logger
        self.next = None

        if isinstance(interval, datetime.timedelta):
            interval = interval.total_seconds()
            interval_units = TimeUnit.SECONDS
        elif interval_units in time_unit2seconds:
            interval *= time_unit2seconds[interval_units]
            interval_units = TimeUnit.SECONDS
        self.interval = interval
        self.interval_units = interval_units
        self.save_on_close_only = (
            self.interval_units == TimeUnit.TIMESTEPS and self.interval == -1
        )
        self.save_initial = save_initial and not self.save_on_close_only

        self.path = path
        self._start = start
        self._stop = stop

    def _next_time(
        self,
        seconds_passed: float,
        itimestep: int,
        time: Optional[cftime.datetime],
    ) -> Union[float, int]:
        if self.interval_units == TimeUnit.SECONDS:
            return seconds_passed + self.interval
        elif self.interval_units == TimeUnit.TIMESTEPS:
            return itimestep + self.interval
        elif self.interval_units == TimeUnit.MONTHS:
            next_month = time.month + self.interval
            next_year = time.year + (next_month - 1) // 12
            next_month = 1 + (next_month - 1) % 12
            next_time = time.replace(month=next_month, year=next_year)
            return seconds_passed + (next_time - time).total_seconds()
        elif self.interval_units == TimeUnit.YEARS:
            next_time = time.replace(year=time.year + self.interval)
            return seconds_passed + (next_time - time).total_seconds()
        raise Exception(f"Unknown value for interval_units: {self.interval_units}")

    def start(
        self,
        seconds_passed: float,
        itimestep: int,
        time: Optional[cftime.datetime],
        default_time_reference: Optional[cftime.datetime] = None,
    ):
        if (
            self.interval_units not in time_unit2seconds
            and self.interval_units != TimeUnit.TIMESTEPS
            and time is None
        ):
            raise Exception(
                f"For {self.interval_units} to be used, OutputManager.start should be"
                " called with an actual cftime.datetime object."
            )
        self.add_coordinates()
        self.start_now(seconds_passed, time, default_time_reference or time)
        if self.save_initial:
            self._logger.debug("Saving initial state")
            self.save_now(seconds_passed, time)
        self.next = self._next_time(seconds_passed, itimestep, time)

    def save(
        self, seconds_passed: float, itimestep: int, time: Optional[cftime.datetime]
    ):
        if self.save_on_close_only:
            return

        now = itimestep if self.interval_units == TimeUnit.TIMESTEPS else seconds_passed
        if now >= self.next:
            self._logger.debug("Saving")
            self.save_now(seconds_passed, time)
            self.next = self._next_time(seconds_passed, itimestep, time)

    def close(self, seconds_passed: float, time: Optional[cftime.datetime]):
        if self.save_on_close_only:
            self.save_now(seconds_passed, time)
        self._logger.debug("Closing")
        self.close_now(seconds_passed, time)

    def start_now(
        self,
        seconds_passed: float,
        time: Optional[cftime.datetime],
        default_time_reference: Optional[cftime.datetime],
    ):
        pass

    def save_now(self, seconds_passed: float, time: Optional[cftime.datetime]):
        raise NotImplementedError

    def close_now(self, seconds_passed: float, time: Optional[cftime.datetime]):
        pass


# Note: specific file types cannot be imported before File is defined,
# because that is imported by those types itself.
# If this import statement came earlier, it would cause a circular dependency error.
from . import netcdf
from . import memory


class OutputManager:
    def __init__(
        self,
        fields: Mapping[str, core.Array],
        rank: int,
        logger: Optional[logging.Logger] = None,
    ):
        self.fields = fields
        self.rank = rank
        self._active_files: List[File] = []
        self._startable_files: List[File] = []
        self._stoppable_files: List[File] = []
        self._time_reference = None
        self._logger = logger or logging.getLogger()

    def add_netcdf_file(self, path: str, **kwargs) -> netcdf.NetCDFFile:
        """Add a NetCDF file for output.

        Args:
            path: NetCDF file to write to.
            **kwargs: additional keyword arguments passed to
                :class:`pygetm.output.netcdf.NetCDFFile`
        """
        self._logger.debug(f"Adding NetCDF file {path}")
        file = netcdf.NetCDFFile(
            self.fields, self._logger.getChild(path), path, rank=self.rank, **kwargs
        )
        self._startable_files.append(file)
        return file

    def add_recorder(self, **kwargs) -> memory.MemoryFile:
        """Add an in-memory file to record output.

        Args:
            **kwargs: additional keyword arguments passed to
                :class:`pygetm.output.File`
        """
        self._logger.debug("Adding in-memory file")
        file = memory.MemoryFile(
            self.fields, self._logger.getChild("memfile"), **kwargs
        )
        self._startable_files.append(file)
        return file

    def add_restart(self, path: str, **kwargs) -> netcdf.NetCDFFile:
        """Add a restart file to write to.

        Args:
            path: NetCDF file to write to.
            **kwargs: additional keyword arguments passed to
                :class:`pygetm.output.netcdf.NetCDFFile`

        This is a wrapper around :meth:`add_netcdf_file` that automatically adds all
        arrays with the ``_part_of_state`` flag. By default, the restart file will be
        configured to be written at the end of the simulation, but this can be
        customized by providing argument ``interval`` (see :class:`File`).
        """
        kwargs.setdefault("interval", -1)
        file = self.add_netcdf_file(path, **kwargs)
        for field in self.fields.values():
            if field.attrs.get("_part_of_state"):
                file.request(field)
        return file

    def dump(
        self,
        path: str,
        *fields: Union[str, core.Array],
        seconds_passed: float = 0.0,
        time: Optional[cftime.datetime] = None,
        **kwargs,
    ):
        kwargs.setdefault("time_reference", self._time_reference)
        file = netcdf.NetCDFFile(
            self.fields, self._logger.getChild(path), path, rank=self.rank, **kwargs
        )
        file.request(*fields)
        file.start(seconds_passed, 0, time)
        file.close(seconds_passed, time)

    def start(
        self,
        itimestep: int = 0,
        time: Optional[cftime.datetime] = None,
        default_time_reference: Optional[cftime.datetime] = None,
    ):
        """Tell the output manager that the simulation has started.
        Output files that are registered to save from this time will be opened;
        if they were also registered with ``save_initial`` (see :class:`File`),
        they will have values written to them.

        Args:
            itimestep: index of the first time step
            time: the current time
            default_time_reference: default reference time to use in formats
                that require this (e.g., NetCDF). This is usually the start time of
                this simulation. However, if starting from a restart, it is the
                time reference from that restart, and thus the start time of the
                very first simulation.
        """
        self._time_reference = default_time_reference or time
        for file in self._startable_files:
            if file._start is not None:
                file._start = (file._start - time).total_seconds()
            else:
                file._start = 0.0
            if file._stop is not None:
                file._stop = (file._stop - time).total_seconds()
        self._start_files(0.0, itimestep, time)
        self._stop_files(0.0, time)

    def _start_files(
        self, seconds_passed: float, itimestep: int, time: Optional[cftime.datetime]
    ):
        for i in range(len(self._startable_files) - 1, -1, -1):
            file = self._startable_files[i]
            if file._start <= seconds_passed:
                file.start(seconds_passed, itimestep, time, self._time_reference)
                self._active_files.append(file)
                if file._stop is not None:
                    self._stoppable_files.append(file)
                del self._startable_files[i]

    def _stop_files(self, seconds_passed: float, time: Optional[cftime.datetime]):
        for i in range(len(self._stoppable_files) - 1, -1, -1):
            file = self._stoppable_files[i]
            if file._stop <= seconds_passed:
                self._active_files.remove(file)
                del self._stoppable_files[i]
                file.close(seconds_passed, time)

    def prepare_save(self, *args, macro: bool = True):
        """Begin a new time step. For time-averaged outputs, the current variable
        values will be added to the temporal mean.

        Args:
            macro: whether quantities defined at the macrotimestep are up to date.
                This is the case if the `previous` time step had "macro" processes
                active. It does `not` relate to the processes of the timestep
                that is newly starting!
        """
        for file in self._active_files:
            file.update(macro)

    def save(
        self,
        seconds_passed: float,
        itimestep: int,
        time: Optional[cftime.datetime] = None,
    ):
        """End the current time step and save outputs.

        Args:
            seconds_passed: total number of seconds since the start of the simulation
            itimestep: index of the time step that has just finished
            time: the current time
        """
        for file in self._active_files:
            file.save(seconds_passed, itimestep, time)
        self._start_files(seconds_passed, itimestep, time)
        self._stop_files(seconds_passed, time)

    def close(self, seconds_passed: float, time: Optional[cftime.datetime] = None):
        """Close all open files. Those marked to save at the end of the simulation
        will first have the final variable values written to them

        Args:
            seconds_passed: total number of seconds since the start of the simulation
            time: the current time
        """
        for file in self._active_files:
            file.close(seconds_passed, time)
