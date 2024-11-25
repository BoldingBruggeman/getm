from typing import Iterable, Mapping, Optional, Tuple, Union, Any, List
import logging
import functools
import pickle
import enum
import weakref

import mpi4py

mpi4py.rc.threads = False

from mpi4py import MPI
import numpy as np
from numpy.typing import ArrayLike, DTypeLike

from . import _pygetm

Waitall = MPI.Request.Waitall
Startall = MPI.Prequest.Startall


def _iterate_rankmap(rankmap):
    for irow in range(rankmap.shape[0]):
        for icol in range(rankmap.shape[1]):
            yield irow, icol, rankmap[irow, icol]


LOGFILE_PREFIX = "getm-"


def get_logger(level=logging.INFO, comm=MPI.COMM_WORLD) -> logging.Logger:
    handlers: List[logging.Handler] = []
    if comm.rank == 0:
        handlers.append(logging.StreamHandler())
    if comm.size > 1:
        logfile = f"{LOGFILE_PREFIX}{comm.rank:04}.log"
        file_handler = logging.FileHandler(logfile, mode="w")
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
        handlers.append(file_handler)
    logging.basicConfig(level=level, handlers=handlers, force=True)
    logger = logging.getLogger()
    logger.info(f"pygetm {_pygetm.get_version()}")
    return logger


def mpi4py_autofree(obj):
    # Ensure underlying MPI memory is freed when obj is garbage-collected
    # https://github.com/mpi4py/mpi4py/issues/32
    def callfree(fromhandle, handle):
        obj = fromhandle(handle)
        if obj:
            obj.Free()

    weakref.finalize(obj, callfree, obj.f2py, obj.py2f())
    return obj


class Tiling:
    @staticmethod
    def autodetect(
        mask: ArrayLike,
        *,
        ncpus: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        comm: Optional[MPI.Comm] = None,
        max_protrude: float = 0.5,
        **kwargs,
    ) -> "Tiling":
        """Auto-detect the optimal subdomain division and return it as
        :class:`Tiling` object

        Args:
            mask: mask for the global domain (0: masked, non-zero: active)
            ncpus: number of cores to use (default: size of MPI communicator)
            logger: logger to use to write diagnostic messages
            max_protrude: maximum fraction (0-1) of a subdomain that can protrude out of
                the global domain
            **kwargs: additional keyword arguments to pass to :class:`Tiling`
        """

        solution = find_optimal_divison(
            mask, ncpus=ncpus, comm=comm, max_protrude=max_protrude, logger=logger
        )
        if not solution:
            raise Exception("No suitable subdomain decompositon found")
        counts = solution["map"]
        rank_map = np.full(counts.shape, -1, dtype=int)
        rank = 0
        for irow, icol, count in _iterate_rankmap(counts):
            if count > 0:
                rank_map[irow, icol] = rank
                rank += 1
        tiling = Tiling(map=rank_map, ncpus=solution["ncpus"], comm=comm, **kwargs)
        tiling.set_extent(
            mask.shape[1],
            mask.shape[0],
            solution["nx"],
            solution["ny"],
            yoffset_global=solution["yoffset"],
            xoffset_global=solution["xoffset"],
        )
        return tiling

    @staticmethod
    def load(path: str) -> "Tiling":
        """Create a :class:`Tiling` object from information in a pickle file
        that was produced with :meth:`Tiling.dump`
        """
        with open(path, "rb") as f:
            (
                map,
                periodic_x,
                periodic_y,
                nx_glob,
                ny_glob,
                nx_sub,
                ny_sub,
                xoffset_global,
                yoffset_global,
            ) = pickle.load(f)
        tiling = Tiling(
            map=map,
            periodic_x=periodic_x,
            periodic_y=periodic_y,
            ncpus=(map != -1).sum(),
        )
        tiling.set_extent(
            nx_glob, ny_glob, nx_sub, ny_sub, xoffset_global, yoffset_global
        )
        return tiling

    def __init__(
        self,
        nrow: Optional[int] = None,
        ncol: Optional[int] = None,
        map: Optional[np.ndarray] = None,
        comm: Optional[MPI.Comm] = None,
        periodic_x: bool = False,
        periodic_y: bool = False,
        ncpus: Optional[int] = None,
    ):
        self.comm = comm or mpi4py_autofree(MPI.COMM_WORLD.Dup())
        self.rank: int = self.comm.rank
        self.n = ncpus if ncpus is not None else self.comm.size

        if nrow is None and ncol is not None:
            nrow = self.n // ncol
        if ncol is None and nrow is not None:
            ncol = self.n // nrow

        if nrow is None and ncol is None:
            assert map is not None, (
                "If the number of rows and columns in the subdomain decomposition is"
                " not provided, the rank map must be provided instead."
            )
            self.map = map
        else:
            self.map = np.arange(nrow * ncol).reshape(nrow, ncol)
        self.nrow, self.ncol = self.map.shape

        self.n_neigbors = 0

        def find_neighbor(ioffset: int, joffset: int) -> int:
            if self.irow is None:
                return -1
            i = self.irow + ioffset
            j = self.icol + joffset
            if periodic_x:
                j = j % self.ncol
            if periodic_y:
                i = i % self.nrow
            if i >= 0 and i < self.nrow and j >= 0 and j < self.ncol:
                self.n_neigbors += 1
                return int(self.map[i, j])
            return -1

        if self.nactive != self.n:
            raise Exception(
                f"number of active subdomains ({self.nactive}) does not match assigned"
                f" number of cores ({self.n}). Map: {self.map}"
            )

        self.periodic_x = periodic_x
        self.periodic_y = periodic_y

        # Determine own row and column in subdomain decomposition
        for self.irow, self.icol, r in _iterate_rankmap(self.map):
            if r == self.rank:
                break
        else:
            self.irow, self.icol = None, None

        self.top = find_neighbor(+1, 0)
        self.bottom = find_neighbor(-1, 0)
        self.left = find_neighbor(0, -1)
        self.right = find_neighbor(0, +1)
        self.topleft = find_neighbor(+1, -1)
        self.topright = find_neighbor(+1, +1)
        self.bottomleft = find_neighbor(-1, -1)
        self.bottomright = find_neighbor(-1, +1)

        self._caches = {}
        self.nx_glob = None

    @property
    def nactive(self) -> int:
        return (self.map[:, :] != -1).sum()

    def dump(self, path: str):
        """Save all information about the subdomain division to a pickle file
        from which a :class:`Tiling` objetc can be recreated with :meth:`Tiling.load`

        Args:
            path: file to save information to
        """
        with open(path, "wb") as f:
            pickle.dump(
                (
                    self.map,
                    self.periodic_x,
                    self.periodic_y,
                    self.nx_glob,
                    self.ny_glob,
                    self.nx_sub,
                    self.ny_sub,
                    self.xoffset_global,
                    self.yoffset_global,
                ),
                f,
            )

    def set_extent(
        self,
        nx_glob: int,
        ny_glob: int,
        nx_sub: Optional[int] = None,
        ny_sub: Optional[int] = None,
        xoffset_global: int = 0,
        yoffset_global: int = 0,
    ):
        """Set extent of the global domain and the subdomain, and the offsets of the
        very first subdomain.

        Args:
            nx_glob: x extent of the global domain
            ny_glob: y extent of the global domain
            nx_sub: x extent of a subdomain
            ny_sub: y extent of a subdomain
            xoffset_global: x offset of left-most subdomains
            yoffset_global: y offset of bottom-most subdomains
        """
        if nx_sub is None:
            nx_sub = int(np.ceil(nx_glob / self.ncol))
        if ny_sub is None:
            ny_sub = int(np.ceil(ny_glob / self.nrow))

        assert self.nx_glob is None, "Domain extent has already been set."
        assert isinstance(nx_glob, (int, np.integer))
        assert isinstance(ny_glob, (int, np.integer))
        assert isinstance(nx_sub, (int, np.integer))
        assert isinstance(ny_sub, (int, np.integer))
        assert isinstance(xoffset_global, (int, np.integer))
        assert isinstance(yoffset_global, (int, np.integer))

        self.nx_glob, self.ny_glob = nx_glob, ny_glob
        self.nx_sub, self.ny_sub = nx_sub, ny_sub
        self.xoffset_global = xoffset_global
        self.yoffset_global = yoffset_global
        if self.irow is not None:
            self.xoffset = self.icol * self.nx_sub + self.xoffset_global
            self.yoffset = self.irow * self.ny_sub + self.yoffset_global
        else:
            # this is a dummy process outside the computation domain
            self.xoffset = -self.nx_sub
            self.yoffset = -self.ny_sub

    def report(self, logger: logging.Logger):
        """Write information about the subdomain decompostion to the log.
        Log messages are suppressed if the decompositon only has one subdomain.
        """
        if self.nrow > 1 or self.ncol > 1:
            logger.info(
                f"Using subdomain decomposition {self.nrow} x {self.ncol}"
                f" ({self.nactive} active nodes)"
            )
            logger.info(
                f"Global domain shape {self.nx_glob} x {self.ny_glob},"
                f" subdomain shape {self.nx_sub} x {self.ny_sub},"
                f" global offsets x={self.xoffset_global}, y={self.yoffset_global}"
            )
            logger.info(
                f"I am rank {self.rank} at subdomain row {self.irow}, "
                f" column {self.icol}, with offset x={self.xoffset}, y={self.yoffset}"
            )

    def __bool__(self) -> bool:
        """Return True if the curent subdomain has any neighbors, False otherwise."""
        return self.n_neigbors > 0

    def wrap(self, *args, **kwargs) -> "DistributedArray":
        return DistributedArray(self, *args, **kwargs)

    def subdomain2slices(
        self,
        irow: Optional[int] = None,
        icol: Optional[int] = None,
        halox_sub: int = 0,
        haloy_sub: int = 0,
        halox_glob: int = 0,
        haloy_glob: int = 0,
        scale: int = 1,
        share: int = 0,
        exclude_halos: bool = True,
        exclude_global_halos: bool = False,
    ) -> Tuple[
        Tuple[Any, slice, slice],
        Tuple[Any, slice, slice],
        Tuple[int, int],
        Tuple[int, int],
    ]:
        """Determine the activate extent of a subdomain in terms of the slices
        spanned in subdomain arrays and in global arrays

        Args:
            irow: row of the subdomain (default: current)
            icol: column of the subdomain (default: current)
            halo_sub: width of the halos in the subdomain array to be sliced
            halo_glob: width of the halos in the global array to be sliced
            scale: the extent of subdomain array relative to the subdomain extent
                provided to :meth:`set_extent` (for instance, 2 for supergrid arrays)
            share: the number of points shared by subdomains (for instance, 1 for
                supergrid arrays and X grids)
            exclude_halos: whether to excludes the halos of the subdomain from the slice
            exclude_global_halos: whether to excludes the halos of the global domain
                from the slice

        Returns:
            Tuple with the local slices, global slices, extent of a subdomain array, and
            extent of a global array. The extents always include halos and shared points
        """
        if irow is None:
            irow = self.irow
        if icol is None:
            icol = self.icol

        assert isinstance(share, int)
        assert isinstance(scale, int)
        assert isinstance(halox_sub, int)
        assert isinstance(haloy_sub, int)
        assert isinstance(halox_glob, int)
        assert isinstance(haloy_glob, int)

        # Shapes of local and global arrays (including halos and inactive strips)
        local_shape = (
            scale * self.ny_sub + 2 * haloy_sub + share,
            scale * self.nx_sub + 2 * halox_sub + share,
        )
        global_shape = (
            scale * self.ny_glob + 2 * haloy_glob + share,
            scale * self.nx_glob + 2 * halox_glob + share,
        )

        if irow is None:
            local_slice = global_slice = (Ellipsis, slice(0, 0), slice(0, 0))
            return local_slice, global_slice, local_shape, global_shape

        assert self.map[irow, icol] >= 0

        # Global start and stop of local subdomain (no halos yet)
        xstart_glob = scale * (icol * self.nx_sub + self.xoffset_global)
        xstop_glob = scale * ((icol + 1) * self.nx_sub + self.xoffset_global)
        ystart_glob = scale * (irow * self.ny_sub + self.yoffset_global)
        ystop_glob = scale * ((irow + 1) * self.ny_sub + self.yoffset_global)

        # Adjust for halos and any overlap ("share")
        xstart_glob += -halox_sub + halox_glob
        xstop_glob += halox_sub + halox_glob + share
        ystart_glob += -haloy_sub + haloy_glob
        ystop_glob += haloy_sub + haloy_glob + share

        # Local start and stop
        xstart_loc = 0
        xstop_loc = xstop_glob - xstart_glob
        ystart_loc = 0
        ystop_loc = ystop_glob - ystart_glob

        # Calculate offsets based on limits of the global domain
        extra_offsetx = halox_sub if exclude_halos else 0
        extra_offsety = haloy_sub if exclude_halos else 0
        marginx_glob = halox_glob if exclude_global_halos else 0
        marginy_glob = haloy_glob if exclude_global_halos else 0
        xstart_offset = max(xstart_glob + extra_offsetx, marginx_glob) - xstart_glob
        xstop_offset = (
            min(xstop_glob - extra_offsetx, global_shape[1] - marginx_glob) - xstop_glob
        )
        ystart_offset = max(ystart_glob + extra_offsety, marginy_glob) - ystart_glob
        ystop_offset = (
            min(ystop_glob - extra_offsety, global_shape[0] - marginy_glob) - ystop_glob
        )
        assert xstart_offset >= 0 and xstop_offset <= 0
        assert ystart_offset >= 0 and ystop_offset <= 0

        global_slice = (
            Ellipsis,
            slice(ystart_glob + ystart_offset, ystop_glob + ystop_offset),
            slice(xstart_glob + xstart_offset, xstop_glob + xstop_offset),
        )
        local_slice = (
            Ellipsis,
            slice(ystart_loc + ystart_offset, ystop_loc + ystop_offset),
            slice(xstart_loc + xstart_offset, xstop_loc + xstop_offset),
        )
        return local_slice, global_slice, local_shape, global_shape

    def describe(self):
        """Print neighbours of the current subdomain"""

        def p(x):
            return "-" if x is None else x

        print(
            "{:^3}  {:^3}  {:^3}".format(p(self.topleft), p(self.top), p(self.topright))
        )
        print("{:^3} [{:^3}] {:^3}".format(p(self.left), self.rank, p(self.right)))
        print(
            "{:^3}  {:^3}  {:^3}".format(
                p(self.bottomleft), p(self.bottom), p(self.bottomright)
            )
        )

    def plot(self, ax=None, background: Optional[np.ndarray] = None, cmap="viridis"):
        """Plot the current subdomain division

        Args:
            ax: :class:`matplotlib.axes.Axes` to plot into. If not provided, a
                new :class:`matplotlib.figure.Figure` with single axes will be created.
            background: field to use as background for the subdomain division. If
                provided, it must have the extent of the global domain. If not provided
                a single filled rectabgle covering the global domain will be used as
                background.
        """
        import matplotlib.patches
        import matplotlib.cm

        x = self.xoffset_global + np.arange(self.ncol + 1) * self.nx_sub
        y = self.yoffset_global + np.arange(self.nrow + 1) * self.ny_sub
        if ax is None:
            import matplotlib.pyplot

            fig, ax = matplotlib.pyplot.subplots()

        ax.set_facecolor((0.8, 0.8, 0.8))

        # Background showing global domain
        rect = (0, 0), self.nx_glob, self.ny_glob
        if background is not None:
            # Background field (e.g., bathymetry) provided
            assert background.shape == (self.ny_glob, self.nx_glob), (
                f"Argument background has incorrrect shape {background.shape}."
                f" Expected ({self.ny_glob}, {self.nx_glob})"
            )
            ax.add_artist(matplotlib.patches.Rectangle(*rect, ec="None", fc="w"))
            ax.pcolormesh(
                np.arange(background.shape[1] + 1),
                np.arange(background.shape[0] + 1),
                background,
                alpha=0.5,
                cmap=cmap,
            )
            fc = "None"
        else:
            # Background field not provided:
            # just show a filled rectangle covering the global domain
            fc = "C0"

        ax.add_artist(matplotlib.patches.Rectangle(*rect, ec="k", fc=fc, lw=0.4))

        # Boundaries between subdomains
        ax.pcolormesh(x, y, np.empty((self.nrow, self.ncol)), ec="k", fc="none")

        # Rank of each subdomain (cross for subdomains that are not used, e.g. land)
        for i in range(self.nrow):
            for j in range(self.ncol):
                if self.map[i, j] >= 0:
                    ax.text(
                        0.5 * (x[j] + x[j + 1]),
                        0.5 * (y[i] + y[i + 1]),
                        str(self.map[i, j]),
                        horizontalalignment="center",
                        verticalalignment="center",
                    )
                else:
                    ax.plot([x[j], x[j + 1]], [y[i], y[i + 1]], "-k")
                    ax.plot([x[j], x[j + 1]], [y[i + 1], y[i]], "-k")

        return ax

    def _get_work_array(
        self,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        fill_value: Optional[float] = None,
    ) -> np.ndarray:
        key = (shape, dtype, np.asarray(fill_value).item())
        if key not in self._caches:
            self._caches[key] = np.empty(shape, dtype)
            if fill_value is not None:
                self._caches[key].fill(fill_value)
        return self._caches[key]

    def reduce(self, field: np.ndarray, op=MPI.SUM) -> Optional[np.ndarray]:
        out = None if self.rank != 0 else np.empty_like(field)
        self.comm.Reduce(field, out, op)
        return out

    def allreduce(self, field: np.ndarray, op=MPI.SUM) -> np.ndarray:
        out = np.empty_like(field)
        self.comm.Allreduce(field, out, op)
        return out


@enum.unique
class Neighbor(enum.IntEnum):
    # Specific neighbors
    BOTTOMLEFT = 1
    BOTTOM = 2
    BOTTOMRIGHT = 3
    LEFT = 4
    RIGHT = 5
    TOPLEFT = 6
    TOP = 7
    TOPRIGHT = 8

    # Groups of neighbors (for update_halos command)
    ALL = 0
    TOP_AND_BOTTOM = 9
    LEFT_AND_RIGHT = 10
    TOP_AND_RIGHT = 11  # for T to U/V grid interpolation
    LEFT_AND_RIGHT_AND_TOP_AND_BOTTOM = 12  # for T to U/V grid 2nd order interpolation


GROUP2PARTS = {
    Neighbor.TOP_AND_BOTTOM: (Neighbor.TOP, Neighbor.BOTTOM),
    Neighbor.LEFT_AND_RIGHT: (Neighbor.LEFT, Neighbor.RIGHT),
    Neighbor.TOP_AND_RIGHT: (Neighbor.TOP, Neighbor.RIGHT),
    Neighbor.LEFT_AND_RIGHT_AND_TOP_AND_BOTTOM: (
        Neighbor.LEFT,
        Neighbor.RIGHT,
        Neighbor.TOP,
        Neighbor.BOTTOM,
    ),
}


class DistributedArray:
    __slots__ = ["rank", "group2task", "halo2name"]

    def __init__(
        self,
        tiling: Tiling,
        field: np.ndarray,
        halox: int,
        haloy: int,
        overlap: int = 0,
        share_caches: bool = False,
    ):
        self.rank = tiling.rank
        self.group2task: List[
            Tuple[
                List[MPI.Prequest],
                List[MPI.Prequest],
                List[Tuple[np.ndarray, np.ndarray]],
                List[Tuple[np.ndarray, np.ndarray]],
            ]
        ] = [([], [], [], []) for _ in range(max(Neighbor) + 1)]
        self.halo2name = {}

        key = (field.shape, halox, haloy, overlap, field.dtype)
        caches = None if not share_caches else tiling._caches.get(key)
        owncaches = []

        def add_task(
            recvtag: Neighbor, sendtag: Neighbor, outer: np.ndarray, inner: np.ndarray
        ):
            name = recvtag.name.lower()
            neighbor = getattr(tiling, name)
            assert isinstance(neighbor, int), (
                f"Wrong type for neighbor {name}:"
                f" {neighbor} (type {type(neighbor)})"
            )
            assert inner.shape == outer.shape, f"{inner.shape!r} vs {outer.shape!r}"
            if neighbor != -1 and inner.size > 0:
                if caches:
                    inner_cache, outer_cache = caches[len(owncaches)]
                    assert inner.shape == inner_cache.shape
                    assert outer.shape == outer_cache.shape
                else:
                    inner_cache, outer_cache = (
                        np.empty_like(inner),
                        np.empty_like(outer),
                    )
                owncaches.append((inner_cache, outer_cache))
                send_req = mpi4py_autofree(
                    tiling.comm.Send_init(inner_cache, neighbor, sendtag)
                )
                recv_req = mpi4py_autofree(
                    tiling.comm.Recv_init(outer_cache, neighbor, recvtag)
                )

                self.halo2name[id(outer)] = name
                for group in [Neighbor.ALL, sendtag] + [
                    group for (group, parts) in GROUP2PARTS.items() if sendtag in parts
                ]:
                    send_reqs, recv_reqs, send_data, recv_data = self.group2task[group]
                    send_reqs.append(send_req)
                    send_data.append((inner, inner_cache))
                for group in [Neighbor.ALL, recvtag] + [
                    group for (group, parts) in GROUP2PARTS.items() if recvtag in parts
                ]:
                    send_reqs, recv_reqs, send_data, recv_data = self.group2task[group]
                    recv_reqs.append(recv_req)
                    recv_data.append((outer, outer_cache))

        in_startx = halox + overlap
        in_starty = haloy + overlap
        in_stopx = in_startx + halox
        in_stopy = in_starty + haloy
        ny, nx = field.shape[-2:]
        add_task(
            Neighbor.BOTTOMLEFT,
            Neighbor.TOPRIGHT,
            field[..., :haloy, :halox],
            field[..., in_starty:in_stopy, in_startx:in_stopx],
        )
        add_task(
            Neighbor.BOTTOM,
            Neighbor.TOP,
            field[..., :haloy, halox : nx - halox],
            field[..., in_starty:in_stopy, halox : nx - halox],
        )
        add_task(
            Neighbor.BOTTOMRIGHT,
            Neighbor.TOPLEFT,
            field[..., :haloy, nx - halox :],
            field[..., in_starty:in_stopy, nx - in_stopx : nx - in_startx],
        )
        add_task(
            Neighbor.LEFT,
            Neighbor.RIGHT,
            field[..., haloy : ny - haloy, :halox],
            field[..., haloy : ny - haloy, in_startx:in_stopx],
        )
        add_task(
            Neighbor.RIGHT,
            Neighbor.LEFT,
            field[..., haloy : ny - haloy, nx - halox :],
            field[..., haloy : ny - haloy, nx - in_stopx : nx - in_startx],
        )
        add_task(
            Neighbor.TOPLEFT,
            Neighbor.BOTTOMRIGHT,
            field[..., ny - haloy :, :halox],
            field[..., ny - in_stopy : ny - in_starty, in_startx:in_stopx],
        )
        add_task(
            Neighbor.TOP,
            Neighbor.BOTTOM,
            field[..., ny - haloy :, halox : nx - halox],
            field[..., ny - in_stopy : ny - in_starty, halox : nx - halox],
        )
        add_task(
            Neighbor.TOPRIGHT,
            Neighbor.BOTTOMLEFT,
            field[..., ny - haloy :, nx - halox :],
            field[..., ny - in_stopy : ny - in_starty, nx - in_stopx : nx - in_startx],
        )
        if caches is None and share_caches:
            tiling._caches[key] = owncaches

    def update_halos(self, group: Neighbor = Neighbor.ALL):
        send_reqs, recv_reqs, send_data, recv_data = self.group2task[group]
        Startall(recv_reqs)
        for inner, cache in send_data:
            cache[...] = inner
        Startall(send_reqs)
        Waitall(recv_reqs)
        for outer, cache in recv_data:
            outer[...] = cache
        Waitall(send_reqs)

    def update_halos_start(self, group: Neighbor = Neighbor.ALL):
        send_reqs, recv_reqs, send_data, _ = self.group2task[group]
        for inner, cache in send_data:
            cache[...] = inner
        Startall(send_reqs + recv_reqs)

    def update_halos_finish(self, group: Neighbor = Neighbor.ALL):
        send_reqs, recv_reqs, _, recv_data = self.group2task[group]
        Waitall(send_reqs + recv_reqs)
        for outer, cache in recv_data:
            outer[...] = cache

    def compare_halos(self, group: Neighbor = Neighbor.ALL) -> bool:
        send_reqs, recv_reqs, send_data, recv_data = self.group2task[group]
        Startall(recv_reqs)
        for inner, cache in send_data:
            cache[...] = inner
        Startall(send_reqs)
        Waitall(recv_reqs)
        match = True
        for outer, cache in recv_data:
            if not np.array_equal(outer, cache, equal_nan=True):
                delta = outer - cache
                print(
                    f"Rank {self.rank}: mismatch in {self.halo2name[id(outer)]} halo!"
                    f" Maximum absolute difference: {np.abs(delta).max()}."
                    f" Current {outer} vs. received {cache}"
                )
                match = False
        Waitall(send_reqs)
        return match


class Gather:
    def __init__(
        self,
        tiling: Tiling,
        shape: Tuple[int, ...],
        dtype: DTypeLike,
        fill_value=None,
        root: int = 0,
    ):
        self.comm = tiling.comm
        self.root = root
        self.recvbuf = None
        self.fill_value = np.nan if fill_value is None else fill_value
        self.dtype = dtype
        if tiling.rank == self.root:
            self.buffers = []
            self.recvbuf = tiling._get_work_array((tiling.comm.size,) + shape, dtype)
            for irow, icol, rank in _iterate_rankmap(tiling.map):
                if rank >= 0:
                    (
                        local_slice,
                        global_slice,
                        local_shape,
                        global_shape,
                    ) = tiling.subdomain2slices(irow, icol, exclude_halos=True)
                    assert shape[-2:] == local_shape, (
                        f"Field shape {shape[-2:]} differs"
                        f" from expected {local_shape}"
                    )
                    self.buffers.append(
                        (self.recvbuf[(rank,) + local_slice], global_slice)
                    )
            self.global_shape = global_shape

    def __call__(
        self, field: np.ndarray, out: Optional[np.ndarray] = None, slice_spec=()
    ) -> Optional[np.ndarray]:
        sendbuf = np.ascontiguousarray(field, self.dtype)
        self.comm.Gather(sendbuf, self.recvbuf, root=self.root)
        if self.recvbuf is not None:
            if out is None:
                out = np.empty(
                    self.recvbuf.shape[1:-2] + self.global_shape, dtype=self.dtype
                )
                if self.fill_value is not None:
                    out.fill(self.fill_value)
            assert out.shape[-2:] == self.global_shape, (
                f"Global shape {out.shape[-2:]} differs"
                f" from expected {self.global_shape}"
            )
            for source, global_slice in self.buffers:
                out[slice_spec + global_slice] = source
            return out


class Scatter:
    def __init__(
        self,
        tiling: Tiling,
        field: np.ndarray,
        halox: int,
        haloy: int,
        share: int = 0,
        fill_value=0.0,
        scale: int = 1,
        exclude_halos: bool = False,
        root: int = 0,
    ):
        self.field = field
        self.recvbuf = np.ascontiguousarray(field)
        self.sendbuf = None
        if tiling.comm.rank == root:
            self.buffers = []
            self.sendbuf = tiling._get_work_array(
                (tiling.comm.size,) + self.recvbuf.shape,
                self.recvbuf.dtype,
                fill_value=fill_value,
            )
            for irow, icol, rank in _iterate_rankmap(tiling.map):
                if rank >= 0:
                    (
                        local_slice,
                        global_slice,
                        local_shape,
                        global_shape,
                    ) = tiling.subdomain2slices(
                        irow,
                        icol,
                        halox_sub=halox,
                        haloy_sub=haloy,
                        halox_glob=0,
                        haloy_glob=0,
                        share=share,
                        scale=scale,
                        exclude_halos=exclude_halos,
                    )
                    assert field.shape[-2:] == local_shape, (
                        f"Field shape {field.shape[-2:]} differs"
                        f" from expected {local_shape}"
                    )
                    self.buffers.append(
                        (self.sendbuf[(rank,) + local_slice], global_slice)
                    )
            self.global_shape = global_shape
        self.mpi_scatter = functools.partial(
            tiling.comm.Scatter, self.sendbuf, self.recvbuf, root=root
        )

    def __call__(self, global_data: Optional[np.ndarray]):
        if self.sendbuf is not None:
            # we are root and have to send the global field
            assert global_data.shape[-2:] == self.global_shape, (
                f"Global shape {global_data.shape[-2:]} differs"
                f" from expected {self.global_shape}"
            )
            for sendbuf, global_slice in self.buffers:
                sendbuf[...] = global_data[global_slice]
        self.mpi_scatter()
        if self.recvbuf is not self.field:
            self.field[...] = self.recvbuf


def find_optimal_divison(
    mask: ArrayLike,
    *,
    ncpus: Optional[int] = None,
    max_aspect_ratio: int = 2,
    weight_unmasked: int = 2,
    weight_any: int = 1,
    weight_halo: int = 10,
    max_protrude: float = 0.5,
    logger: Optional[logging.Logger] = None,
    comm: Optional[MPI.Comm] = None,
) -> Optional[Mapping[str, Any]]:
    if ncpus is None:
        ncpus = (comm or MPI.COMM_WORLD).size

    ny, nx = mask.shape

    # If we only have 1 CPU, just use the full domain
    if ncpus == 1:
        return {
            "ncpus": 1,
            "nx": nx,
            "ny": ny,
            "xoffset": 0,
            "yoffset": 0,
            "cost": 0,
            "map": np.ones((1, 1), dtype=np.intc),
        }

    # If we have a 1D grid with only unmasked (water) points,
    # return simple equal subdomian division
    if ny == 1 and np.all(mask):
        return {
            "ncpus": ncpus,
            "nx": int(np.ceil(nx / ncpus)),
            "ny": 1,
            "xoffset": 0,
            "yoffset": 0,
            "cost": 0,
            "map": np.ones((1, ncpus), dtype=np.intc),
        }

    comm = comm or mpi4py_autofree(MPI.COMM_WORLD.Dup())

    # Determine mask extent excluding any outer fully masked strips
    mask = np.asarray(mask)
    (mask_x,) = mask.any(axis=0).nonzero()
    (mask_y,) = mask.any(axis=1).nonzero()
    imin, imax = mask_x[0], mask_x[-1]
    jmin, jmax = mask_y[0], mask_y[-1]

    # Convert to contiguous mask that the cython code expects
    mask = np.ascontiguousarray(mask[jmin : jmax + 1, imin : imax + 1], dtype=np.intc)

    # Determine potential number of subdomain combinations
    nx_ny_combos = []
    for ny_sub in range(4, ny + 1):
        for nx_sub in range(
            max(4, ny_sub // max_aspect_ratio),
            min(max_aspect_ratio * ny_sub, nx + 1),
        ):
            nx_ny_combos.append((nx_sub, ny_sub))
    nx_ny_combos = np.array(nx_ny_combos, dtype=int)

    if logger:
        logger.info(
            (
                "Determining optimal subdomain decomposition of global domain of "
                f"{nx} x {ny} ({(mask != 0).sum()} active cells)"
                f" for {ncpus} cores"
            )
        )
        logger.info(f"Trying {nx_ny_combos.shape[0]} possible subdomain sizes")

    cost, solution = None, None
    for nx_sub, ny_sub in nx_ny_combos[comm.rank :: comm.size]:
        current_solution = _pygetm.find_subdiv_solutions(
            mask,
            nx_sub,
            ny_sub,
            ncpus,
            weight_unmasked,
            weight_any,
            weight_halo,
            max_protrude,
        )
        if current_solution:
            xoffset, yoffset, current_cost, submap = current_solution
            if cost is None or current_cost < cost:
                solution = {
                    "ncpus": ncpus,
                    "nx": nx_sub,
                    "ny": ny_sub,
                    "xoffset": imin + xoffset,
                    "yoffset": jmin + yoffset,
                    "cost": current_cost,
                    "map": submap,
                }
                cost = current_cost
    solutions = comm.allgather(solution)
    solution = min(filter(None, solutions), key=lambda x: x["cost"], default=None)
    if logger and solution:
        logger.info(f"Optimal subdomain decomposition: {solution}")
    return solution


def find_all_optimal_divisons(
    mask: ArrayLike, max_ncpus: Union[int, Iterable[int]], **kwargs
) -> Optional[Mapping[str, Any]]:
    if isinstance(max_ncpus, int):
        max_ncpus = np.arange(1, max_ncpus + 1)
    for ncpus in max_ncpus:
        solution = find_optimal_divison(mask, ncpus=ncpus, **kwargs)
        if solution:
            print(
                (
                    "{ncpus} cores, optimal subdomain size: {nx} x {ny}, {0} land-only"
                    "subdomains were dropped, max cost per core: {cost}"
                ).format((solution["map"] == 0).sum(), **solution)
            )


def test_scaling_command():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "setup_script", help="path to Python script that starts the simulation"
    )
    parser.add_argument(
        "--nmin", type=int, default=1, help="minimum number of CPUs to test with"
    )
    parser.add_argument("--nmax", type=int, help="maximum number of CPUs to test with")
    parser.add_argument(
        "--plot", action="store_true", help="whether to create scaling plot"
    )
    parser.add_argument("--out", help="path to write scaling figure to")
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        help="Path of NetCDF file to compare across simulations",
    )
    args, leftover_args = parser.parse_known_args()
    if leftover_args:
        print(f"Arguments {leftover_args} will be passed to {args.setup_script}")
    test_scaling(
        args.setup_script,
        args.nmax,
        args.nmin,
        extra_args=leftover_args,
        plot=args.plot,
        out=args.out,
        compare=args.compare,
    )


def test_scaling(
    setup_script: str,
    nmax: Optional[int] = None,
    nmin: int = 1,
    extra_args: Iterable[str] = [],
    plot: bool = True,
    out: Optional[str] = None,
    compare: Optional[Iterable[str]] = None,
):
    import subprocess
    import sys
    import os
    import re
    import tempfile
    import shutil
    from .util import compare_nc

    if nmax is None:
        nmax = os.cpu_count()
    ncpus = []
    durations = []
    success = True
    refnames = []
    ntest = list(range(nmin, nmax + 1))
    if nmin > 1 and compare:
        ntest.insert(0, 1)
    for n in ntest:
        print(
            f"Running {os.path.basename(setup_script)} with {n} CPUs... ",
            end="",
            flush=True,
        )
        p = subprocess.run(
            ["mpiexec", "-n", str(n), sys.executable, setup_script] + extra_args,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        if p.returncode != 0:
            success = False
            if n == ntest[0]:
                print(
                    f"First simulation failed with return code {p.returncode}."
                    "Quitting."
                    "Last result:"
                )
                print(p.stdout)
                print()
                break
            log_path = f"scaling-{n:03}.log"
            with open(log_path, "w") as f:
                f.write(p.stdout)
            print(f"FAILED with return code {p.returncode}, log written to {log_path}")
            continue
        m = re.search("Time spent in main loop: ([\\d\\.]+) s", p.stdout)
        duration = float(m.group(1))
        print(f"{duration:.3} s in main loop")
        ncpus.append(n)
        durations.append(duration)
        for i, path in enumerate(compare):
            if n == 1:
                print(
                    f"  copying reference {path} to temporary file... ",
                    end="",
                    flush=True,
                )
                with open(path, "rb") as fin, tempfile.NamedTemporaryFile(
                    suffix=".nc", delete=False
                ) as fout:
                    shutil.copyfileobj(fin, fout)
                    refnames.append(fout.name)
                print("done.")
            else:
                print(
                    f"  comparing {path} with reference produced with 1 CPUs... ",
                    flush=True,
                )
                if not compare_nc.compare(path, refnames[i]):
                    success = False

    for refname in refnames:
        print(f"Deleting reference result {refname}... ", end="")
        os.remove(refname)
        print("done.")

    if plot and success:
        from matplotlib import pyplot

        fig, ax = pyplot.subplots()
        ax.plot(ncpus, durations, ".")
        ax.grid()
        ax.set_xlabel("number of CPUs")
        ax.set_ylabel(f"duration of {os.path.basename(setup_script)}")
        ax.set_ylim(0, None)
        if out:
            fig.savefig(out, dpi=300)
        else:
            pyplot.show()

    sys.exit(0 if success else 1)
