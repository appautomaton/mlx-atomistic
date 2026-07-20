"""Run one command with a hard physical-memory ceiling on its process tree."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import time
from collections.abc import Sequence


class _RUsageInfoV0(ctypes.Structure):
    _fields_ = [
        ("ri_uuid", ctypes.c_uint8 * 16),
        ("ri_user_time", ctypes.c_uint64),
        ("ri_system_time", ctypes.c_uint64),
        ("ri_pkg_idle_wkups", ctypes.c_uint64),
        ("ri_interrupt_wkups", ctypes.c_uint64),
        ("ri_pageins", ctypes.c_uint64),
        ("ri_wired_size", ctypes.c_uint64),
        ("ri_resident_size", ctypes.c_uint64),
        ("ri_phys_footprint", ctypes.c_uint64),
        ("ri_proc_start_abstime", ctypes.c_uint64),
        ("ri_proc_exit_abstime", ctypes.c_uint64),
    ]


_LIBPROC = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
_LIBPROC.proc_listchildpids.argtypes = (
    ctypes.c_int,
    ctypes.c_void_p,
    ctypes.c_int,
)
_LIBPROC.proc_listchildpids.restype = ctypes.c_int
_LIBPROC.proc_pid_rusage.argtypes = (
    ctypes.c_int,
    ctypes.c_int,
    ctypes.POINTER(_RUsageInfoV0),
)
_LIBPROC.proc_pid_rusage.restype = ctypes.c_int


def _child_pids(parent: int) -> tuple[int, ...]:
    count = _LIBPROC.proc_listchildpids(parent, None, 0)
    if count <= 0:
        return ()
    buffer = (ctypes.c_int * count)()
    returned = _LIBPROC.proc_listchildpids(
        parent,
        buffer,
        ctypes.sizeof(buffer),
    )
    if returned < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return tuple(int(buffer[index]) for index in range(min(returned, count)))


def _descendants(root: int) -> set[int]:
    selected: set[int] = set()
    pending = [root]
    while pending:
        pid = pending.pop()
        if pid in selected:
            continue
        selected.add(pid)
        pending.extend(_child_pids(pid))
    return selected


def _physical_footprint(pid: int) -> int:
    usage = _RUsageInfoV0()
    if _LIBPROC.proc_pid_rusage(pid, 0, ctypes.byref(usage)) != 0:
        error = ctypes.get_errno()
        if error in {3, 22}:
            return 0
        raise OSError(error, os.strerror(error))
    return int(usage.ri_phys_footprint)


def _process_tree_physical_bytes(root: int) -> tuple[int, set[int]]:
    pids = _descendants(root)
    return sum(_physical_footprint(pid) for pid in pids), pids


def _signal_processes(pids: set[int], signum: signal.Signals) -> None:
    for pid in sorted(pids, reverse=True):
        try:
            os.kill(pid, signum)
        except (PermissionError, ProcessLookupError):
            continue


def _alive_processes(pids: set[int]) -> set[int]:
    alive: set[int] = set()
    for pid in pids:
        try:
            os.kill(pid, 0)
        except (PermissionError, ProcessLookupError):
            continue
        alive.add(pid)
    return alive


def _terminate_processes(pids: set[int]) -> None:
    pids = _alive_processes(pids)
    if not pids:
        return
    _signal_processes(pids, signal.SIGSTOP)
    _signal_processes(pids, signal.SIGTERM)
    _signal_processes(pids, signal.SIGCONT)
    time.sleep(0.1)
    _signal_processes(_alive_processes(pids), signal.SIGKILL)


def run_bounded(
    command: Sequence[str],
    *,
    max_bytes: int,
    poll_seconds: float,
) -> int:
    """Run ``command`` and terminate its process tree above ``max_bytes``."""

    if not command:
        raise ValueError("bounded process command must not be empty")
    if max_bytes <= 0:
        raise ValueError("bounded process max_bytes must be positive")
    if not 0.05 <= poll_seconds <= 5.0:
        raise ValueError("bounded process poll_seconds must lie in [0.05, 5]")
    process = subprocess.Popen(tuple(command), start_new_session=True)
    peak_bytes = 0
    exceeded = False
    observed_pids: set[int] = {process.pid}
    tracked_pids: set[int] = {process.pid}
    try:
        while process.poll() is None:
            physical_bytes, observed_pids = _process_tree_physical_bytes(process.pid)
            tracked_pids.update(observed_pids)
            peak_bytes = max(peak_bytes, physical_bytes)
            if physical_bytes > max_bytes:
                exceeded = True
                _terminate_processes(tracked_pids)
                break
            time.sleep(poll_seconds)
    except BaseException:
        if process.poll() is None:
            try:
                _, observed_pids = _process_tree_physical_bytes(process.pid)
            except (OSError, subprocess.SubprocessError, ValueError):
                observed_pids = {process.pid}
            tracked_pids.update(observed_pids)
            _terminate_processes(tracked_pids)
        raise
    returncode = process.wait()
    orphans = _alive_processes(tracked_pids - {process.pid})
    _terminate_processes(orphans)
    print(
        json.dumps(
            {
                "bounded_process_exceeded": exceeded,
                "bounded_process_limit_bytes": max_bytes,
                "bounded_process_orphans_terminated": len(orphans),
                "bounded_process_peak_physical_bytes": peak_bytes,
                "bounded_process_returncode": returncode,
            },
            sort_keys=True,
        ),
        file=sys.stderr,
        flush=True,
    )
    return 137 if exceeded else returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-bytes", type=int, required=True)
    parser.add_argument("--poll-seconds", type=float, default=0.25)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args()
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    return run_bounded(
        command,
        max_bytes=arguments.max_bytes,
        poll_seconds=arguments.poll_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
