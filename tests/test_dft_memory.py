from __future__ import annotations

import pytest

from mlx_atomistic.dft import _memory


def test_bounded_dft_allocator_is_a_noop_on_cpu(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(_memory.mx, "default_device", lambda: "Device(cpu, 0)")
    monkeypatch.setattr(
        _memory.mx,
        "set_memory_limit",
        lambda value: calls.append(value),
    )

    with _memory._bounded_dft_allocator():
        pass

    assert calls == []


def test_bounded_dft_allocator_restores_caller_limits(monkeypatch):
    state = {"memory": 70_000_000_000, "cache": 12_000_000_000}
    calls: list[tuple[str, int]] = []
    monkeypatch.setattr(_memory.mx, "default_device", lambda: "Device(gpu, 0)")
    monkeypatch.setattr(_memory.mx, "get_active_memory", lambda: 0)

    def set_limit(name, value):
        previous = state[name]
        state[name] = value
        calls.append((name, value))
        return previous

    monkeypatch.setattr(
        _memory.mx,
        "set_memory_limit",
        lambda value: set_limit("memory", value),
    )
    monkeypatch.setattr(
        _memory.mx,
        "set_cache_limit",
        lambda value: set_limit("cache", value),
    )

    with _memory._bounded_dft_allocator():
        assert state == {"memory": 40_000_000_000, "cache": 4_000_000_000}

    assert state == {"memory": 70_000_000_000, "cache": 12_000_000_000}
    assert calls == [
        ("memory", 40_000_000_000),
        ("cache", 4_000_000_000),
        ("cache", 12_000_000_000),
        ("memory", 70_000_000_000),
    ]


def test_bounded_dft_allocator_restores_limits_after_failure(monkeypatch):
    state = {"memory": 70_000_000_000, "cache": 12_000_000_000}
    monkeypatch.setattr(_memory.mx, "default_device", lambda: "Device(gpu, 0)")
    monkeypatch.setattr(_memory.mx, "get_active_memory", lambda: 0)

    def set_limit(name, value):
        previous = state[name]
        state[name] = value
        return previous

    monkeypatch.setattr(
        _memory.mx,
        "set_memory_limit",
        lambda value: set_limit("memory", value),
    )
    monkeypatch.setattr(
        _memory.mx,
        "set_cache_limit",
        lambda value: set_limit("cache", value),
    )

    with (
        pytest.raises(RuntimeError, match="injected SCF failure"),
        _memory._bounded_dft_allocator(),
    ):
        raise RuntimeError("injected SCF failure")

    assert state == {"memory": 70_000_000_000, "cache": 12_000_000_000}
