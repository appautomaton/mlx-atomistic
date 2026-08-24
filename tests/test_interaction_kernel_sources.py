"""Static contracts for production Interaction32 Metal sources."""

from __future__ import annotations

import mlx_atomistic.metal_kernels as kernels


def test_fused_half_kernels_enable_inline_active_right_compaction(monkeypatch):
    """Ordinary and NBFIX production variants share the compaction layout."""

    calls = []

    def fake_metal_kernel(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(kernels.mx.fast, "metal_kernel", fake_metal_kernel)
    monkeypatch.setattr(
        kernels,
        "_interaction32_fused_half_canonical_force_kernel_singleton",
        None,
    )
    monkeypatch.setattr(
        kernels,
        "_interaction32_fused_half_nbfix_canonical_force_kernel_singleton",
        None,
    )

    kernels._interaction32_fused_half_canonical_force_kernel()
    kernels._interaction32_fused_half_nbfix_canonical_force_kernel()

    assert len(calls) == 2
    compaction_define = "#define MLX_ATOMISTIC_INTERACTION32_ACTIVE_COMPACTION 1\n"
    assert all(compaction_define in call["source"] for call in calls)
    assert all("simd_prefix_exclusive_sum(right_active)" in call["source"] for call in calls)
    assert "#define MLX_ATOMISTIC_NBFIX 1\n" not in calls[0]["source"]
    assert "#define MLX_ATOMISTIC_NBFIX 1\n" in calls[1]["source"]


def test_ordinary_builder_kernels_use_constant_time_special_membership(monkeypatch):
    """Count and fallback scatter share the generation-owned special bitset."""

    calls = []

    def fake_metal_kernel(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(kernels.mx.fast, "metal_kernel", fake_metal_kernel)
    for name in (
        "_interaction32_ordinary_count_kernel_singleton",
        "_interaction32_ordinary_cached_count_kernel_singleton",
        "_interaction32_ordinary_scatter_kernel_singleton",
    ):
        monkeypatch.setattr(kernels, name, None)

    kernels._interaction32_ordinary_count_kernel()
    kernels._interaction32_ordinary_cached_count_kernel()
    kernels._interaction32_ordinary_scatter_kernel()

    assert len(calls) == 3
    assert all("special_pair_words" in call["input_names"] for call in calls)
    assert all("special_word >>" in call["source"] for call in calls)
    assert all("while (low < high)" not in call["source"] for call in calls)
    assert "lane == 0u || lane == 16u" not in calls[1]["source"]
