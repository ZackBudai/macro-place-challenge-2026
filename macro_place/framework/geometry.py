"""Geometry and placement helpers shared by competition placers."""

import random
from typing import Iterable, Optional, Sequence

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def macro_bbox(benchmark: Benchmark, index: int, placement: Optional[torch.Tensor] = None):
    """Return (x_min, y_min, x_max, y_max) for a macro."""
    tensor = placement if placement is not None else benchmark.macro_positions
    x, y = tensor[index].tolist()
    w, h = benchmark.macro_sizes[index].tolist()
    half_w = w / 2.0
    half_h = h / 2.0
    return x - half_w, y - half_h, x + half_w, y + half_h


def clamp_placement_to_canvas(
    placement: torch.Tensor,
    benchmark: Benchmark,
    indices: Optional[Sequence[int]] = None,
    gap: float = 0.0,
    preserve_fixed: bool = True,
) -> torch.Tensor:
    """Clamp macro centers so the full macro rectangles stay inside the canvas."""
    clamped = placement.clone()
    if indices is None:
        indices = range(benchmark.num_macros)

    for index in indices:
        width = float(benchmark.macro_sizes[index, 0].item())
        height = float(benchmark.macro_sizes[index, 1].item())
        x_min = width / 2.0 + gap
        y_min = height / 2.0 + gap
        x_max = float(benchmark.canvas_width) - width / 2.0 - gap
        y_max = float(benchmark.canvas_height) - height / 2.0 - gap

        if x_max < x_min:
            x_min = x_max = float(benchmark.canvas_width) / 2.0
        if y_max < y_min:
            y_min = y_max = float(benchmark.canvas_height) / 2.0

        clamped[index, 0] = torch.clamp(clamped[index, 0], x_min, x_max)
        clamped[index, 1] = torch.clamp(clamped[index, 1], y_min, y_max)

    if preserve_fixed and benchmark.macro_fixed.any():
        clamped[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

    return clamped


def hard_movable_indices(benchmark: Benchmark) -> torch.Tensor:
    """Return indices of movable hard macros."""
    mask = benchmark.get_movable_mask() & benchmark.get_hard_macro_mask()
    return torch.where(mask)[0]


def soft_movable_indices(benchmark: Benchmark) -> torch.Tensor:
    """Return indices of movable soft macros."""
    mask = benchmark.get_movable_mask() & benchmark.get_soft_macro_mask()
    return torch.where(mask)[0]


def pack_macros_in_rows(
    benchmark: Benchmark,
    placement: Optional[torch.Tensor] = None,
    indices: Optional[Iterable[int]] = None,
    gap: float = 1.0e-3,
) -> torch.Tensor:
    """Shelf-pack selected macros left-to-right in rows.

    This is a simple legal initializer that works well as a starting point
    for competition placers. It preserves fixed macros and leaves unselected
    macros unchanged.
    """
    packed = benchmark.macro_positions.clone() if placement is None else placement.clone()
    if indices is None:
        indices = hard_movable_indices(benchmark).tolist()
    else:
        indices = list(indices)

    if not indices:
        return packed

    sizes = benchmark.macro_sizes
    canvas_w = float(benchmark.canvas_width)
    canvas_h = float(benchmark.canvas_height)

    indices = sorted(indices, key=lambda index: (-float(sizes[index, 1]), -float(sizes[index, 0])))

    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0

    for index in indices:
        width = float(sizes[index, 0].item())
        height = float(sizes[index, 1].item())

        if cursor_x > 0.0 and cursor_x + width > canvas_w:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0

        x_center = cursor_x + width / 2.0
        y_center = cursor_y + height / 2.0

        if cursor_y + height > canvas_h:
            x_center = min(max(width / 2.0, x_center), canvas_w - width / 2.0)
            y_center = min(max(height / 2.0, y_center), canvas_h - height / 2.0)

        packed[index, 0] = x_center
        packed[index, 1] = y_center

        cursor_x += width + gap
        row_height = max(row_height, height)

    if benchmark.macro_fixed.any():
        packed[benchmark.macro_fixed] = benchmark.macro_positions[benchmark.macro_fixed]

    return packed
