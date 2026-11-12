"""One fixed view of a task: a dihedral transform and a color permutation, and its inverse.

Why: run ``P6.3-04``'s direct-answer arm asks the model twice per task, once on the task and
once on one fixed augmentation of it, and maps the second answer back. DEV40 is public, so
Qwen3.8 may have seen it; a model that recalls outputs scores far better on the original than on
the view, which no published copy shows. The operations are those of NVARC's
``ArcDataset.forward_mod`` and ``invert_mod`` (``arc26/nvarc/arc_loader.py``, which imports
transformers and so is not imported here): ``rot90`` (counterclockwise, numpy's), ``transpose``,
and a color permutation, applied in order and undone in reverse order. Color 0 stays 0: the
background carries meaning in many ARC tasks, and a view that recolors it is a different puzzle.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from arc26.agent.types import AgentTask, Pair
from arc26.data import Grid

OPS = ("rot90", "transpose")


@dataclass(frozen=True)
class View:
    """A dihedral transform and a color permutation.

    Attributes:
        ops: Spatial operations in order, each ``"rot90"`` or ``"transpose"``.
        colors: ``colors[c]`` is the new color of color ``c``; a permutation of 0..9 with 0 fixed.
    """

    ops: tuple[str, ...]
    colors: tuple[int, ...]

    def __post_init__(self) -> None:
        """Reject an unknown operation or a color map that is not a permutation fixing 0.

        Raises:
            ValueError: If an operation is unknown or ``colors`` is not such a permutation.
        """
        if any(op not in OPS for op in self.ops):
            raise ValueError(f"ops must be among {OPS}; got {self.ops}")
        if sorted(self.colors) != list(range(10)) or self.colors[0] != 0:
            raise ValueError(f"colors must permute 0..9 and keep 0; got {self.colors}")

    @property
    def key(self) -> str:
        """Return the view in NVARC's key notation, such as ``x.transpose.rot90.permute0...``."""
        return ".".join(["x", *self.ops, "permute" + "".join(map(str, self.colors))])

    def forward(self, grid: Grid) -> Grid:
        """Apply the view to a grid.

        Args:
            grid: The grid.

        Returns:
            The transformed grid as lists of ints.
        """
        a = np.asarray(grid, dtype=int)
        for op in self.ops:
            a = np.rot90(a) if op == "rot90" else np.swapaxes(a, 0, 1)
        return np.asarray(self.colors)[a].tolist()

    def inverse(self, grid: Grid) -> Grid:
        """Undo the view on a grid (an answer given on the view).

        Args:
            grid: A grid in the view's frame.

        Returns:
            The grid in the task's own frame, as lists of ints.
        """
        a = np.argsort(self.colors)[np.asarray(grid, dtype=int)]
        for op in reversed(self.ops):
            a = np.rot90(a, k=3) if op == "rot90" else np.swapaxes(a, 0, 1)
        return a.tolist()

    def task(self, task: AgentTask, suffix: str = "~v1") -> AgentTask:
        """Return the task seen through the view, under a new id.

        Args:
            task: The task.
            suffix: Appended to the task id, so the view's attempts are told apart.

        Returns:
            Every demonstration grid and test input transformed.
        """
        return AgentTask(
            task.task_id + suffix,
            tuple(Pair(self.forward(p.input), self.forward(p.output)) for p in task.train),
            tuple(self.forward(g) for g in task.test_inputs),
        )


def fixed_view(ops: Sequence[str], seed: int) -> View:
    """Return the view with the given operations and a color permutation drawn from a seed.

    Args:
        ops: The spatial operations.
        seed: The permutation's seed (the run's fixed seed).

    Returns:
        The view; colors 1..9 shuffled by ``random.Random(seed)``, 0 kept.
    """
    rest = list(range(1, 10))
    random.Random(seed).shuffle(rest)
    return View(tuple(ops), (0, *rest))
