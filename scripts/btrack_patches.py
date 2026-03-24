"""Monkey-patches for btrack's GLPK-based track optimiser.

btrack's ``TrackOptimiser.optimise`` rejects any GLPK status that is not
``"optimal"``.  When a ``mip_gap`` or ``tm_lim`` option causes the solver to
stop early, GLPK returns ``"feasible"`` — a valid solution that simply hasn't
been *proven* globally optimal.  For cell-tracking problems the feasible
solution is virtually always good enough (typical gaps are < 1%).

Call :func:`patch_optimiser` once before running ``tracker.optimise()`` to
accept ``"feasible"`` alongside ``"optimal"``.
"""

from __future__ import annotations
import logging

import btrack.optimise.optimiser as _optimiser_mod
from btrack.constants import Fates
from cvxopt import matrix, spmatrix
from cvxopt.glpk import ilp

logger = logging.getLogger(__name__)

_PATCHED = False

_ACCEPTABLE_STATUSES = frozenset({"optimal", "feasible"})


def _patched_optimise(self):  # noqa: PLR0912
    """Drop-in replacement for ``TrackOptimiser.optimise`` that accepts
    ``"feasible"`` solutions from GLPK in addition to ``"optimal"``."""

    logger.info("Setting up constraints matrix for global optimisation...")
    if self.options:
        logger.info(f"Using GLPK options: {self.options}...")

    def trk_idx(_h):
        return int(_h) - 1

    n_hypotheses = len(self.hypotheses)
    N = max(int(h.ID) for h in self.hypotheses)

    A = spmatrix([], [], [], (2 * N, n_hypotheses), "d")
    rho = matrix(0.0, (n_hypotheses, 1), "d")

    for counter, h in enumerate(self.hypotheses):
        rho[counter] = h.log_likelihood

        if h.hypothesis_type == Fates.FALSE_POSITIVE:
            trk = trk_idx(h.ID)
            A[trk, counter] = 1
            A[N + trk, counter] = 1
        elif h.hypothesis_type in _optimiser_mod.INIT_FATES:
            trk = trk_idx(h.ID)
            A[N + trk, counter] = 1
        elif h.hypothesis_type in _optimiser_mod.TERM_FATES:
            trk = trk_idx(h.ID)
            A[trk, counter] = 1
        elif h.hypothesis_type == Fates.APOPTOSIS:
            trk = trk_idx(h.ID)
            A[trk, counter] = 1
        elif h.hypothesis_type == Fates.LINK:
            trk_i = trk_idx(h.ID)
            trk_j = trk_idx(h.link_ID)
            A[trk_i, counter] = 1
            A[N + trk_j, counter] = 1
        elif h.hypothesis_type == Fates.DIVIDE:
            trk = trk_idx(h.ID)
            child_one = trk_idx(h.child_one_ID)
            child_two = trk_idx(h.child_two_ID)
            A[trk, counter] = 1
            A[N + child_one, counter] = 1
            A[N + child_two, counter] = 1
        elif h.hypothesis_type == Fates.MERGE:
            trk = trk_idx(h.ID)
            parent_one = trk_idx(h.parent_one_ID)
            parent_two = trk_idx(h.parent_two_ID)
            A[N + trk, counter] = 1
            A[parent_one, counter] = 1
            A[parent_two, counter] = 1
        else:
            raise ValueError(f"Unknown hypothesis: {h.hypothesis_type}")

    logger.info("Optimizing...")

    G = spmatrix([], [], [], (2 * N, n_hypotheses), "d")
    h_vec = matrix(0.0, (2 * N, 1), "d")
    Ix = set()
    B = set(range(n_hypotheses))
    b = matrix(1.0, (2 * N, 1), "d")

    status, x = ilp(-rho, -G, h_vec, A, b, Ix, B, options=self.options)

    if status not in _ACCEPTABLE_STATUSES:
        logger.warning(f"Optimizer returned status: {status}")
        return []

    if status == "feasible":
        logger.info("Optimizer returned 'feasible' (gap tolerance reached). Accepting solution.")

    results = [i for i, h in enumerate(self.hypotheses) if x[i] > 0]
    logger.info(f"Optimization complete. (Solution: {status})")
    return results


def patch_optimiser() -> None:
    """Replace ``TrackOptimiser.optimise`` with a version that accepts
    ``"feasible"`` GLPK solutions.  Safe to call multiple times."""
    global _PATCHED  # noqa: PLW0603
    if _PATCHED:
        return
    _optimiser_mod.TrackOptimiser.optimise = _patched_optimise
    _PATCHED = True
    logger.info("btrack optimiser patched to accept 'feasible' GLPK solutions")
