"""
Ray-tracing wrapper.

Generates ground-truth RSSI (in dBm) using a Sionna `PathSolver`.

Two public entry points are exposed:

- :meth:`RayTracer.step_measurements` — ray-traces all intra-zone TX-RX
  pairs of nodes inside each zone using bounded TX chunks (all N nodes in
  the zone still act as both TX and RX; only the solver call is split).
- :meth:`RayTracer.measure_pairs` — ray-traces an arbitrary
  ``[(tx_xy, [rx_xy, ...]), ...]`` grouping using bounded TX chunks.

Both methods are built on top of the private batched primitive
:meth:`RayTracer._solve_pairs_batched`, which performs a single
``solver(scene, ...)`` call with multiple TX and multiple RX in the scene
and returns the dense :math:`[num\\_rx, num\\_tx]` path-gain matrix.

Sionna 1.x's ``paths.a`` tensor has shape
``[num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]`` (verified
empirically on the GPU compute node, see the ``probe-sionna-shape*``
logs). To get the gain matrix we sum over the path axis and then over
the two antenna axes:

.. code-block:: python

    a = solver(scene, ...).a
    a_c = a[0].numpy() + 1j * a[1].numpy()                   # if tuple
    g = np.sum(np.abs(a_c) ** 2, axis=-1).sum(axis=(1, 3))   # [num_rx, num_tx]

Chunking the dense step-measurement solver calls keeps the memory footprint
bounded while preserving the same output matrix semantics. The numerical
result is identical up to Monte-Carlo ray-sampling noise.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

try:
    import sionna.rt as rt
except (ModuleNotFoundError, ImportError):  # Cached replay needs no RT backend.
    rt = None  # type: ignore[assignment]

try:
    import drjit as dr
except ImportError:  # pragma: no cover - depends on the Sionna backend install
    dr = None


_Pos2 = tuple[float, float]
_Pos3 = tuple[float, float, float]
_Position = _Pos2 | _Pos3


def _position_xyz(position: Sequence[float], z_plane: float) -> tuple[float, float, float]:
    if len(position) >= 3:
        return float(position[0]), float(position[1]), float(position[2])
    return float(position[0]), float(position[1]), float(z_plane)


def _position_key(position: Sequence[float]) -> _Position:
    """Return a stable 2D/3D key without discarding supplied elevation."""
    if len(position) >= 3:
        return float(position[0]), float(position[1]), float(position[2])
    return float(position[0]), float(position[1])


def _clear_tx_rx(scene) -> None:
    for name in list(scene.transmitters.keys()):
        scene.remove(name)
    for name in list(scene.receivers.keys()):
        scene.remove(name)


def _flush_drjit_cache() -> None:
    """Return cached Dr.Jit allocations to the backend when supported."""
    if dr is not None and hasattr(dr, "flush_malloc_cache"):
        dr.flush_malloc_cache()


def _gains_to_rssi(
    gains: np.ndarray,
    *,
    tx_power_dbm: float,
    rssi_min: float,
    rssi_max: float,
) -> np.ndarray:
    """Convert path-gain magnitudes to clipped RSSI in dBm.

    Works for any array shape (e.g. 1-D vectors and 2-D ``[rx, tx]``
    matrices). Entries with effectively-zero gain are floored to
    ``rssi_min`` to avoid ``-inf`` from ``log10(0)``.
    """
    g = np.asarray(gains, dtype=np.float64)
    safe = np.maximum(g, 1e-15)
    rssi = tx_power_dbm + 10.0 * np.log10(safe)
    rssi = np.where(g > 1e-15, rssi, rssi_min)
    return np.clip(rssi, rssi_min, rssi_max)


class RayTracer:
    """Thin stateful facade around a Sionna scene + ``PathSolver``."""

    def __init__(
        self,
        scene,
        *,
        num_rays: int,
        max_depth: int,
        tx_power_dbm: float,
        rssi_min: float,
        rssi_max: float,
        z_plane: float = 1.5,
        tx_batch_size: int = 32,
        refraction: bool = True,
    ) -> None:
        if rt is None:
            raise RuntimeError("sionna.rt is required for live ray tracing; cached replay must replace RayTracer")
        self.scene = scene
        self.solver = rt.PathSolver()
        self.num_rays = int(num_rays)
        self.max_depth = int(max_depth)
        self.tx_power_dbm = float(tx_power_dbm)
        self.rssi_min = float(rssi_min)
        self.rssi_max = float(rssi_max)
        self.z_plane = float(z_plane)
        self.tx_batch_size = max(1, int(tx_batch_size))
        self.refraction = bool(refraction)

    # ------------------------------------------------------- batched core --

    def _solve_pairs_batched(
        self,
        tx_positions: Sequence[_Pos2],
        rx_positions: Sequence[_Pos2],
    ) -> np.ndarray:
        """Run a single solver call with multiple TX *and* multiple RX.

        Returns
        -------
        ndarray, shape ``(len(rx_positions), len(tx_positions))``
            ``G[j, i]`` is the path-gain magnitude received at
            ``rx_positions[j]`` when transmitting from
            ``tx_positions[i]``. The caller is responsible for masking
            out self-pairs (i.e. when a (tx, rx) refers to the same
            physical node).

        Notes
        -----
        Both lists may be empty; in that case an array with the
        corresponding zero-length axis is returned without invoking the
        solver.
        """
        n_tx = len(tx_positions)
        n_rx = len(rx_positions)
        if n_tx == 0 or n_rx == 0:
            return np.zeros((n_rx, n_tx), dtype=np.float64)

        _clear_tx_rx(self.scene)
        for i, position in enumerate(tx_positions):
            x, y, z = _position_xyz(position, self.z_plane)
            self.scene.add(
                rt.Transmitter(
                    name=f"TX_{i}",
                    position=[x, y, z],
                )
            )
        for j, position in enumerate(rx_positions):
            x, y, z = _position_xyz(position, self.z_plane)
            self.scene.add(
                rt.Receiver(
                    name=f"RX_{j}",
                    position=[x, y, z],
                )
            )

        try:
            paths = self.solver(
                self.scene,
                max_depth=self.max_depth,
                samples_per_src=self.num_rays,
                los=True,
                specular_reflection=True,
                diffuse_reflection=False,
                refraction=self.refraction,
                diffraction=False,
                edge_diffraction=False,
            )

            a = paths.a
            if isinstance(a, tuple):
                a_c = a[0].numpy() + 1j * a[1].numpy()
            else:
                a_c = a.numpy()

            # a_c: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths]
            g = np.sum(np.abs(a_c) ** 2, axis=-1)        # [rx, rx_ant, tx, tx_ant]
            g = g.sum(axis=(1, 3))                        # [rx, tx]

            # Sionna may return fewer rows/cols than expected when no paths
            # were found for the entire scene (rare, but defensive).
            out = np.zeros((n_rx, n_tx), dtype=np.float64)
            rr = min(n_rx, g.shape[0])
            tt = min(n_tx, g.shape[1])
            out[:rr, :tt] = g[:rr, :tt]
            return out
        except RuntimeError:
            _flush_drjit_cache()
            raise
        finally:
            _clear_tx_rx(self.scene)

    def _solve_pairs_tx_chunked(
        self,
        tx_positions: Sequence[_Pos2],
        rx_positions: Sequence[_Pos2],
    ) -> np.ndarray:
        """Solve all RXs against TX chunks to bound Sionna path-buffer memory."""
        n_tx = len(tx_positions)
        n_rx = len(rx_positions)
        if n_tx == 0 or n_rx == 0:
            return np.zeros((n_rx, n_tx), dtype=np.float64)
        if n_tx <= self.tx_batch_size:
            return self._solve_pairs_batched(tx_positions, rx_positions)

        out = np.zeros((n_rx, n_tx), dtype=np.float64)
        for start in range(0, n_tx, self.tx_batch_size):
            stop = min(n_tx, start + self.tx_batch_size)
            out[:, start:stop] = self._solve_pairs_batched(
                tx_positions[start:stop], rx_positions
            )
        return out

    # ------------------------------------------------------------------ API --

    def measure_pairs(
        self,
        tx_groups: Sequence[tuple[_Position, Sequence[_Position]]],
    ) -> list[list[float]]:
        """Return RSSIs for each group: ``[[rssi_for_rx_1, ...], ...]``.

        Groups are evaluated in bounded TX chunks:

        1. Build the union of unique TX positions (preserving group
           order).
        2. For each TX chunk, build the union of RX positions requested by
           TXs in that chunk.
        3. Trace that chunk and keep only the requested entries.
        4. For each input group, look up the per-(tx, rx) entry and emit
           it in the order requested.

        Empty input groups (``rxs == []``) are preserved as empty output
        lists. If every group is empty, no solver call is made.
        """
        # 1. Build canonical TX/RX tables. `dict` preserves
        #    insertion order on Python 3.7+ which is what we want.
        tx_to_rx: dict[_Position, dict[_Position, None]] = {}
        for tx, rxs in tx_groups:
            if not rxs:
                continue
            tx_t = _position_key(tx)
            rx_map = tx_to_rx.setdefault(tx_t, {})
            for rx in rxs:
                rx_map[_position_key(rx)] = None

        if not tx_to_rx:
            return [[] for _ in tx_groups]

        tx_pos = list(tx_to_rx.keys())

        # 2. Trace only requested receiver sets per TX chunk. This avoids
        #    the full TX-union x RX-union cross-product while keeping the
        #    normal 10-TX fidelity/oracle probes compact and fast.
        rssi_by_tx: dict[_Position, dict[_Position, float]] = {}
        for start in range(0, len(tx_pos), self.tx_batch_size):
            tx_chunk = tx_pos[start : start + self.tx_batch_size]
            rx_index: dict[_Position, int] = {}
            for tx_t in tx_chunk:
                for rx_t in tx_to_rx[tx_t]:
                    if rx_t not in rx_index:
                        rx_index[rx_t] = len(rx_index)
            rx_pos = list(rx_index.keys())
            G = self._solve_pairs_batched(tx_chunk, rx_pos)
            rssi = _gains_to_rssi(
                G,
                tx_power_dbm=self.tx_power_dbm,
                rssi_min=self.rssi_min,
                rssi_max=self.rssi_max,
            )
            for ti, tx_t in enumerate(tx_chunk):
                tx_vals = rssi_by_tx.setdefault(tx_t, {})
                for rx_t in tx_to_rx[tx_t]:
                    tx_vals[rx_t] = float(rssi[rx_index[rx_t], ti])

        # 3. Pull out the requested per-group entries.
        out: list[list[float]] = []
        for tx, rxs in tx_groups:
            if not rxs:
                out.append([])
                continue
            tx_t = _position_key(tx)
            row = []
            for rx in rxs:
                rx_t = _position_key(rx)
                row.append(rssi_by_tx[tx_t][rx_t])
            out.append(row)
        return out

    def step_measurements(
        self,
        nodes,
        zone_node_indices: dict[int, list[int]],
    ) -> list[tuple[int, int, int, float]]:
        """Ray-trace every intra-zone TX-RX pair.

        For each zone, all nodes in the zone act as both TX *and* RX; the
        dense matrix is solved in TX chunks. The returned tuples skip
        self-pairs.

        Returns
        -------
        list[tuple[int, int, int, float]]
            ``[(az, tx_idx, rx_idx, rssi), ...]`` covering every ordered
            pair ``(tx, rx)`` of distinct same-zone nodes.
        """
        out: list[tuple[int, int, int, float]] = []
        for az, indices in zone_node_indices.items():
            n = len(indices)
            if n < 2:
                continue
            positions = [(nodes[i].x, nodes[i].y) for i in indices]

            G = self._solve_pairs_tx_chunked(positions, positions)
            rssi_mat = _gains_to_rssi(
                G,
                tx_power_dbm=self.tx_power_dbm,
                rssi_min=self.rssi_min,
                rssi_max=self.rssi_max,
            )

            for j, rx_idx in enumerate(indices):
                for i, tx_idx in enumerate(indices):
                    if i == j:
                        continue
                    out.append((az, int(tx_idx), int(rx_idx), float(rssi_mat[j, i])))
        return out
