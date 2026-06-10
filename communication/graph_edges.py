"""Array-native graph edge builders for communication GNNs."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


def unique_edge_index(edge_index: np.ndarray) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int32)
    if edge_index.size == 0:
        return np.zeros((2, 0), dtype=np.int32)
    edge_index = edge_index.reshape(2, -1)
    edges = np.ascontiguousarray(edge_index.T)
    uniq = np.unique(edges, axis=0)
    return uniq.astype(np.int32, copy=False).T


def add_self_loops(edge_index: np.ndarray, num_nodes: int) -> np.ndarray:
    loops = np.vstack((
        np.arange(int(num_nodes), dtype=np.int32),
        np.arange(int(num_nodes), dtype=np.int32),
    ))
    if edge_index.size == 0:
        return loops
    return unique_edge_index(np.concatenate((edge_index, loops), axis=1))


def build_link_edge_index(
    num_v2v_list: np.ndarray,
    *,
    node_positions: np.ndarray | None,
    use_proximity_edges: bool,
    proximity_radius: float,
    max_proximity_neighbors: int,
    add_self_loop: bool,
) -> Tuple[np.ndarray, List[str], np.ndarray]:
    """Build directed edge_index for link nodes without NetworkX."""

    num_v2v_list = np.asarray(num_v2v_list)
    n_veh = int(num_v2v_list.shape[0])
    n_nodes = 3 * n_veh
    if n_veh <= 0:
        return np.zeros((2, 0), dtype=np.int32), [], np.zeros((0, 2), dtype=np.int32)

    veh = np.repeat(np.arange(n_veh, dtype=np.int32), 3)
    link = np.tile(np.arange(3, dtype=np.int32), n_veh)
    link_array = np.stack((veh, link), axis=1)
    order_nodes = [f"{int(v)}_{int(l)}" for v, l in link_array]

    src_chunks = []
    dst_chunks = []

    # Intra-vehicle link clique, directed.
    base = (3 * np.arange(n_veh, dtype=np.int32))[:, None]
    local_src = np.repeat(np.arange(3, dtype=np.int32), 2)
    local_dst = np.asarray([1, 2, 0, 2, 0, 1], dtype=np.int32)
    src_chunks.append((base + local_src[None, :]).reshape(-1))
    dst_chunks.append((base + local_dst[None, :]).reshape(-1))

    # Communication topology: first destination per vehicle, same link index.
    dest_first = np.full(n_veh, -1, dtype=np.int32)
    has_dest = num_v2v_list > 0
    rows = np.flatnonzero(np.any(has_dest, axis=1))
    if rows.size:
        dest_first[rows] = np.argmax(has_dest[rows], axis=1).astype(np.int32)
        valid = (dest_first >= 0) & (dest_first < n_veh) & (dest_first != np.arange(n_veh, dtype=np.int32))
        if np.any(valid):
            src_veh = np.flatnonzero(valid).astype(np.int32)
            dst_veh = dest_first[valid].astype(np.int32)
            src = (3 * src_veh[:, None] + np.arange(3, dtype=np.int32)[None, :]).reshape(-1)
            dst = (3 * dst_veh[:, None] + np.arange(3, dtype=np.int32)[None, :]).reshape(-1)
            src_chunks.extend((src, dst))
            dst_chunks.extend((dst, src))

    # Spatial proximity edges, vehicle-level top-k expanded to same-link edges.
    if (
        use_proximity_edges
        and node_positions is not None
        and np.asarray(node_positions).shape[0] >= n_nodes
    ):
        veh_pos = np.asarray(node_positions, dtype=np.float32)[0:n_nodes:3, :]
        diff = veh_pos[:, None, :] - veh_pos[None, :, :]
        dist = np.sqrt(np.sum(diff * diff, axis=-1))
        radius = float(proximity_radius)
        max_neigh = int(max_proximity_neighbors)
        order = np.argsort(dist, axis=1)
        order_dist = np.take_along_axis(dist, order, axis=1)
        valid_order = (order != np.arange(n_veh, dtype=np.int32)[:, None]) & (order_dist <= radius)
        if max_neigh > 0:
            rank = np.cumsum(valid_order, axis=1)
            valid_order &= rank <= max_neigh
        src_veh = np.repeat(np.arange(n_veh, dtype=np.int32), n_veh)[valid_order.reshape(-1)]
        dst_veh = order.reshape(-1)[valid_order.reshape(-1)].astype(np.int32, copy=False)
        if src_veh.size:
            same_links = np.arange(3, dtype=np.int32)
            src = (3 * src_veh[:, None] + same_links[None, :]).reshape(-1)
            dst = (3 * dst_veh[:, None] + same_links[None, :]).reshape(-1)
            src_chunks.extend((src, dst))
            dst_chunks.extend((dst, src))

    edge_index = unique_edge_index(np.vstack((np.concatenate(src_chunks), np.concatenate(dst_chunks))))
    if add_self_loop:
        edge_index = add_self_loops(edge_index, n_nodes)
    return edge_index, order_nodes, link_array.astype(np.int32, copy=False)
