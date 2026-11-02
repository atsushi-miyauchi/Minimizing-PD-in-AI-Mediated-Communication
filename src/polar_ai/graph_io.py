"""Load edge-list graphs and build sparse matrices."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Iterable

import networkx as nx
import numpy as np
from scipy import sparse


Edge = tuple[Hashable, Hashable, float]


@dataclass(frozen=True)
class GraphMatrices:
    """Sparse graph matrices plus node labels."""

    W: sparse.csr_matrix
    P: sparse.csr_matrix
    Wbar: sparse.csr_matrix
    nodes: tuple[Hashable, ...]
    metadata: dict[str, Any]


def load_edge_list(
    path: str | Path,
    *,
    directed: bool = True,
    weighted: bool | None = None,
    delimiter: str | None = None,
    comments: tuple[str, ...] = ("#", "%"),
    has_header: bool = False,
) -> GraphMatrices:
    """Load an edge list and return W, row-stochastic P, and Wbar.

    Lines must contain at least source and target. A third column is interpreted
    as weight when ``weighted`` is True or None. Missing weights default to 1.
    Undirected inputs are expanded into two directed edges.
    """

    path = Path(path)
    edges: list[Edge] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith(comments):
                continue
            if has_header:
                has_header = False
                continue

            parts = _split_edge_line(stripped, delimiter)
            if len(parts) < 2:
                raise ValueError(f"Line {line_number} has fewer than two columns.")

            weight = 1.0
            if len(parts) >= 3 and weighted is not False:
                try:
                    weight = float(parts[2])
                except ValueError as exc:
                    raise ValueError(
                        f"Line {line_number} has a nonnumeric weight: {parts[2]!r}."
                    ) from exc
            edges.append((parts[0], parts[1], weight))

    matrices = build_graph_matrices(edges, directed=directed)
    metadata = dict(matrices.metadata)
    metadata.update(
        {
            "path": str(path),
            "directed_input": directed,
            "weighted_input": weighted,
        }
    )
    return GraphMatrices(
        W=matrices.W,
        P=matrices.P,
        Wbar=matrices.Wbar,
        nodes=matrices.nodes,
        metadata=metadata,
    )


def build_graph_matrices(
    edges: Iterable[Edge],
    *,
    directed: bool = True,
    keep_largest_wcc: bool = True,
) -> GraphMatrices:
    """Build sparse matrices from weighted edges.

    Directed edge weights are stored as ``W[u, v]``. For dangling rows in W, P
    uses a self-loop so every row of P sums to 1.
    """

    directed_edges: list[Edge] = []
    node_order: list[Hashable] = []
    seen: set[Hashable] = set()
    graph = nx.DiGraph()

    for source, target, weight in edges:
        weight = float(weight)
        if not np.isfinite(weight) or weight < 0:
            raise ValueError("Edge weights must be finite and nonnegative.")

        for node in (source, target):
            if node not in seen:
                seen.add(node)
                node_order.append(node)
                graph.add_node(node)

        edge_items = [(source, target, weight)]
        if not directed and source != target:
            edge_items.append((target, source, weight))

        for u, v, w in edge_items:
            directed_edges.append((u, v, w))
            graph.add_edge(u, v)

    if not node_order:
        raise ValueError("Cannot build matrices for an empty graph.")

    original_n_nodes = len(node_order)
    original_n_edges = len(directed_edges)

    if keep_largest_wcc:
        components = list(nx.weakly_connected_components(graph))
        if not components:
            raise ValueError("Graph has no weakly connected components.")
        position = {node: idx for idx, node in enumerate(node_order)}
        component = max(
            components,
            key=lambda comp: (len(comp), -min(position[node] for node in comp)),
        )
        nodes = tuple(node for node in node_order if node in component)
    else:
        nodes = tuple(node_order)
        component = set(nodes)

    index = {node: i for i, node in enumerate(nodes)}
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []

    for source, target, weight in directed_edges:
        if source in component and target in component and weight != 0.0:
            rows.append(index[source])
            cols.append(index[target])
            data.append(weight)

    n = len(nodes)
    W = sparse.coo_matrix((data, (rows, cols)), shape=(n, n), dtype=float).tocsr()
    W.sum_duplicates()
    P = build_row_stochastic_matrix(W)
    Wbar = build_symmetrized_weights(W)

    return GraphMatrices(
        W=W,
        P=P,
        Wbar=Wbar,
        nodes=nodes,
        metadata={
            "directed_input": directed,
            "n_nodes": n,
            "n_edges": int(W.nnz),
            "original_n_nodes": original_n_nodes,
            "original_n_edges": original_n_edges,
            "kept_largest_wcc": keep_largest_wcc,
        },
    )


def build_row_stochastic_matrix(W: sparse.spmatrix) -> sparse.csr_matrix:
    """Return a row-stochastic matrix from a nonnegative weighted adjacency."""

    W = W.tocsr().astype(float)
    row_sums = np.asarray(W.sum(axis=1)).ravel()
    positive = row_sums > 0

    inv = np.zeros_like(row_sums, dtype=float)
    inv[positive] = 1.0 / row_sums[positive]
    P = sparse.diags(inv, format="csr") @ W

    dangling = np.flatnonzero(~positive)
    if dangling.size:
        self_loops = sparse.csr_matrix(
            (np.ones(dangling.size), (dangling, dangling)), shape=W.shape
        )
        P = P + self_loops

    return P.tocsr()


def build_symmetrized_weights(W: sparse.spmatrix) -> sparse.csr_matrix:
    """Return symmetric unordered disagreement weights.

    For a pair with both directed edges, Wbar uses the average directed weight.
    For a pair with only one directed edge, Wbar uses the existing weight.
    """

    W = W.tocoo().astype(float)
    pair_weights: dict[tuple[int, int], list[float]] = {}
    for row, col, value in zip(W.row, W.col, W.data):
        if row == col:
            continue
        u, v = sorted((int(row), int(col)))
        pair_weights.setdefault((u, v), []).append(float(value))

    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    for (u, v), values in pair_weights.items():
        weight = float(sum(values) / len(values))
        rows.extend([u, v])
        cols.extend([v, u])
        data.extend([weight, weight])

    Wbar = sparse.coo_matrix((data, (rows, cols)), shape=W.shape, dtype=float).tocsr()
    Wbar.sum_duplicates()
    Wbar.eliminate_zeros()
    return Wbar


def _split_edge_line(line: str, delimiter: str | None) -> list[str]:
    if delimiter is not None:
        return [part.strip() for part in line.split(delimiter) if part.strip()]
    if "," in line:
        return [part.strip() for part in line.split(",") if part.strip()]
    return line.split()
