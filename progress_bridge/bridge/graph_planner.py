"""Baseline with no bridge network: subgoals come straight from the graph.

Start and goal snap to their nearest nodes and Dijkstra finds the path between
them. The subgoal is the first node at least `ahead` cost along it, or the true
goal if the whole path is shorter. If no path exists, CEM plans straight to the
goal. Cross edges that teacher.py never checked are checked here on first use.
"""

import numpy as np
import torch
from scipy.sparse.csgraph import dijkstra

from bridge.planner import BridgeSolver
from bridge.teacher import EdgeTable, reachability_residual


def load_edges(graph, paths):
    """The teacher's final edge state: removed edges stay removed, checks are kept."""
    table = EdgeTable(graph["src"], graph["dst"], graph["cost"], graph["is_cross"], len(graph["z"]))
    assert (table.keys == paths["edge_keys"]).all()
    table.alive = paths["edge_alive"].copy()
    table.residual = paths["edge_residual"].copy()
    table.checked = ~np.isnan(table.residual)
    return table


def shortest_path(table, source, target, residual_fn, threshold):
    """(nodes, cumulative cost) from source to target over passing edges, or None."""
    while True:
        dist, pred = dijkstra(table.csr(), directed=True, indices=source, return_predecessors=True)
        if not np.isfinite(dist[target]):
            return None
        path = [target]
        while path[-1] != source:
            path.append(pred[path[-1]])
        path = np.array(path[::-1])
        if len(path) == 1:
            return path, np.zeros(1)
        edges = table.lookup(path[:-1], path[1:])
        new = edges[table.is_cross[edges] & ~table.checked[edges]]
        if len(new) == 0:
            return path, np.concatenate([[0.0], np.cumsum(table.cost[edges])])
        res = residual_fn(table.keys[new] // table.n, table.keys[new] % table.n)
        table.checked[new], table.residual[new] = True, res
        table.alive[new[res > threshold]] = False


class GraphSolver(BridgeSolver):
    def __init__(self, model, cem, nodes, table, threshold, close_dist, ahead=25.0):
        super().__init__(model, None, cem, sigma=None, r=None, close_dist=close_dist, n_paths=1)
        self.nodes, self.table, self.threshold, self.ahead = nodes, table, threshold, ahead
        self.stats.update(no_path=0, edges_checked=0)

    def residual(self, u, v):
        self.stats["edges_checked"] += len(u)
        return reachability_residual(self.model, self.nodes[u].cpu(), self.nodes[v].cpu()).numpy()

    def propose(self, z, z_goal):
        sources = torch.cdist(z, self.nodes).argmin(1).tolist()
        targets = torch.cdist(z_goal, self.nodes).argmin(1).tolist()
        out = []
        for i, (s, t) in enumerate(zip(sources, targets)):
            found = shortest_path(self.table, s, t, self.residual, self.threshold)
            if found is None:
                self.stats["no_path"] += 1
                out.append(z_goal[i])
                continue
            path, cum = found
            far = np.flatnonzero(cum >= self.ahead)
            out.append(self.nodes[path[far[0]]] if len(far) else z_goal[i])
        return torch.stack(out)
