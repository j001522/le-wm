from unittest import mock

import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra

from bridge.graph import build_edges, select_nodes
from bridge.graph_planner import shortest_path
from bridge.model import BridgeDiT
from bridge.planner import BridgeSolver, first_subgoal, sample_paths
from bridge.teacher import EdgeTable, uses_edge, walk_back
from bridge.train import TeacherPaths, targets
from bridge.visualize import panel_x, subgoal_frames

D = "cuda" if torch.cuda.is_available() else "cpu"


def test_select_nodes_keeps_stride_within_episodes():
    frames, eps = select_nodes(np.array([12, 7, 20]), np.array([0, 12, 19]), 5, 100, np.random.default_rng(0))
    assert sorted(frames.tolist()) == [0, 5, 10, 12, 17, 19, 24, 29, 34]
    assert eps.tolist() == [0, 0, 0, 1, 1, 2, 2, 2, 2]


def test_within_edges_are_forward_only_and_links_cross_episodes():
    z = np.array([[0.0], [10.0], [0.1], [10.1]], dtype=np.float32)
    eps = np.array([0, 0, 1, 1])
    link = (np.array([1]), np.array([2]), np.array([9.9], dtype=np.float32))
    with mock.patch("bridge.graph.cross_episode_neighbours", return_value=link):
        e = build_edges(z, eps, 5, 1, 25)
    pairs = set(zip(e["src"].tolist(), e["dst"].tolist()))
    assert (0, 1) in pairs and (1, 0) not in pairs  # time runs forward within an episode
    assert (1, 2) in pairs and (2, 1) in pairs  # candidate links go both ways
    assert e["is_cross"].sum() == 2


def test_walk_back_matches_graph():
    # 0->1->2->3, plus a shortcut 0->3 that costs more
    A = sp.csr_matrix(([1.0, 1.0, 1.0, 9.0], ([0, 1, 2, 0], [1, 2, 3, 3])), shape=(5, 5))
    _, pred = dijkstra(A, indices=0, return_predecessors=True)
    assert [q.tolist() for q in walk_back(pred, 0, np.array([3, 1, 0]))] == [[0, 1, 2, 3], [0, 1], [0]]


def test_edge_table_removal_changes_shortest_path():
    t = EdgeTable(np.array([0, 1, 0]), np.array([1, 2, 2]), np.array([1.0, 1.0, 5.0]), np.array([1, 1, 0], bool), 3)
    t.alive[t.lookup(np.array([0]), np.array([1]))] = False
    _, pred = dijkstra(t.csr(), indices=0, return_predecessors=True)
    assert walk_back(pred, 0, np.array([2]))[0].tolist() == [0, 2]
    assert uses_edge([np.array([0, 1, 2])], 3, np.array([0 * 3 + 1]))
    assert not uses_edge([np.array([0, 2])], 3, np.array([0 * 3 + 1]))


def make_data():
    z = torch.randn(6, 192).numpy()
    return TeacherPaths(
        z, np.array([0, 1, 2, 3, 4, 5]), np.array([0, 4, 6]),
        np.array([0.0, 5.0, 7.0, 12.0, 0.0, 3.0]), device=D,
    ), z


def test_curve_interpolates_nodes_by_cumulative_cost():
    data, z = make_data()
    p = torch.tensor([0, 0, 0], device=D)
    first, last = torch.zeros(3, dtype=torch.long, device=D), torch.full((3,), 3, device=D)
    # cost 3 of 12 is inside the first edge (cost 5), 60% along it
    I, dI, s, g = data.curve(p, first, last, torch.tensor([0.25, 0.0, 1.0], device=D))
    assert np.allclose(I[0].cpu(), z[0] + 0.6 * (z[1] - z[0]), atol=1e-5)
    assert np.allclose(dI[0].cpu(), (z[1] - z[0]) / 5 * 12, atol=1e-4)
    assert np.allclose(I[1].cpu(), z[0], atol=1e-5) and np.allclose(I[2].cpu(), z[3], atol=1e-5)
    assert np.allclose(s[0].cpu(), z[0]) and np.allclose(g[0].cpu(), z[3])


def test_curve_stays_inside_its_path():
    data, z = make_data()
    p = torch.tensor([1], device=D)
    I, _, s, g = data.curve(p, torch.tensor([0], device=D), torch.tensor([1], device=D), torch.tensor([1.0], device=D))
    assert np.allclose(I[0].cpu(), z[5], atol=1e-5) and np.allclose(s[0].cpu(), z[4]) and np.allclose(g[0].cpu(), z[5])


def test_sampled_segments_are_valid():
    data, _ = make_data()
    # Path 1 has one hop, which teacher.py never emits, so sample only path 0.
    I, dI, s, g, tau = data.sample(2048, subset=torch.tensor([0], device=D))
    assert torch.isfinite(I).all() and torch.isfinite(dI).all()
    # A segment of at least 2 hops in a 3-hop path can start at node 0 or 1 only.
    assert ((s == data.z[0]).all(1) | (s == data.z[1]).all(1)).all()
    assert ((g == data.z[2]).all(1) | (g == data.z[3]).all(1)).all()


def test_velocity_target_is_derivative_of_x():
    I0, dI = torch.randn(4, 192), torch.randn(4, 192)
    tau, h, sigma = torch.tensor([0.1, 0.3, 0.5, 0.9]), 1e-3, 1.0
    torch.manual_seed(0)
    x0, v, eps = targets(I0, dI, tau, sigma)
    x1 = I0 + h * dI + (sigma * (tau + h) * (1 - tau - h))[:, None] * eps
    assert torch.allclose((x1 - x0) / h, v, atol=2e-2)


class Oracle(torch.nn.Module):
    """A perfect noiseless straight-line velocity field, with eta = 0."""

    def forward(self, x, tau, s, g):
        return g - s, torch.zeros_like(x)


def test_sampler_deterministic_endpoints_and_noise():
    s, g = torch.zeros(4, 192), torch.ones(4, 192)
    paths = sample_paths(Oracle(), s, g, sigma=1.0, steps=10, c=0.0)
    assert paths.shape == (4, 11, 192)
    assert torch.allclose(paths[:, 0], s) and torch.allclose(paths[:, -1], g, atol=1e-5)
    assert torch.allclose(paths[:, 5], 0.5 * g, atol=1e-5)
    noisy = sample_paths(Oracle(), s, g, sigma=1.0, steps=10, c=1.0)
    assert not torch.allclose(noisy[0, 5], noisy[1, 5])


def test_noisy_iterates_match_clean_at_the_ends():
    s, g = torch.zeros(2, 192), torch.ones(2, 192)
    clean, noisy = sample_paths(Oracle(), s, g, sigma=1.0, steps=10, c=1.0, return_noisy=True)
    assert clean.shape == noisy.shape == (2, 11, 192)
    assert torch.equal(clean[:, 0], noisy[:, 0]) and torch.equal(clean[:, -1], noisy[:, -1])


def test_trace_records_real_env_ids_when_envs_are_dropped():
    model = mock.Mock(encode=lambda info: {"emb": torch.zeros(len(info["pixels"]), 1, 192, device=D)})
    cem = mock.Mock()
    cem.solve.return_value = {"costs": [1.0, 2.0], "actions": torch.zeros(2, 5, 10)}
    solver = BridgeSolver(model, None, cem, sigma=1.0, r=5.0)
    pixels = torch.zeros(2, 1, 3, 4, 4)
    solver.solve({"pixels": pixels, "goal": pixels, "env_id": np.array([3, 7])})
    assert [t["env"] for t in solver.trace] == [3, 7]


def test_shortest_path_checks_cross_edges_lazily():
    # 0 -> 1 -> 2 costs 10; the unchecked cross edge 0 -> 2 costs 1 but fails its check.
    src, dst = np.array([0, 1, 0]), np.array([1, 2, 2])
    table = EdgeTable(src, dst, np.array([5.0, 5.0, 1.0]), np.array([False, False, True]), 3)
    calls = []

    def residual(u, v):
        calls.append((u.tolist(), v.tolist()))
        return np.array([99.0])

    path, cum = shortest_path(table, 0, 2, residual, threshold=1.0)
    assert path.tolist() == [0, 1, 2] and cum.tolist() == [0, 5, 10]
    assert calls == [([0], [2])]
    # The failed edge stays removed, so it is not checked again.
    shortest_path(table, 0, 2, residual, threshold=1.0)
    assert len(calls) == 1
    assert shortest_path(table, 2, 0, residual, threshold=1.0) is None


def test_snapped_subgoals_are_graph_nodes():
    nodes = torch.randn(50, 4)
    cem = mock.Mock()
    solver = BridgeSolver(None, Oracle(), cem, sigma=1.0, r=0.1, n_paths=3, steps=4, snap_nodes=nodes)
    out = solver.propose(torch.randn(2, 4), torch.randn(2, 4))
    assert out.shape == (6, 4)
    assert (torch.cdist(out, nodes).min(1).values < 1e-5).all()


def test_first_subgoal():
    path = torch.tensor([0.0, 1.0, 2.0, 3.0, 4.0])[None, :, None].repeat(3, 1, 1)
    assert first_subgoal(path, 2.5)[0].item() == 3.0
    assert first_subgoal(path, 99.0)[0].item() == 4.0  # never far enough: use the end


def test_subgoal_panel_shows_the_subgoal_in_force():
    entries = [{"subgoal": torch.zeros(192), "direct": False}, {"subgoal": torch.zeros(192), "direct": True}]
    decoder = mock.Mock(side_effect=lambda z: torch.stack([torch.full((3, 96, 96), v) for v in (0.0, 1.0)]).to(z.device))
    frames = subgoal_frames(entries, decoder, 60)
    assert frames.shape == (60, 224, 224, 3)
    assert frames[24, -1, -1, 0] == 0 and frames[25, -1, -1, 0] == 255  # replan at step 25
    assert (frames[59] == frames[25]).all()  # the last subgoal holds after planning stops


def test_panel_positions_match_library_layout():
    assert [panel_x(j) for j in range(3)] == [16, 254, 492]


def test_model_shapes_and_zero_init():
    m = BridgeDiT(depth=2, v_scale=3.0)
    z = torch.randn(5, 192)
    v, eta = m(z, torch.rand(5), z, z)
    assert v.shape == (5, 192) and eta.shape == (5, 192)
    assert v.abs().max() == 0 and eta.abs().max() == 0


def test_velocity_includes_the_noise_term():
    torch.manual_seed(0)
    m = BridgeDiT(depth=2, v_scale=3.0, sigma=2.0)
    torch.nn.init.ones_(m.head_eta.bias)
    z, tau = torch.randn(4, 192), torch.tensor([0.0, 0.25, 0.5, 1.0])
    v, eta = m(z, tau, z, z)
    assert torch.allclose(v, (2.0 * (1 - 2 * tau))[:, None] * eta)


def test_model_depends_on_all_inputs_after_training():
    torch.manual_seed(0)
    m = BridgeDiT(depth=2)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(100):
        x, s, g, t = torch.randn(32, 192), torch.randn(32, 192), torch.randn(32, 192), torch.rand(32)
        target = 0.3 * x + 0.3 * s + 0.3 * g + t[:, None]
        v, eta = m.forward_raw(x, t, s, g)
        ((v - target).pow(2).mean() + (eta - target).pow(2).mean()).backward()
        opt.step()
        opt.zero_grad()
    x, s, g, t = torch.randn(8, 192), torch.randn(8, 192), torch.randn(8, 192), torch.rand(8)
    base, _ = m(x, t, s, g)
    for alt in (m(x + 1, t, s, g), m(x, t * 0.5, s, g), m(x, t, s + 1, g), m(x, t, s, g + 1)):
        assert (alt[0] - base).abs().mean() > 1e-3


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
