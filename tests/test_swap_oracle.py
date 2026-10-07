import torch

from experiments.hierarchical_mae.swap_oracle import search_space, token_sets, search
from experiments.hierarchical_mae.swap_selection import proposals


def test_unique_sets_and_single_swap_equivalence():
    states, children = search_space(4, 6, 4)
    assert [sum(len(s[0]) == i for s in states) for i in range(5)] == [1, 24, 90, 80, 15]
    activity = torch.tensor([[9., 8., 7., 6., 5., 5., 5., 4., 3., 2., 1., 0.]])
    valid = torch.ones_like(activity, dtype=torch.bool)
    ids, baseline, removed, added = token_sets(activity, valid, 6, 4, 6, states)
    original, _, _ = proposals(activity, valid, 6, 4, 6)
    assert torch.equal(ids[:, :25], original)
    assert len({tuple(row.tolist()) for row in ids[0]}) == 210
    base = set(ids[0, 0].tolist())
    for i, state in enumerate(states):
        selected = set(ids[0, i].tolist())
        assert len(selected) == 6
        assert len(base-selected) == len(selected-base) == len(state[0])


def test_beam_crosses_barrier_greedy_stops_and_exhaustive_bounds():
    states, children = search_space(2, 2, 2)
    losses = torch.tensor([1., 2., 3., 4., 5., .1])
    greedy, counts = search(losses, states, children, 2, method='greedy')
    beam, _ = search(losses, states, children, 2, beam_width=2)
    exact, _ = search(losses, states, children, 2, method='exhaustive')
    assert greedy == [0, 0, 0] and counts == [1, 5, 5]
    assert beam == exact == [0, 0, 5]
    for seed in range(10):
        values = torch.rand(len(states), generator=torch.Generator().manual_seed(seed))
        for method in ('greedy', 'beam', 'exhaustive'):
            selected, _ = search(values, states, children, 2, method=method)
            assert (values[selected][1:] <= values[selected][:-1]).all()
            for depth, index in enumerate(selected):
                eligible = [i for i, s in enumerate(states) if len(s[0]) <= depth]
                assert len(states[index][0]) <= depth
                assert values[index] >= values[eligible].min()
