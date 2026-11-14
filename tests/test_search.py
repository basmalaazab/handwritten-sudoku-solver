"""Search tests on a small example with a known answer, plus pipeline tests (no trained model needed)."""
import math
import random

import numpy as np
import pytest

from p7.csp import conflicts_of, make_puzzle
from p7.integration import solve_handwritten
from p7.search import CorrectionProblem, search


def fake_probs(grid, rng, weak=None):
    """Every given cell: 0.9 on its digit.  Cells in `weak` = {cell: (read, p_read, true, p_true)}."""
    P = np.zeros((81, 9))
    for i, v in enumerate(grid):
        if v:
            P[i] = 0.1 / 8
            P[i, v - 1] = 0.9
    for i, (r, pr, t, pt) in (weak or {}).items():
        P[i] = (1 - pr - pt) / 7
        P[i, r - 1], P[i, t - 1] = pr, pt
    return P


@pytest.fixture(scope='module')
def case():
    """Known answer: cell c is read as d' (a duplicate of a peer) with p=0.52 while the true digit has 0.45."""
    p, s = make_puzzle(random.Random(21), 30)
    givens = [i for i, v in enumerate(p) if v]
    for c in givens:
        row_peers = [j for j in givens if j != c and j // 9 == c // 9 and p[j] != p[c]]
        if row_peers:
            wrong = p[row_peers[0]]
            break
    read = p[:]
    read[c] = wrong
    P = fake_probs(read, None, {c: (wrong, 0.52, p[c], 0.45)})
    return p, s, read, P, c


def test_normal_known_answer_ucs(case):
    p, s, read, P, c = case
    prob = CorrectionProblem(read, P)
    r = search(prob, 'ucs')
    assert r['found'] and r['state'] == frozenset({(c, p[c])})
    assert r['cost'] == pytest.approx(math.log(0.52 / 0.45), abs=1e-6)
    assert r['solution'] == s


@pytest.mark.parametrize('algo', ['ucs', 'greedy', 'astar'])
def test_normal_all_algorithms_find_the_known_correction(case, algo):
    p, s, read, P, c = case
    r = search(CorrectionProblem(read, P), algo)
    assert r['found'] and r['solution'] == s and r['expanded'] >= 1 and r['generated'] >= r['expanded']


def test_normal_astar_never_expands_more_than_ucs_and_has_same_cost(case):
    _, _, read, P, _ = case
    u, a = search(CorrectionProblem(read, P), 'ucs'), search(CorrectionProblem(read, P), 'astar')
    assert a['cost'] == pytest.approx(u['cost']) and a['expanded'] <= u['expanded']


def test_normal_heuristic_is_admissible(case):
    _, _, read, P, _ = case
    prob = CorrectionProblem(read, P)
    opt = search(CorrectionProblem(read, P), 'ucs')['cost']
    assert prob.heuristic(frozenset()) <= opt + 1e-9


def test_normal_costs_are_non_negative(case):
    _, _, read, P, _ = case
    prob = CorrectionProblem(read, P)
    assert all(c >= 0 for c in prob.cost_of.values())


def test_normal_pipeline_corrects_two_misreads():
    p, s = make_puzzle(random.Random(8), 30)
    g = [i for i, v in enumerate(p) if v]
    read, weak = p[:], {}
    for c in g[:2]:
        row = [j for j in g if j // 9 == c // 9 and p[j] != p[c] and j != c]
        wrong = p[row[0]]
        read[c] = wrong
        weak[c] = (wrong, 0.5, p[c], 0.4)
    out = solve_handwritten(read, fake_probs(read, None, weak))
    assert out['status'] == 'solved' and out['solution'] == s and len(out['corrections']) == 2
    assert out['ac3_conflict']


def test_edge_clean_reading_needs_no_correction():
    p, s = make_puzzle(random.Random(9), 30)
    out = solve_handwritten(p, fake_probs(p, None))
    assert out['status'] == 'solved' and out['corrections'] == [] and out['stats']['expanded'] == 0


def test_edge_correction_budget_zero_never_corrects(case):
    _, _, read, P, _ = case
    out = solve_handwritten(read, P, max_corr=0)
    assert out['status'] == 'unreadable'


def test_impossible_too_many_misreads_is_reported_unreadable():
    p, s = make_puzzle(random.Random(8), 30)
    g = [i for i, v in enumerate(p) if v]
    read, weak = p[:], {}
    for c in g[:6]:                                      # 6 duplicates, only 3 corrections allowed
        row = [j for j in g if j // 9 == c // 9 and p[j] != p[c] and j != c]
        if row:
            read[c] = p[row[0]]
            weak[c] = (read[c], 0.5, p[c], 0.4)
    out = solve_handwritten(read, fake_probs(read, None, weak))
    assert out['status'] == 'unreadable' and out['message'].startswith('No solution')
    assert len(out['least_confident']) == 5


def test_impossible_conflict_among_confident_cells_is_unreadable():
    p, _ = make_puzzle(random.Random(8), 30)
    g = [i for i, v in enumerate(p) if v]
    c = g[0]
    j = next(j for j in g if j // 9 == c // 9 and p[j] != p[c])
    read = p[:]
    read[c] = p[j]                                       # duplicate, but the network is 99.9% sure of both
    P = fake_probs(read, None)
    P[c] = 0.0001
    P[c, read[c] - 1] = 0.9992
    out = solve_handwritten(read, P, n_cand=0)           # no cell may be corrected
    assert out['status'] == 'unreadable' and 'No solution' in out['message']
