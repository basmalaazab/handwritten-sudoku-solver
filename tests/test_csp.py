"""CSP tests: normal / edge / impossible."""
import random

import pytest

from p7.csp import (NO_SOLUTION_CONFLICT, NO_SOLUTION_DOMAIN, ac3_detects_conflict, conflicts_of,
                    count_solutions, make_puzzle, solve, solve_or_message)

MODES = ['bt', 'bt_ac3', 'fc', 'fc_ac3', 'mac']


def valid(grid):
    return all(v for v in grid) and not conflicts_of(grid)


@pytest.fixture(scope='module')
def puzzle():
    return make_puzzle(random.Random(11), 30)


# ---------------------------------------------------------------- normal (5)
@pytest.mark.parametrize('mode', MODES)
def test_normal_all_modes_find_the_unique_solution(puzzle, mode):
    p, s = puzzle
    sols, st, hit = solve(p, mode=mode, limit=2)
    assert not hit and sols == [s]


def test_normal_generator_gives_unique_puzzle_with_requested_givens(puzzle):
    p, _ = puzzle
    assert sum(1 for v in p if v) == 30 and count_solutions(p, 2) == 1


def test_normal_forward_checking_needs_fewer_assignments_than_bt(puzzle):
    p, _ = puzzle
    bt = solve(p, mode='bt')[1]['assignments']
    fc = solve(p, mode='fc')[1]['assignments']
    assert fc < bt


def test_normal_ac3_alone_prunes_values_and_helps_bt(puzzle):
    p, _ = puzzle
    _, st, _ = solve(p, mode='bt_ac3')
    assert st['ac3_prunings'] > 0
    assert st['assignments'] <= solve(p, mode='bt')[1]['assignments']


def test_normal_switches_are_independent(puzzle):
    p, s = puzzle
    for fc in (False, True):
        for ac in ('none', 'pre') + (('mac',) if fc else ()):
            for mrv in (False, True):
                sols, st, _ = solve(p, fc=fc, ac3_mode=ac, mrv=mrv)
                assert sols == [s]
                assert (st['fc_prunings'] > 0) == fc and (st['ac3_prunings'] > 0) == (ac != 'none')


# ---------------------------------------------------------------- edge (3+)
def test_edge_empty_grid_is_solvable():
    sols, _, _ = solve([0] * 81, mode='fc_ac3')
    assert valid(sols[0])


def test_edge_complete_valid_grid_returns_itself_without_search(puzzle):
    _, s = puzzle
    sols, st, _ = solve(s, mode='fc_ac3')
    assert sols == [s] and st['assignments'] == 0


def test_edge_single_blank_cell(puzzle):
    _, s = puzzle
    g = s[:]
    g[40] = 0
    assert solve(g, mode='bt')[0] == [s]


def test_edge_puzzle_with_two_solutions_is_detected():
    s = make_puzzle(random.Random(5), 81)[1]
    g = s[:]
    for i in range(0, 81, 3):
        g[i] = 0                                         # remove a lot: many solutions
    assert count_solutions(g, 2) == 2


# ---------------------------------------------------------------- impossible (2+)
def test_impossible_duplicate_givens_give_clear_message():
    g = [0] * 81
    g[0] = g[8] = 5                                      # two 5s in row 1
    assert ac3_detects_conflict(g)
    for mode in MODES:
        sol, msg, _ = solve_or_message(g, mode=mode)
        assert sol is None and msg == NO_SOLUTION_CONFLICT and msg.startswith('No solution')


def test_impossible_cell_without_candidates_gives_clear_message():
    g = [0] * 81
    g[:8] = [1, 2, 3, 4, 5, 6, 7, 8]                     # row 1 needs a 9 in the last cell ...
    g[17] = 9                                            # ... but column 9 already has a 9
    assert not conflicts_of(g)                           # no duplicate digit, still impossible
    sol, msg, _ = solve_or_message(g, mode='fc_ac3')
    assert sol is None and msg == NO_SOLUTION_DOMAIN
    sol, msg, _ = solve_or_message(g, mode='bt')
    assert sol is None and msg.startswith('No solution')
