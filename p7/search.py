"""Search part: correct the first reading with the cheapest set of digit changes.

State  : frozenset of (cell, new_digit); each cell corrected at most once; |state| <= max_corr (3).
Action : add one correction (cell, alt_digit), cell among the n_cand least-confident given cells,
         alt_digit among that cell's top m_alts alternatives.
Cost   : cost(cell -> d') = log p(read) - log p(d')  >= 0  (read digit is the argmax).
         g(state) = sum of costs = -log likelihood ratio, so UCS returns the most likely correction set.
Goal   : corrected puzzle is solvable by the CSP ('solvable') or has exactly one solution ('unique').
A* h   : every duplicate-digit conflict needs >=1 of its cells changed. Take a family of pairwise
         cell-disjoint conflicts; each needs its own changed cell, so h = sum over the family of the
         cheapest still-possible change among the conflict's cells (admissible). If the family is larger
         than the remaining correction budget, or a conflict has no changeable cell, h = inf (prune).
"""
import heapq
import itertools
import math
import time

import numpy as np

from .csp import conflicts_of, solve

INF = float('inf')


class CorrectionProblem:
    def __init__(self, read_grid, P, n_cand=15, m_alts=3, max_corr=3, goal='solvable',
                 solver_mode='fc_ac3', solver_node_limit=50000):
        self.read = list(read_grid)
        self.max_corr, self.goal, self.mode = max_corr, goal, solver_mode
        self.node_limit = solver_node_limit
        given = [i for i, v in enumerate(read_grid) if v]
        self.cands = sorted(given, key=lambda i: P[i].max())[:n_cand]
        self.alts, self.min_cost = {}, {}
        for i in self.cands:
            lp = np.log(np.clip(P[i], 1e-12, 1.0))
            r = read_grid[i]
            order = [int(d) for d in np.argsort(-P[i]) + 1 if d != r][:m_alts]
            self.alts[i] = [(d, max(0.0, float(lp[r - 1] - lp[d - 1]))) for d in order]
            self.min_cost[i] = min(c for _, c in self.alts[i])
        self.cost_of = {(i, d): c for i, a in self.alts.items() for d, c in a}
        self._cache = {}
        self.goal_tests = self.solver_calls = self.solver_nodes = self.solver_timeouts = 0

    # -- helpers
    def grid_of(self, state):
        g = self.read[:]
        for c, d in state:
            g[c] = d
        return g

    def g_cost(self, state):
        return sum(self.cost_of[s] for s in state)

    def goal_test(self, state):
        """-> (ok, solution or None). Cached; counts every test, and CSP calls separately."""
        self.goal_tests += 1
        if state in self._cache:
            return self._cache[state]
        g = self.grid_of(state)
        if conflicts_of(g):                       # duplicate digit => trivially unsolvable (same verdict as AC-3)
            res = (False, None)
        else:
            sols, st, hit = solve(g, mode=self.mode, limit=1 if self.goal == 'solvable' else 2,
                                  node_limit=self.node_limit)
            self.solver_calls += 1
            self.solver_nodes += st['assignments']
            self.solver_timeouts += int(hit and not sols)
            ok = len(sols) >= 1 if self.goal == 'solvable' else len(sols) == 1
            res = (ok, sols[0] if ok else None)
        self._cache[state] = res
        return res

    def heuristic(self, state):
        g = self.grid_of(state)
        changed = {c for c, _ in state}
        confs = []
        for cells in conflicts_of(g):
            best = min((self.min_cost[x] for x in cells if x not in changed and x in self.min_cost),
                       default=INF)
            if best == INF:
                return INF
            confs.append((best, cells))
        confs.sort(key=lambda t: -t[0])
        used, total, n = set(), 0.0, 0
        for best, cells in confs:
            if used.isdisjoint(cells):
                used.update(cells)
                total += best
                n += 1
        return INF if n > self.max_corr - len(state) else total

    def successors(self, state):
        used = {c for c, _ in state}
        for cell in self.cands:
            if cell in used:
                continue
            for d, _ in self.alts[cell]:
                yield state | {(cell, d)}

    def describe(self, state):
        """Correction log rows: (cell index, row, col, read digit, new digit, cost)."""
        return [{'cell': c, 'row': c // 9 + 1, 'col': c % 9 + 1, 'read': self.read[c], 'new': d,
                 'cost': round(self.cost_of[(c, d)], 4)} for c, d in sorted(state)]


def search(prob, algo='ucs', time_limit=60.0, trace=None):
    """algo in {'ucs','greedy','astar'}. Returns dict with the result and effort counters."""
    t0 = time.time()
    cnt = itertools.count()
    start = frozenset()

    def key(state, g):
        if algo == 'ucs':
            return (g,), 0.0
        h = prob.heuristic(state)
        if h == INF:
            return None, h
        return ((h, g) if algo == 'greedy' else (g + h,)), h

    k0, _ = key(start, 0.0)
    heap = [(k0, next(cnt), start, 0.0)]
    seen = {start}
    expanded = generated = 0
    timeout = False
    while heap:
        if time.time() - t0 > time_limit:
            timeout = True
            break
        key_, _, state, g = heapq.heappop(heap)
        ok, sol = prob.goal_test(state)
        if trace is not None:
            trace.append({'order': len(trace) + 1, 'state': sorted(state), 'g': g, 'priority': key_[0],
                          'goal': ok})
        if ok:
            return _res(prob, True, state, g, sol, expanded, generated, t0, False)
        if len(state) >= prob.max_corr:
            continue
        expanded += 1
        for ns in prob.successors(state):
            if ns in seen:
                continue
            seen.add(ns)
            gn = prob.g_cost(ns)
            k, _ = key(ns, gn)
            if k is None:
                continue
            generated += 1
            heapq.heappush(heap, (k, next(cnt), ns, gn))
    return _res(prob, False, None, None, None, expanded, generated, t0, timeout)


def random_search(prob, budget, rng):
    """Random correction: same candidate pool, same goal test, at most `budget` goal tests
    (the uncorrected puzzle counts as the first one)."""
    t0 = time.time()
    start = frozenset()
    ok, sol = prob.goal_test(start)
    tested = 1
    if ok:
        return _res(prob, True, start, 0.0, sol, 0, 0, t0, False, tested)
    seen, attempts = {start}, 0
    n_c = len(prob.cands)
    while tested < budget and attempts < 50 * max(budget, 10):
        attempts += 1
        size = rng.randint(1, min(prob.max_corr, n_c))
        cells = rng.sample(prob.cands, size)
        state = frozenset((c, rng.choice(prob.alts[c])[0]) for c in cells)
        if state in seen:
            continue
        seen.add(state)
        ok, sol = prob.goal_test(state)
        tested += 1
        if ok:
            return _res(prob, True, state, prob.g_cost(state), sol, 0, 0, t0, False, tested)
    return _res(prob, False, None, None, None, 0, 0, t0, False, tested)


def _res(prob, found, state, g, sol, expanded, generated, t0, timeout, tested=None):
    return {'found': found, 'state': state, 'cost': g, 'solution': sol, 'expanded': expanded,
            'generated': generated, 'goal_tests': tested if tested is not None else prob.goal_tests,
            'solver_calls': prob.solver_calls, 'solver_nodes': prob.solver_nodes,
            'solver_timeouts': prob.solver_timeouts, 'time': time.time() - t0, 'timeout': timeout,
            'n_corrections': len(state) if state is not None else None}
