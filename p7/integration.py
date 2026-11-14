"""Integration: NN reads the grid -> AC-3 looks for a conflict -> search tries the cheapest corrections ->
CSP tests each corrected puzzle (first success wins) -> otherwise 'unreadable' + least-confident cells."""
from .csp import ac3_detects_conflict, conflicts_of
from .search import CorrectionProblem, search

UNREADABLE_MSG = ('No solution: the puzzle is unreadable - no set of at most {k} corrections of the '
                  'candidate cells gives a valid puzzle. Check the least-confident cells below.')


def solve_handwritten(read_grid, P, algo='ucs', goal='solvable', n_cand=15, m_alts=3, max_corr=3,
                      time_limit=60.0, trace=None):
    prob = CorrectionProblem(read_grid, P, n_cand=n_cand, m_alts=m_alts, max_corr=max_corr, goal=goal)
    res = search(prob, algo, time_limit=time_limit, trace=trace)
    out = {'ac3_conflict': ac3_detects_conflict(read_grid),
           'conflicts_on_read_grid': [tuple(c) for c in conflicts_of(read_grid)],
           'status': 'solved' if res['found'] else 'unreadable',
           'message': 'Solved.' if res['found'] else UNREADABLE_MSG.format(k=max_corr),
           'solution': res['solution'], 'cost': res['cost'],
           'stats': {k: v for k, v in res.items() if k not in ('state', 'solution')},
           'corrections': prob.describe(res['state']) if res['found'] else [],
           'problem': prob}
    if not res['found']:
        given = [i for i, v in enumerate(read_grid) if v]
        low = sorted(given, key=lambda i: P[i].max())[:5]
        out['least_confident'] = [{'row': i // 9 + 1, 'col': i % 9 + 1, 'read': read_grid[i],
                                   'confidence': round(float(P[i].max()), 3)} for i in low]
    return out
