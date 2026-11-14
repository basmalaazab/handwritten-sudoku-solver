"""CSP part: 81 variables (cells), domain 1-9, givens fixed, all-different on rows/cols/boxes.

The solver has three SEPARATE switches (see `solve`):
  fc   : forward checking   - after each assignment remove the value from the peers' domains
  ac3  : 'none' | 'pre' | 'mac'
         'pre' = AC-3 once on the raw puzzle (detects conflicts, shrinks domains)
         'mac' = AC-3 once, then again after every assignment (maintaining arc consistency; needs fc)
  mrv  : choose the cell with the smallest domain (otherwise: first empty cell)
Plain backtracking (BT) = fc False, ac3 'none', mrv False.

Counters (dict `stats`):
  assignments   value assignments tried during search (givens are not counted)
  backtracks    assignments that were undone
  ac3_prunings  values removed by AC-3 (preprocessing + maintained)
  fc_prunings   values removed by forward checking
Domains are 9-bit masks (bit d-1 set <=> digit d possible).
"""
import random
from collections import deque

FULL = 0x1FF
ROWI = [i // 9 for i in range(81)]
COLI = [i % 9 for i in range(81)]
BOXI = [(i // 9) // 3 * 3 + (i % 9) // 3 for i in range(81)]
UNITS = ([[r * 9 + c for c in range(9)] for r in range(9)] +
         [[r * 9 + c for r in range(9)] for c in range(9)] +
         [[(br * 3 + r) * 9 + bc * 3 + c for r in range(3) for c in range(3)]
          for br in range(3) for bc in range(3)])
PEERS_L = []
for _i in range(81):
    _p = set()
    for _u in UNITS:
        if _i in _u:
            _p.update(_u)
    _p.discard(_i)
    PEERS_L.append(sorted(_p))

NO_SOLUTION_CONFLICT = 'No solution: the givens contain a duplicate digit (AC-3 empties a domain).'
NO_SOLUTION_DOMAIN = 'No solution: AC-3 emptied the domain of a cell (no digit fits it).'
NO_SOLUTION_SEARCH = 'No solution: the search space was exhausted without finding a consistent assignment.'
NO_SOLUTION_LIMIT = 'No answer: the node limit was reached before a solution or a proof of impossibility.'


def bit(d):
    return 1 << (d - 1)


def digits_of(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length()
        mask ^= low


# ------------------------------------------------------------- conflicts / AC-3
def conflicts_of(grid):
    """Duplicate-digit conflicts: list of cell tuples (>=2 cells with the same digit in one unit)."""
    out = []
    for u in UNITS:
        seen = {}
        for c in u:
            if grid[c]:
                seen.setdefault(grid[c], []).append(c)
        out.extend(tuple(v) for v in seen.values() if len(v) > 1)
    return out


def raw_domains(grid):
    return [bit(v) if v else FULL for v in grid]


def ac3(dom, cells=None, stats=None):
    """AC-3 for the binary != constraints. Modifies `dom` in place.
    cells=None: start with all arcs; otherwise only arcs (xk, c) for c in cells (used by MAC).
    Returns False as soon as a domain becomes empty."""
    if cells is None:
        q = deque((i, j) for i in range(81) for j in PEERS_L[i])
    else:
        q = deque((xk, c) for c in cells for xk in PEERS_L[c])
    while q:
        xi, xj = q.popleft()
        dj = dom[xj]
        if dj & (dj - 1) == 0 and dom[xi] & dj:          # xj fixed -> remove its value from xi
            dom[xi] &= ~dj
            if stats is not None:
                stats['ac3_prunings'] += 1
            if dom[xi] == 0:
                return False
            for xk in PEERS_L[xi]:
                if xk != xj:
                    q.append((xk, xi))
    return True


def ac3_detects_conflict(grid):
    return not ac3(raw_domains(grid))


# ------------------------------------------------------------- solver
def _fc_assign(dom, cell, val, stats=None):
    nd = dom[:]
    b = bit(val)
    nd[cell] = b
    for p in PEERS_L[cell]:
        if nd[p] & b:
            nd[p] &= ~b
            if stats is not None:
                stats['fc_prunings'] += 1
            if nd[p] == 0:
                return None
    return nd


def new_stats():
    return {'assignments': 0, 'backtracks': 0, 'ac3_prunings': 0, 'fc_prunings': 0, 'ac3_conflict': False}


def solve(grid, fc=True, ac3_mode='pre', mrv=None, limit=1, node_limit=None, rng=None, mode=None):
    """Return (solutions, stats, hit_node_limit).

    `mode` is a shortcut: 'bt' | 'bt_ac3' | 'fc' | 'fc_ac3' | 'mac'  (others given explicitly)."""
    if mode is not None:
        fc, ac3_mode, mrv = {'bt': (False, 'none', False), 'bt_ac3': (False, 'pre', False),
                             'fc': (True, 'none', True), 'fc_ac3': (True, 'pre', True),
                             'mac': (True, 'mac', True)}[mode]
    if mrv is None:
        mrv = fc
    if ac3_mode == 'mac' and not fc:
        raise ValueError("ac3_mode='mac' needs fc=True")
    stats = new_stats()
    dom = raw_domains(grid)
    if ac3_mode in ('pre', 'mac'):
        if not ac3(dom, stats=stats):
            stats['ac3_conflict'] = True
            return [], stats, False
    sols, hit = [], [False]

    def over():
        stats['assignments'] += 1
        if node_limit and stats['assignments'] > node_limit:
            hit[0] = True
            return True
        return False

    if not fc:                                           # ------- plain BT (optionally on AC-3-reduced domains)
        g = list(grid)
        rows, cols, boxes = [0] * 9, [0] * 9, [0] * 9
        for i, v in enumerate(g):
            if v:
                b = bit(v)
                if (rows[ROWI[i]] | cols[COLI[i]] | boxes[BOXI[i]]) & b:
                    return [], stats, False
                rows[ROWI[i]] |= b
                cols[COLI[i]] |= b
                boxes[BOXI[i]] |= b
        empties = [i for i, v in enumerate(g) if v == 0]

        def rec(k):
            if k == len(empties):
                sols.append(g[:])
                return len(sols) >= limit
            i = empties[k]
            r, c, bx = ROWI[i], COLI[i], BOXI[i]
            cand = dom[i] & ~(rows[r] | cols[c] | boxes[bx])
            for v in digits_of(cand):
                if over():
                    return True
                b = bit(v)
                g[i] = v
                rows[r] |= b
                cols[c] |= b
                boxes[bx] |= b
                if rec(k + 1):
                    return True
                g[i] = 0
                rows[r] ^= b
                cols[c] ^= b
                boxes[bx] ^= b
                stats['backtracks'] += 1
            return False

        rec(0)
        return sols, stats, hit[0]

    # ----------------------------------------------------- FC (+ MRV, + MAC)
    assigned = 0
    for i, v in enumerate(grid):                         # propagate givens (not counted)
        if v:
            if not dom[i] & bit(v):
                return [], stats, False
            dom = _fc_assign(dom, i, v, stats)
            if dom is None:
                return [], stats, False
            assigned |= 1 << i

    def rec2(dom, assigned):
        best, bc = -1, 10
        for i in range(81):
            if not (assigned >> i) & 1:
                if not mrv:
                    best, bc = i, dom[i].bit_count()
                    break
                c = dom[i].bit_count()
                if c < bc:
                    bc, best = c, i
                    if c <= 1:
                        break
        if best < 0:
            sols.append([d.bit_length() for d in dom])
            return len(sols) >= limit
        if bc == 0:
            return False
        vals = list(digits_of(dom[best]))
        if rng is not None:
            rng.shuffle(vals)
        for v in vals:
            if over():
                return True
            nd = _fc_assign(dom, best, v, stats)
            if nd is not None and ac3_mode == 'mac':
                changed = [p for p in PEERS_L[best] if nd[p] != dom[p]]
                if not ac3(nd, cells=changed, stats=stats):
                    nd = None
            if nd is not None and rec2(nd, assigned | (1 << best)):
                return True
            stats['backtracks'] += 1
        return False

    rec2(dom, assigned)
    return sols, stats, hit[0]


def solve_or_message(grid, **kw):
    """Solve and return (solution or None, message, stats). The message is explicit for impossible puzzles."""
    sols, st, hit = solve(grid, **kw)
    if sols:
        return sols[0], 'Solved.', st
    if hit:
        return None, NO_SOLUTION_LIMIT, st
    if conflicts_of(grid):
        return None, NO_SOLUTION_CONFLICT, st
    return None, (NO_SOLUTION_DOMAIN if st['ac3_conflict'] else NO_SOLUTION_SEARCH), st


def count_solutions(grid, upto=2):
    return len(solve(grid, mode='fc_ac3', limit=upto)[0])


# ------------------------------------------------------------- generator
def random_solution(rng):
    return solve([0] * 81, mode='fc', limit=1, rng=rng)[0][0]


def make_puzzle(rng, givens=30):
    """Full grid by randomised backtracking; remove cells while the solution stays unique.
    Returns (puzzle, solution); may stop above `givens` if uniqueness blocks further removal."""
    sol = random_solution(rng)
    grid, n = sol[:], 81
    cells = list(range(81))
    rng.shuffle(cells)
    for c in cells:
        if n <= givens:
            break
        v = grid[c]
        grid[c] = 0
        if count_solutions(grid, 2) == 1:
            n -= 1
        else:
            grid[c] = v
    return grid, sol


def fmt(grid):
    return '\n'.join(' '.join(str(grid[r * 9 + c]) if grid[r * 9 + c] else '.' for c in range(9))
                     for r in range(9))
