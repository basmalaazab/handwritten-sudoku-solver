"""Experiments (>= 5 seeds, mean and standard deviation).

Exp1  UCS vs Greedy vs A*                    : effort counters + recovery, 0-4 injected misreads
Exp2  BT / BT+AC3 / BT+FC / BT+FC+AC3 / MAC  : assignments, backtracks, time; puzzles with 25/30/35 givens
Exp3  NN-guided vs no correction vs random   : exact-solution recovery (same candidate pool and budget)
Exp4  extensions                             : uniqueness goal, raw vs temperature-scaled probabilities
                                               (std CNN and the over-confident CNN), natural (uninjected) errors
"""
import csv
import json
import random
import time

import numpy as np

from . import nn
from .csp import make_puzzle, solve
from .data import Bank, build_bank, build_mosaic, read_mosaic, run_dir, RES
from .integration import solve_handwritten
from .search import CorrectionProblem, random_search, search

LEVELS = [0, 1, 2, 3, 4]
CSP_VARIANTS = [('bt', 'BT'), ('bt_ac3', 'BT+AC3'), ('fc', 'BT+FC'), ('fc_ac3', 'BT+FC+AC3'), ('mac', 'BT+FC+MAC')]


def mean(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


def gen_puzzles(n, givens, seed):
    rng = random.Random(seed)
    return [make_puzzle(rng, givens) for _ in range(n)]


def mosaic_for(puz, sol, bank, seed, j, k, natural=False):
    return build_mosaic(puz, sol, bank, random.Random(seed * 1_000_000 + j * 100 + k), k, natural)


# ------------------------------------------------------------------ experiments (one seed)
def exp1(bank, puzzles, seed, log_rows, summary_rows):
    out = {}
    for k in LEVELS:
        mosaics = [mosaic_for(p, s, bank, seed, j, k) for j, (p, s) in enumerate(puzzles)]
        reads = [read_mosaic(m) for m in mosaics]
        per = {}
        for algo in ('ucs', 'greedy', 'astar'):
            rows = []
            for j, (m, (g, P)) in enumerate(zip(mosaics, reads)):
                r = solve_handwritten(g, P, algo=algo)
                exact = r['solution'] == m.solution
                st = r['stats']
                rows.append({'found': r['status'] == 'solved', 'exact': bool(exact), 'cost': r['cost'],
                             **{x: st[x] for x in ('expanded', 'generated', 'goal_tests', 'solver_calls', 'time')}})
                summary_rows.append({'seed': seed, 'level': k, 'algo': algo, 'puzzle': j, 'status': r['status'],
                                     'exact': int(exact), 'n_corrections': len(r['corrections']), 'cost': r['cost'],
                                     'ac3_conflict_on_read_grid': int(r['ac3_conflict']),
                                     'injected': ' '.join(f'r{c // 9 + 1}c{c % 9 + 1}' for c in m.injected),
                                     'least_confident': json.dumps(r.get('least_confident', []))})
                if algo == 'ucs':
                    for c in r['corrections']:
                        log_rows.append({'seed': seed, 'level': k, 'puzzle': j, 'row': c['row'], 'col': c['col'],
                                         'read_digit': c['read'], 'new_digit': c['new'], 'cost': c['cost'],
                                         'injected_here': int(c['cell'] in m.injected), 'exact': int(exact)})
            per[algo] = rows
        lvl = {}
        for algo, rows in per.items():
            lvl[algo] = {'solved_rate': mean([r['found'] for r in rows]),
                         'exact_recovery': mean([r['exact'] for r in rows]),
                         'nodes_expanded': mean([r['expanded'] for r in rows]),
                         'nodes_generated': mean([r['generated'] for r in rows]),
                         'goal_tests': mean([r['goal_tests'] for r in rows]),
                         'csp_calls': mean([r['solver_calls'] for r in rows]),
                         'time_s': mean([r['time'] for r in rows])}
        u, a, g_ = per['ucs'], per['astar'], per['greedy']
        both = [(x, y) for x, y in zip(u, a) if x['found'] and y['found']]
        lvl['astar_equals_ucs_rate'] = mean([abs(x['cost'] - y['cost']) < 1e-9 for x, y in both])
        gb = [(x, y) for x, y in zip(u, g_) if x['found'] and y['found']]
        lvl['greedy_suboptimal_rate'] = mean([y['cost'] > x['cost'] + 1e-9 for x, y in gb])
        out[k] = lvl
    return out


def exp2(n, seed, node_limit=200_000):
    out = {}
    for givens in (25, 30, 35):
        pz = gen_puzzles(n, givens, seed + 1000 + givens)
        res = {'actual_givens': mean([sum(1 for v in p if v) for p, _ in pz])}
        for mode, name in CSP_VARIANTS:
            a, b, ts, to, pr = [], [], [], 0, []
            for p, s in pz:
                t = time.time()
                sols, st, hit = solve(p, mode=mode, limit=1, node_limit=node_limit)
                ts.append(time.time() - t)
                a.append(st['assignments'])
                b.append(st['backtracks'])
                pr.append(st['ac3_prunings'])
                to += int(hit)
                if not hit:
                    assert sols and sols[0] == s
            res[name] = {'assignments': mean(a), 'backtracks': mean(b), 'ac3_prunings': mean(pr),
                         'time_ms': 1000 * mean(ts), 'node_limit_hits': to}
        out[givens] = res
    return out


def exp3(bank, puzzles, seed):
    out = {}
    for k in LEVELS:
        stats = {s: [] for s in ('no_correction', 'random_equal_budget', 'random_budget_200', 'nn_guided_ucs')}
        for j, (p, s) in enumerate(puzzles):
            m = mosaic_for(p, s, bank, seed, j, k)
            g, P = read_mosaic(m)
            ok, sol = CorrectionProblem(g, P).goal_test(frozenset())
            stats['no_correction'].append((ok, sol == s))
            r = search(CorrectionProblem(g, P), 'ucs')
            stats['nn_guided_ucs'].append((r['found'], r['solution'] == s))
            for name, B in (('random_equal_budget', r['goal_tests']), ('random_budget_200', 200)):
                rr = random_search(CorrectionProblem(g, P), B, random.Random(seed * 7919 + j * 31 + k))
                stats[name].append((rr['found'], rr['solution'] == s))
        out[k] = {n_: {'solved_rate': mean([a for a, _ in v]), 'exact_recovery': mean([b for _, b in v])}
                  for n_, v in stats.items()}
    return out


def _ucs_variants(bank, puzzles, seed, levels, variants, natural=False):
    out = {}
    for k in levels:
        row = {}
        for name, cfg in variants.items():
            ex, fo, gt = [], [], []
            for j, (p, s) in enumerate(puzzles):
                m = mosaic_for(p, s, bank, seed, j, k, natural)
                g, P = read_mosaic(m, source=cfg['source'])
                r = solve_handwritten(g, P, algo='ucs', goal=cfg['goal'])
                ex.append(r['solution'] == s)
                fo.append(r['status'] == 'solved')
                gt.append(r['stats']['goal_tests'])
            row[name] = {'solved_rate': mean(fo), 'exact_recovery': mean(ex), 'goal_tests': mean(gt)}
        out[k] = row
    return out


def exp4(bank_std, bank_over, puzzles, seed):
    var = {'solvable_cal': dict(goal='solvable', source='cal'), 'unique_cal': dict(goal='unique', source='cal'),
           'solvable_raw': dict(goal='solvable', source='raw'), 'unique_raw': dict(goal='unique', source='raw')}
    res = {'std_cnn': _ucs_variants(bank_std, puzzles, seed, [1, 2, 3], var),
           'natural_errors': _ucs_variants(bank_std, puzzles, seed, [0], {'solvable_cal': var['solvable_cal'],
                                                                         'unique_cal': var['unique_cal']}, True)}
    if bank_over is not None:
        res['overconfident_cnn'] = _ucs_variants(bank_over, puzzles, seed, [1, 2, 3],
                                                 {k: var[k] for k in ('solvable_cal', 'solvable_raw')})
    return res


# ------------------------------------------------------------------ orchestration
def _csv(path, rows):
    if rows:
        with open(path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


def prepare_models(seed, with_overconf=True):
    for variant in (['std', 'overconf'] if with_overconf else ['std']):
        rd = run_dir(seed, variant)
        if not (rd / 'bank.npz').exists():
            t = time.time()
            nn.train(seed, variant, baseline=(variant == 'std'))
            build_bank(seed, variant)
            print(f'seed {seed} {variant}: trained + bank in {time.time() - t:.0f}s', flush=True)


def run_seed(seed, n=100, givens=30, with_overconf=True):
    rd = run_dir(seed, 'std')
    f = rd / 'experiments.json'
    if f.exists():
        return json.load(open(f))
    prepare_models(seed, with_overconf)
    bank = Bank(seed, 'std')
    bank_over = Bank(seed, 'overconf') if with_overconf else None
    puzzles = gen_puzzles(n, givens, seed)
    log_rows, summary_rows, t0 = [], [], time.time()
    R = {'seed': seed, 'n': n, 'givens': givens}
    R['exp1'] = exp1(bank, puzzles, seed, log_rows, summary_rows)
    R['exp2'] = exp2(n, seed)
    R['exp3'] = exp3(bank, puzzles, seed)
    R['exp4'] = exp4(bank, bank_over, puzzles, seed)
    R['classifier'] = json.load(open(rd / 'classifier_report.json'))
    if with_overconf:
        R['classifier_overconf'] = json.load(open(run_dir(seed, 'overconf') / 'classifier_report.json'))
    _csv(rd / 'correction_log.csv', log_rows)
    _csv(rd / 'puzzle_summary.csv', summary_rows)
    json.dump(R, open(f, 'w'), indent=1)
    print(f'seed {seed}: experiments in {time.time() - t0:.0f}s', flush=True)
    return json.load(open(f))


def _agg(vals):
    vals = [v for v in vals if v is not None]
    return {'mean': float(np.mean(vals)), 'std': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            'n': len(vals)}


def aggregate(runs):
    """Replace every numeric leaf by {mean, std, n} across the per-seed results."""
    def rec(items):
        first = items[0]
        if isinstance(first, dict):
            return {k: rec([it[k] for it in items if k in it]) for k in first
                    if k not in ('confusion_matrix_cnn_test', 'seed', 'n', 'givens', 'sizes')}
        if isinstance(first, (int, float)) and not isinstance(first, bool):
            return _agg(items)
        return None
    return rec(runs)


def ms(d, p=3):
    return f"{d['mean']:.{p}f} ± {d['std']:.{p}f}"


def tables(A, seeds):
    L = []
    L.append(f'Seeds: {seeds}  (mean ± sample std over seeds; each seed = new NN + new puzzles/mosaics)\n')
    L.append('### Classifier (test, digits 1-9)\n\n| model | accuracy | ECE | NLL |\n|---|---|---|---|')
    for key, nm in (('cnn_raw', 'CNN raw'), ('cnn_temp_scaled', 'CNN + temperature scaling'), ('logreg', 'Logistic regression')):
        c = A['classifier'][key]
        L.append(f"| {nm} | {ms(c['test_acc'], 4)} | {ms(c['ECE'], 4)} | {ms(c['NLL'], 4)} |")
    L.append(f"\nTemperature (std CNN): {ms(A['classifier']['temperature'], 3)}")
    if 'classifier_overconf' in A:
        c = A['classifier_overconf']
        L.append(f"\nOver-confident CNN: acc {ms(c['cnn_raw']['test_acc'], 4)}, ECE raw {ms(c['cnn_raw']['ECE'], 4)} -> "
                 f"scaled {ms(c['cnn_temp_scaled']['ECE'], 4)}, NLL raw {ms(c['cnn_raw']['NLL'], 3)} -> scaled "
                 f"{ms(c['cnn_temp_scaled']['NLL'], 3)}, T = {ms(c['temperature'], 2)}")
    L.append('\n### Exp1: exact-solution recovery / goal tests / nodes expanded / nodes generated\n')
    L.append('| misreads | algo | exact recovery | solved | goal tests | expanded | generated |\n|---|---|---|---|---|---|---|')
    for k in LEVELS:
        for a in ('ucs', 'greedy', 'astar'):
            e = A['exp1'][str(k)][a] if str(k) in A['exp1'] else A['exp1'][k][a]
            L.append(f"| {k} | {a} | {ms(e['exact_recovery'], 2)} | {ms(e['solved_rate'], 2)} | "
                     f"{ms(e['goal_tests'], 1)} | {ms(e['nodes_expanded'], 1)} | {ms(e['nodes_generated'], 1)} |")
    L.append('\n| misreads | A* cost == UCS cost (rate) | Greedy more expensive than UCS (rate) |\n|---|---|---|')
    for k in LEVELS:
        e = A['exp1'][str(k)]
        L.append(f"| {k} | {ms(e['astar_equals_ucs_rate'], 3)} | {ms(e['greedy_suboptimal_rate'], 3)} |")
    L.append('\n### Exp2: CSP variants (mean assignments / backtracks / time ms / node-limit hits)\n')
    L.append('| givens | variant | assignments | backtracks | AC-3 prunings | time (ms) | limit hits/100 |\n|---|---|---|---|---|---|---|')
    for g in ('25', '30', '35'):
        for _, nm in CSP_VARIANTS:
            e = A['exp2'][g][nm]
            L.append(f"| {g} | {nm} | {ms(e['assignments'], 1)} | {ms(e['backtracks'], 1)} | {ms(e['ac3_prunings'], 1)} | "
                     f"{ms(e['time_ms'], 2)} | {ms(e['node_limit_hits'], 1)} |")
    L.append('\n### Exp3: exact-solution recovery by strategy\n')
    names = list(A['exp3']['0'])
    L.append('| misreads | ' + ' | '.join(names) + ' |\n|---|' + '---|' * len(names))
    for k in LEVELS:
        L.append(f'| {k} | ' + ' | '.join(ms(A['exp3'][str(k)][n_]['exact_recovery'], 2) for n_ in names) + ' |')
    L.append('\n### Exp4: extensions (exact recovery)\n')
    for grp, tag in (('std_cnn', 'std CNN'), ('overconfident_cnn', 'over-confident CNN'), ('natural_errors', 'natural errors, no injection')):
        if grp not in A['exp4']:
            continue
        lv = list(A['exp4'][grp])
        cols = list(A['exp4'][grp][lv[0]])
        L.append(f'\n{tag}\n\n| misreads | ' + ' | '.join(cols) + ' |\n|---|' + '---|' * len(cols))
        for k in lv:
            L.append(f'| {k} | ' + ' | '.join(ms(A['exp4'][grp][k][c]['exact_recovery'], 2) for c in cols) + ' |')
    return '\n'.join(L)


def plots(A):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    g = lambda d: (d['mean'], d['std'])
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for algo, mk in (('ucs', 'o-'), ('greedy', 's--'), ('astar', '^:')):
        for a_, key in zip(ax, ('exact_recovery', 'goal_tests', 'nodes_expanded')):
            m = [g(A['exp1'][str(k)][algo][key]) for k in LEVELS]
            a_.errorbar(LEVELS, [x[0] for x in m], [x[1] for x in m], fmt=mk, capsize=3, label=algo)
    for a_, t in zip(ax, ('Exact-solution recovery', 'Goal tests (mean)', 'Nodes expanded (mean)')):
        a_.set_title(t)
        a_.set_xlabel('injected misreads')
        a_.set_xticks(LEVELS)
        a_.legend()
    ax[1].set_yscale('log')
    ax[2].set_yscale('log')
    fig.tight_layout()
    fig.savefig(RES / 'exp1.png', dpi=140)

    fig, ax = plt.subplots(figsize=(8, 4))
    w, gs = 0.16, ['25', '30', '35']
    for i, (_, nm) in enumerate(CSP_VARIANTS):
        m = [g(A['exp2'][x][nm]['assignments']) for x in gs]
        ax.bar(np.arange(3) + i * w, [x[0] for x in m], w, yerr=[x[1] for x in m], capsize=2, label=nm)
    ax.set_xticks(np.arange(3) + 2 * w, [f'{x} givens' for x in gs])
    ax.set_yscale('log')
    ax.set_ylabel('mean assignments')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / 'exp2.png', dpi=140)

    fig, ax = plt.subplots(figsize=(8, 4))
    names = list(A['exp3']['0'])
    for i, nm in enumerate(names):
        m = [g(A['exp3'][str(k)][nm]['exact_recovery']) for k in LEVELS]
        ax.bar(np.arange(5) + i * 0.2, [x[0] for x in m], 0.2, yerr=[x[1] for x in m], capsize=2, label=nm)
    ax.set_xticks(np.arange(5) + 0.3, LEVELS)
    ax.set_xlabel('injected misreads')
    ax.set_ylabel('exact-solution recovery')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(RES / 'exp3.png', dpi=140)


def main(seeds=(1, 2, 3, 4, 5), n=100, givens=30, with_overconf=True):
    runs = [run_seed(s, n, givens, with_overconf) for s in seeds]
    A = aggregate(runs)
    json.dump(A, open(RES / 'summary_mean_std.json', 'w'), indent=1)
    open(RES / 'tables.md', 'w').write(tables(A, list(seeds)))
    plots(A)
    print(open(RES / 'tables.md').read())


if __name__ == '__main__':
    main()
