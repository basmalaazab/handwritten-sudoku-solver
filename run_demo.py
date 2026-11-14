"""Integrated demo:  python run_demo.py [--seed 1] [--k 2] [--puzzle-seed 7]

Shows (1) the handwritten input, (2) what the NN read, (3) the AC-3 conflict, (4) the search trace,
(5) CSP pruning with and without FC / AC-3, (6) the system with the NN ON vs OFF, and (7) the solution.
Everything is also drawn in results/demo.png (how the three parts cooperate)."""
import argparse
import random

import numpy as np

from p7 import nn
from p7.csp import FULL, ac3, conflicts_of, fmt, make_puzzle, raw_domains, solve
from p7.data import Bank, build_mosaic, read_mosaic, run_dir, RES
from p7.eval import prepare_models
from p7.integration import solve_handwritten
from p7.search import CorrectionProblem, search


def sizes(dom):
    return np.array([d.bit_count() for d in dom]).reshape(9, 9)


def grid_text(ax, grid, hi=(), color='k', hi_color='red'):
    for i, v in enumerate(grid):
        if v:
            ax.text(i % 9, i // 9, v, ha='center', va='center', fontsize=11,
                    color=hi_color if i in hi else color, fontweight='bold' if i in hi else 'normal')


def lines(ax):
    for t in range(10):
        ax.axhline(t - .5, color='k' if t % 3 == 0 else 'lightgray', lw=1.8 if t % 3 == 0 else .5)
        ax.axvline(t - .5, color='k' if t % 3 == 0 else 'lightgray', lw=1.8 if t % 3 == 0 else .5)
    ax.set_xlim(-.5, 8.5)
    ax.set_ylim(8.5, -.5)
    ax.axis('off')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=1, help='network/bank seed (results/seed*/std)')
    ap.add_argument('--puzzle-seed', type=int, default=7)
    ap.add_argument('--k', type=int, default=2, help='injected misreads')
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()
    prepare_models(a.seed, with_overconf=False)
    bank, rd = Bank(a.seed, 'std'), run_dir(a.seed, 'std')
    model = nn.load_model(rd)

    rng = random.Random(a.puzzle_seed)
    puz, sol = make_puzzle(rng, 30)
    m = build_mosaic(puz, sol, bank, random.Random(a.puzzle_seed + a.k), a.k)
    grid, P = read_mosaic(m, predict_fn=lambda X: nn.softmax(nn.predict_logits(model, X), bank.T))
    print('=' * 70, '\n1. INPUT: handwritten mosaic (%d injected hard digits at %s)' % (
        a.k, [f'r{c // 9 + 1}c{c % 9 + 1}' for c in m.injected]))
    print('\n2. NN READING (blank cells by brightness rule):\n' + fmt(grid))
    conf = conflicts_of(grid)
    dom = raw_domains(grid)
    print('\n3. AC-3 on the read grid:', 'CONFLICT (a domain became empty)' if not ac3(dom) else 'no conflict',
          '| duplicate-digit conflicts:', [[f'r{c // 9 + 1}c{c % 9 + 1}' for c in cs] for cs in conf])

    print('\n4. SEARCH (cost = log p(read) - log p(new)); counters:')
    traces, results = {}, {}
    for algo in ('ucs', 'greedy', 'astar'):
        tr = []
        results[algo] = solve_handwritten(grid, P, algo=algo, trace=tr)
        traces[algo] = tr
        st = results[algo]['stats']
        print(f"  {algo:7s} status={results[algo]['status']} cost={results[algo]['cost']} "
              f"expanded={st['expanded']} generated={st['generated']} goal_tests={st['goal_tests']} "
              f"csp_calls={st['solver_calls']}")
    print('\n  UCS trace (order | corrections [cell->digit] | g | CSP goal?):')
    for t in traces['ucs'][:12]:
        print(f"   {t['order']:3d} | {[(f'r{c // 9 + 1}c{c % 9 + 1}', d) for c, d in t['state']]} | "
              f"g={t['g']:.3f} | {'GOAL' if t['goal'] else 'no'}")
    res = results['ucs']
    if res['status'] != 'solved':
        print('\n', res['message'], '\n least confident cells:', res['least_confident'])
        return
    print('\n  corrections chosen by UCS:')
    for c in res['corrections']:
        print('   ', c)
    corrected = res['problem'].grid_of(frozenset((c['cell'], c['new']) for c in res['corrections']))

    print('\n5. CSP PRUNING on the corrected puzzle (assignments / backtracks):')
    csp_rows = {}
    for mode, name in (('bt', 'BT'), ('bt_ac3', 'BT+AC3'), ('fc', 'BT+FC'), ('fc_ac3', 'BT+FC+AC3'), ('mac', 'BT+FC+MAC')):
        _, st, _ = solve(corrected, mode=mode, limit=1)
        csp_rows[name] = st
        print(f"   {name:11s} assignments={st['assignments']:6d} backtracks={st['backtracks']:6d} "
              f"AC-3 prunings={st['ac3_prunings']:4d} FC prunings={st['fc_prunings']:5d}")

    print('\n6. NN ON vs OFF (same read grid):')
    ok0, _ = CorrectionProblem(grid, P).goal_test(frozenset())
    print(f'   no correction            : {"solved" if ok0 else "NO SOLUTION"}')
    P_off = np.where(P > 0, 1 / 9 + np.random.RandomState(0).uniform(0, 1e-3, P.shape), 0)
    for i, v in enumerate(grid):
        if v:
            P_off[i, v - 1] += 0.01                         # keep the read digit as argmax, otherwise flat
    r_off = solve_handwritten(grid, P_off, algo='ucs')
    print(f"   NN OFF (flat costs)      : {r_off['status']}, exact={r_off['solution'] == sol}, "
          f"goal_tests={r_off['stats']['goal_tests']}")
    print(f"   NN ON  (probability costs): {res['status']}, exact={res['solution'] == sol}, "
          f"goal_tests={res['stats']['goal_tests']}")
    print('\n7. SOLUTION (corrected cells in the corrections list):\n' + fmt(res['solution']))
    print('exact solution recovered:', res['solution'] == sol)

    if a.no_plot:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 4, figsize=(19, 9))
    ax = ax.ravel()
    ax[0].imshow(1 - m.image(), cmap='gray', extent=(-.5, 8.5, 8.5, -.5))
    for c in m.injected:
        ax[0].add_patch(plt.Rectangle((c % 9 - .5, c // 9 - .5), 1, 1, fill=False, ec='red', lw=2))
    lines(ax[0])
    ax[0].set_title('1. Handwritten input (red = injected hard digit)', fontsize=10)
    conf_map = np.array([P[i].max() if grid[i] else np.nan for i in range(81)]).reshape(9, 9)
    im = ax[1].imshow(conf_map, cmap='RdYlGn', vmin=0.3, vmax=1)
    grid_text(ax[1], grid, hi=set(m.injected))
    lines(ax[1])
    fig.colorbar(im, ax=ax[1], fraction=.046)
    ax[1].set_title('2. NN reading: colour = max probability', fontsize=10)
    cm = np.zeros(81)
    for cs in conf:
        cm[list(cs)] = 1
    ax[2].imshow(cm.reshape(9, 9), cmap='Reds', vmin=0, vmax=1.6)
    grid_text(ax[2], grid, hi={c for cs in conf for c in cs})
    lines(ax[2])
    ax[2].set_title('3. AC-3 / duplicate conflicts (domain empties)', fontsize=10)
    for algo, mk, col in (('ucs', 'o-', 'C0'), ('greedy', 's--', 'C1'), ('astar', '^:', 'C2')):
        tr = traces[algo]
        ax[3].plot([t['order'] for t in tr], [t['g'] for t in tr], mk, color=col, ms=3, lw=.8,
                   label=f"{algo} ({len(tr)} tested)")
        goal = [t for t in tr if t['goal']]
        if goal:
            ax[3].scatter([goal[0]['order']], [goal[0]['g']], s=120, facecolors='none', edgecolors=col, lw=2)
    ax[3].set_xlabel('order in which states are tested by the CSP')
    ax[3].set_ylabel('path cost g (-log likelihood ratio)')
    ax[3].legend(fontsize=8)
    ax[3].set_title('4. Search trace (circle = goal found)', fontsize=10)
    d_raw = raw_domains(corrected)
    d_ac = d_raw[:]
    ac3(d_ac)
    for k_, (d, t) in enumerate(((d_raw, '5a. domain sizes before AC-3'), (d_ac, '5b. after AC-3'))):
        ax[4 + k_].imshow(sizes(d), cmap='viridis_r', vmin=1, vmax=9)
        for i in range(81):
            ax[4 + k_].text(i % 9, i // 9, int(d[i].bit_count()), ha='center', va='center', fontsize=8,
                            color='white' if d[i].bit_count() >= 5 else 'k')
        lines(ax[4 + k_])
        ax[4 + k_].set_title(t, fontsize=10)
    names = list(csp_rows)
    ax[6].bar(names, [csp_rows[n]['assignments'] for n in names], color=['C3', 'C1', 'C0', 'C2', 'C4'])
    ax[6].set_yscale('log')
    ax[6].set_ylabel('assignments')
    ax[6].tick_params(axis='x', labelrotation=30)
    ax[6].set_title('5c. CSP effort with / without FC and AC-3', fontsize=10)
    ax[7].imshow(np.zeros((9, 9)), cmap='gray', vmin=0, vmax=1)
    ax[7].imshow(np.ones((9, 9)), cmap='Greens', vmin=0, vmax=3)
    hi = {c['cell'] for c in res['corrections']}
    grid_text(ax[7], res['solution'], hi=hi, color='k', hi_color='red')
    lines(ax[7])
    ax[7].set_title('6. Solution (red = corrected cells)', fontsize=10)
    fig.tight_layout()
    fig.savefig(RES / 'demo.png', dpi=130)
    print('\nfigure written to results/demo.png')


if __name__ == '__main__':
    main()
