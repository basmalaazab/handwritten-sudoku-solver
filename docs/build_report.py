"""python docs/build_report.py  ->  docs/P7_report.pdf  (asserts <= 12 pages). Numbers come from results/summary_mean_std.json."""
import json
import sys
from pathlib import Path

from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / 'results'
A = json.load(open(RES / 'summary_mean_std.json'))
OUT = ROOT / 'docs' / 'P7_report.pdf'

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
FD = '/usr/share/fonts/truetype/dejavu/'
for n_, f_ in (('DV', 'DejaVuSans.ttf'), ('DV-B', 'DejaVuSans-Bold.ttf'), ('DV-I', 'DejaVuSans-Oblique.ttf'), ('DV-BI', 'DejaVuSans-BoldOblique.ttf')):
    pdfmetrics.registerFont(TTFont(n_, FD + f_))
pdfmetrics.registerFontFamily('DV', normal='DV', bold='DV-B', italic='DV-I', boldItalic='DV-BI')
ss = getSampleStyleSheet()
B = ParagraphStyle('B', parent=ss['BodyText'], fontName='DV', fontSize=9.8, leading=13, alignment=TA_JUSTIFY, spaceAfter=4)
H1 = ParagraphStyle('H1', parent=ss['Heading2'], fontName='DV-B', fontSize=12.5, spaceBefore=7, spaceAfter=3)
H2 = ParagraphStyle('H2', parent=ss['Heading3'], fontName='DV-B', fontSize=10.5, spaceBefore=4, spaceAfter=2)
CAP = ParagraphStyle('CAP', parent=B, fontSize=8, leading=9.5, textColor=colors.HexColor('#444444'))
TT = ParagraphStyle('TT', parent=ss['Title'], fontName='DV-B', fontSize=16, spaceAfter=2)
CELL = ParagraphStyle('CELL', parent=B, fontSize=8.2, leading=10, alignment=0, spaceAfter=0)


def m(d, p=2):
    return f"{d['mean']:.{p}f}±{d['std']:.{p}f}"


def P(t, s=B):
    return Paragraph(t, s)


def fig(name, w_cm):
    from PIL import Image as PI
    w, h = PI.open(RES / name).size
    return Image(str(RES / name), width=w_cm * cm, height=w_cm * cm * h / w)


def table(rows, widths=None, head=1):
    rows = [[Paragraph(str(c), CELL) for c in r] for r in rows]
    t = Table(rows, colWidths=widths, repeatRows=head)
    t.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), .3, colors.grey), ('BACKGROUND', (0, 0), (-1, head - 1), colors.HexColor('#e8eef7')),
                           ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'), ('TOPPADDING', (0, 0), (-1, -1), 1.5),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5)]))
    return t


E1, E2, E3, E4, C = A['exp1'], A['exp2'], A['exp3'], A['exp4'], A['classifier']
S = []
S += [P('P7 — Handwritten Sudoku Reader and Solver', TT),
      P('Neural network + search + constraint satisfaction. Report (course project P7). Seeds 1-5; MNIST digits 1-9.', CAP)]

# ---------------------------------------------------------------- 1
S += [P('1. Problem', H1),
      P('A handwritten Sudoku is read digit by digit by a neural network. A single misread digit usually makes the puzzle '
        'invalid (two equal digits in a row) or changes its solution. Goal: find the <i>most likely</i> set of misread digits '
        '(at most 3), correct them and solve the puzzle; if no such set exists, report “unreadable” and list the least '
        'confident cells. The three parts cooperate: the <b>NN</b> turns an image into probabilities, '
        'the <b>search</b> turns probabilities into costs and proposes correction sets in order of likelihood, and the '
        '<b>CSP</b> decides whether a corrected puzzle is solvable (Fig. 1).'),
      fig('architecture.png', 16.5), P('Fig. 1. Pipeline. Blank cells are found by a brightness rule, not by the NN.', CAP)]

# ---------------------------------------------------------------- 2
S += [P('2. Data and design choices', H1),
      P('<b>Real data:</b> MNIST (60,000 train / 10,000 test images, 28×28, MD5-verified by <font face="Courier">make_data.py</font>). '
        'Digit 0 is dropped (Sudoku uses 1-9); the 60,000 training images are split 50,000/10,000 train/validation with a seeded '
        'permutation; the test set is untouched (9,020 test images of digits 1-9). <b>Generated data:</b> Sudoku puzzles with a '
        'unique solution (randomised backtracking, then cell removal while uniqueness holds; 30 givens in Exp. 1/3/4, 25/30/35 in '
        'Exp. 2), the 9×9 mosaics, the blank cells and the “hard” digits. This is not photographed Sudoku; see '
        '<font face="Courier">docs/DATA_DICTIONARY.md</font>.'),
      P('<b>Injection rule (hard digits).</b> A well-trained CNN has almost nothing to correct, so misreads are injected. For each '
        'digit d we take natural test images of d plus augmented copies (rotation 15-45°, shear ±0.3, scale 0.85-1.1, 8×8 erase, noise) '
        'and accept an image iff the calibrated CNN gives argmax ≠ d, d as 2nd or 3rd choice, max p ≤ 0.80 and p(d) ≥ 0.05 '
        '(60 images per digit, about 85 % augmented). Background digits of a mosaic are test images the CNN reads correctly, so '
        '“k misreads” means exactly k injected cells. Consequence: the correct digit is always among the top-3 choices, so the '
        'search can reach it; recovery rates are therefore an upper-bound style estimate, not a field measurement.'),
      P('<b>Seeds.</b> Seed s fixes the network initialisation, the train/validation split, the puzzles, the augmentation and '
        'the placement of the injected cells. All results are mean ± sample standard deviation over seeds 1-5 '
        '(100 puzzles per misread level per seed).')]

# ---------------------------------------------------------------- 3
cr, ct, cl = C['cnn_raw'], C['cnn_temp_scaled'], C['logreg']
co = A['classifier_overconf']
S += [P('3. Neural network', H1),
      P('Small CNN (Conv16-pool-Conv32-pool-Dense64, dropout 0.3, Adam, 5 epochs, logits output; CPU, ≈6 s/epoch) against a '
        'logistic regression on raw pixels (baseline). Calibration is measured by the expected calibration error (ECE, 15 bins) and '
        'the negative log-likelihood; temperature scaling fits one temperature T on the validation set.'),
      table([['model (test, digits 1-9)', 'accuracy', 'ECE', 'NLL'],
             ['CNN, raw softmax', m(cr['test_acc'], 4), m(cr['ECE'], 4), m(cr['NLL'], 4)],
             [f"CNN + temperature scaling (T = {m(C['temperature'], 2)})", m(ct['test_acc'], 4), m(ct['ECE'], 4), m(ct['NLL'], 4)],
             ['Logistic regression (baseline)', m(cl['test_acc'], 4), m(cl['ECE'], 4), m(cl['NLL'], 4)],
             [f"Over-confident CNN (no dropout, 15 epochs), raw → scaled (T = {m(co['temperature'], 2)})", m(co['cnn_raw']['test_acc'], 4),
              f"{m(co['cnn_raw']['ECE'], 4)} → {m(co['cnn_temp_scaled']['ECE'], 4)}",
              f"{m(co['cnn_raw']['NLL'], 3)} → {m(co['cnn_temp_scaled']['NLL'], 3)}"]],
            [8.3 * cm, 2.6 * cm, 3.6 * cm, 3.4 * cm]),
      Spacer(1, 3),
      P('The CNN is 6.5 points more accurate than the baseline and its NLL is 8× lower. The standard CNN is already well calibrated '
        '(T ≈ 1), so temperature scaling changes almost nothing; the deliberately over-confident variant needs T ≈ 1.8 and its ECE '
        'halves. The probabilities matter because the cost of a correction is derived from them (next section).'),
      fig('classifier.png', 17), P('Fig. 2. Confusion matrix (seed 1) and reliability diagrams before/after temperature scaling.', CAP)]

# ---------------------------------------------------------------- 4
S += [P('4. Search: finding the most likely corrections', H1),
      P('<b>State</b> = set of corrections {(cell, new digit)}, each cell at most once, ≤ 3 corrections (stored as a frozenset). '
        '<b>Action</b> = add one correction; candidates are the 15 least-confident given cells and each cell’s 3 best alternative '
        'digits (branching ≤ 45). <b>Cost</b> of changing a cell from the read digit r to d′ is log p(r) − log p(d′) ≥ 0 '
        '(r is the argmax), so g(state) = Σ costs = −log of the likelihood ratio and UCS returns the most likely correction set. '
        '<b>Goal</b> = the corrected puzzle is solvable (CSP: FC + AC-3); optional extension: exactly one solution. Results of the CSP '
        'are cached per state. Counters: nodes expanded, nodes generated, goal tests, CSP calls.'),
      P('<b>Algorithms.</b> UCS orders by g. Greedy orders by (h, g). A* orders by g + h. <b>Heuristic.</b> Every duplicate-digit '
        'conflict needs at least one of its cells changed; conflicts that share no cell need different changed cells. h = sum, over a '
        'family of pairwise cell-disjoint conflicts, of the cheapest still-possible change of any cell of the conflict. h = ∞ if the '
        'family is larger than the remaining budget or a conflict has no changeable cell (the true cost is then also ∞). h is a '
        'lower bound, hence admissible. g(state) does not depend on the path, so a closed state never gets a better g and A* graph '
        'search stays optimal even if h is not consistent. Unit test: on a hand-made case (read 7 as 1 with 52 % vs 45 %) all three '
        'algorithms return the single correction of cost ln(0.52/0.45).')]

# ---------------------------------------------------------------- 5
S += [P('5. CSP: backtracking, forward checking, AC-3', H1),
      P('81 variables, domain 1-9, givens fixed, all-different on rows/columns/boxes. The three techniques are separate switches: '
        '<b>BT</b> (static order, consistency check on assignment), <b>FC</b> (remove the value from the peers’ domains; with MRV '
        'variable ordering) and <b>AC-3</b> (<i>none</i>, <i>pre</i> = once before search, <i>mac</i> = also after every assignment). '
        'AC-3 detects conflicts without search: two equal givens in a unit empty a domain (the integration uses this to log the '
        'conflict and to give the “no solution” message). Counters: assignments, backtracks, AC-3 prunings, FC prunings.')]

# ---------------------------------------------------------------- 6
S += [P('6. Integration and demo', H1),
      P('<font face="Courier">integration.solve_handwritten</font> reads the grid (NN probabilities), runs AC-3, calls the search and, '
        'if no correction set with ≤ 3 changes gives a solvable puzzle, returns status “unreadable” with the message '
        '“No solution: the puzzle is unreadable …” and the five least-confident cells. <font face="Courier">python run_demo.py</font> '
        'prints each stage and draws Fig. 3: input, NN confidence, AC-3 conflicts, search trace, domain sizes before/after AC-3, CSP '
        'effort, solution. In the demo the UCS needs 21 goal tests (1 CSP call), Greedy 3 and A* 4, all with cost 3.42; without NN '
        'probabilities (flat costs) 198 goal tests are needed and the answer is wrong.'),
      fig('demo.png', 17.2), P('Fig. 3. Output of run_demo.py (seed 1, puzzle seed 7, 2 injected misreads).', CAP)]

# ---------------------------------------------------------------- 7
rows = [['misreads', 'algo', 'exact recovery', 'solved', 'goal tests', 'expanded', 'generated']]
for k in '01234':
    for a in ('ucs', 'greedy', 'astar'):
        e = E1[k][a]
        rows.append([k if a == 'ucs' else '', {'ucs': 'UCS', 'greedy': 'Greedy', 'astar': 'A*'}[a], m(e['exact_recovery']), m(e['solved_rate']),
                     m(e['goal_tests'], 1), m(e['nodes_expanded'], 1), m(e['nodes_generated'], 1)])
S += [P('7. Experiments (5 seeds, mean ± std)', H1), P('Exp. 1 — UCS vs Greedy vs A*', H2), table(rows, [1.6 * cm, 1.8 * cm, 3 * cm, 2.6 * cm, 3 * cm, 2.6 * cm, 3 * cm]),
      Spacer(1, 3),
      P(f"A* returned exactly the UCS cost in {m(E1['3']['astar_equals_ucs_rate'], 3)} of the cases at every level (it is optimal) while "
        f"testing {E1['3']['ucs']['goal_tests']['mean'] / E1['3']['astar']['goal_tests']['mean']:.1f}× fewer states at 3 misreads and "
        f"{E1['4']['ucs']['goal_tests']['mean'] / E1['4']['astar']['goal_tests']['mean']:.0f}× fewer at 4 misreads (4,987 vs 29). Greedy is the cheapest "
        f"but not optimal: it returned a more expensive correction set in {m(E1['2']['greedy_suboptimal_rate'], 3)}, {m(E1['3']['greedy_suboptimal_rate'], 3)} and "
        f"{m(E1['4']['greedy_suboptimal_rate'], 3)} of the cases at 2, 3, 4 misreads, and its exact recovery is lower at 2 and 3 misreads "
        f"({m(E1['2']['greedy']['exact_recovery'])} vs {m(E1['2']['ucs']['exact_recovery'])}; {m(E1['3']['greedy']['exact_recovery'])} vs {m(E1['3']['ucs']['exact_recovery'])}). "
        "With 4 misreads and a budget of 3 corrections the true puzzle cannot be restored (exact recovery 0); in 63 % of the cases the system "
        "still returns a solvable but <i>wrong</i> puzzle, which is why exact recovery, not “solved”, is the metric. The level-1 row also shows "
        "why recovery is below 1 for one misread: the misread puzzle is sometimes still solvable (with another solution), so the goal "
        "“solvable” accepts it without correction (fixed by the uniqueness goal, Exp. 4)."),
      fig('exp1.png', 16.5), P('Fig. 4. Exp. 1: recovery, goal tests and expanded nodes (error bars = std over seeds).', CAP)]

rows = [['givens', 'variant', 'assignments', 'backtracks', 'AC-3 prunings', 'time (ms)', 'limit hits/100']]
for g in ('25', '30', '35'):
    for nm in ('BT', 'BT+AC3', 'BT+FC', 'BT+FC+AC3', 'BT+FC+MAC'):
        e = E2[g][nm]
        rows.append([g if nm == 'BT' else '', nm, m(e['assignments'], 1), m(e['backtracks'], 1), m(e['ac3_prunings'], 1), m(e['time_ms'], 2), m(e['node_limit_hits'], 1)])
S += [P('Exp. 2 — BT vs BT+FC vs BT+FC+AC-3 (and AC-3 variants)', H2), table(rows, [1.4 * cm, 2.4 * cm, 3.3 * cm, 3.3 * cm, 2.7 * cm, 2.4 * cm, 2.2 * cm]),
      Spacer(1, 3),
      P('Same 100 puzzles per seed and level for all variants; limit 200,000 assignments (counted as 200,000 when hit, so BT values at 25 givens are lower bounds). '
        f"Forward checking cuts the assignments by {E2['25']['BT']['assignments']['mean'] / E2['25']['BT+FC']['assignments']['mean']:.0f}× (25 givens), "
        f"{E2['30']['BT']['assignments']['mean'] / E2['30']['BT+FC']['assignments']['mean']:.0f}× (30) and {E2['35']['BT']['assignments']['mean'] / E2['35']['BT+FC']['assignments']['mean']:.0f}× (35), and 5.2 of 100 BT runs hit the limit at 25 givens. "
        'AC-3 used once as preprocessing helps plain BT (1.7× fewer assignments at 25 givens, 3.7× at 30, 5.6× at 35) but gives <i>exactly the same</i> assignments as FC alone when FC is on: FC with MRV already '
        'performs that pruning, and its extra cost shows in the time column. AC-3 pays off when it is maintained after every assignment (MAC): '
        f"{E2['25']['BT+FC']['assignments']['mean'] / E2['25']['BT+FC+MAC']['assignments']['mean']:.1f}× fewer assignments than FC at 25 givens, but each assignment is more expensive so wall-clock time is not better in our Python implementation.")]

rows = [['misreads', 'no correction', 'random, same budget as UCS', 'random, budget 200', 'NN-guided (UCS)']]
for k in '01234':
    rows.append([k] + [m(E3[k][n]['exact_recovery']) for n in ('no_correction', 'random_equal_budget', 'random_budget_200', 'nn_guided_ucs')])
S += [P('Exp. 3 — NN-guided vs no correction vs random correction (exact recovery)', H2), table(rows, [2 * cm, 3 * cm, 4.6 * cm, 3.5 * cm, 3.5 * cm]),
      Spacer(1, 3),
      P('Random correction draws sets of 1-3 corrections from the <i>same</i> candidate pool and the same goal test; budget = number of goal tests UCS used on that '
        'puzzle (first row of data) or a fixed 200. Without the probabilities the system almost never recovers the puzzle (at 200 trials only 50 % for one misread and ≈ 0 for two or three), '
        'while the NN-guided search needs 2-19 goal tests on average. Without any correction the recovery is 0 as soon as one digit is misread.'),
      Table([[fig('exp2.png', 8.4), fig('exp3.png', 8.4)]]), P('Fig. 5. Left: mean assignments of the CSP variants (Exp. 2, log scale). Right: exact recovery by strategy (Exp. 3). Error bars = std over seeds.', CAP)]

sc, su, so = E4['std_cnn'], E4['std_cnn'], E4['overconfident_cnn']
S += [P('Exp. 4 — Extensions', H2),
      table([['misreads', 'solvable goal', 'unique goal', 'raw probs (solvable)', 'over-confident CNN: raw / scaled'],
             *[[k, m(sc[k]['solvable_cal']['exact_recovery']), m(sc[k]['unique_cal']['exact_recovery']), m(sc[k]['solvable_raw']['exact_recovery']),
                f"{m(so[k]['solvable_raw']['exact_recovery'])} / {m(so[k]['solvable_cal']['exact_recovery'])}"] for k in '123']],
            [2 * cm, 3.2 * cm, 3.2 * cm, 3.6 * cm, 5 * cm]),
      Spacer(1, 3),
      P('<b>Uniqueness goal</b> (exactly one solution) raises the recovery from 0.84/0.72/0.61 to 0.95/0.90/0.85 for 1/2/3 misreads because a wrong digit that leaves the puzzle solvable '
        'but ambiguous is now rejected. <b>Temperature scaling does not change the recovery</b>, even for the over-confident CNN: the cost log p(r) − log p(d′) = (z_r − z_d′)/T, so every cost is divided by the same T, '
        'the order of all correction sets is unchanged and UCS returns the same set. Calibration matters for the <i>meaning</i> of the costs (a likelihood ratio) and for the injection rule, '
        f"not for the ranking. <b>Natural errors</b> (no injection, background digits are random test images): exact recovery {m(E4['natural_errors']['0']['solvable_cal']['exact_recovery'])} with the solvable goal "
        f"and {m(E4['natural_errors']['0']['unique_cal']['exact_recovery'])} with the unique goal.")]

# ---------------------------------------------------------------- 8
S += [P('8. Tests and reproducibility', H1),
      P('<font face="Courier">python -m pytest tests -q</font>: 31 tests — 20 normal (all solver modes find the unique solution, generator uniqueness, FC &lt; BT, '
        'switch independence, known-answer search for UCS/Greedy/A*, A* = UCS cost, heuristic ≤ optimal cost, non-negative costs, pipeline with 2 misreads, brightness rule, '
        'temperature scaling), 7 edge (empty grid, complete grid, one blank, two solutions, clean reading needs no correction, budget 0, ECE of one-hot) and 4 impossible '
        '(duplicate givens and a cell without candidates give “No solution: …”; 6 misreads and a conflict among confident cells give “unreadable”). '
        'Pinned requirements, exact commands and seeds are in the README; the MNIST MD5 values are checked automatically.')]

# ---------------------------------------------------------------- 9
S += [P('9. Limits', H1),
      P('(1) Synthetic mosaics: MNIST digits glued into a grid, no photographs, lines, rotation or varying stroke width. (2) Hard digits are mostly augmented and always have the correct '
        'digit among the top-3 choices, so recovery rates are optimistic; real misreads can be confident and wrong. (3) At most 3 corrections and 15 candidate cells: more misreads, or a '
        'misread outside the 15 least-confident cells, cannot be repaired. (4) “Solvable” is a weak goal; the uniqueness goal helps but still cannot detect a misread that yields another '
        'unique puzzle. (5) The heuristic only sees duplicate-digit conflicts; if the corrupted puzzle has no duplicate, h = 0 and A* behaves like UCS. (6) Python timings are '
        'implementation-specific; the counters are the portable measure. (7) Five seeds give standard deviations of 1-6 points; differences such as Greedy vs UCS recovery at 3 misreads '
        '(0.57 vs 0.61) are about one standard deviation.')]

# ---------------------------------------------------------------- 10
S += [P('10. Conclusions', H1),
      P('The three components solve the problem together: the CNN supplies calibrated costs, A* finds the same cheapest correction set as UCS with 2-170× fewer tested states, and the CSP '
        '(forward checking, optionally maintained AC-3) makes each test cheap. Using the NN probabilities turns an almost hopeless random search into a recovery of 0.84/0.72/0.61 '
        '(0.95/0.90/0.85 with the uniqueness goal) for 1/2/3 misreads, and the system says “unreadable” when it cannot repair the grid. '
        'Two lessons: exact recovery, not “solvable”, is the right metric, and a single temperature cannot change UCS ranking, so calibration is useful for interpretation, not for this search.')]

doc = SimpleDocTemplate(str(OUT), pagesize=A4, leftMargin=1.8 * cm, rightMargin=1.8 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm,
                        title='P7 Handwritten Sudoku Reader and Solver — report')
doc.build(S)
n = len(PdfReader(str(OUT)).pages)
print('pages:', n)
if n > 12:
    sys.exit('report longer than 12 pages')
