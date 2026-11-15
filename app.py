"""Web app:  streamlit run app.py     (or run_app.bat / run_app.ps1 one level above this folder)

Generated mode : make a unique Sudoku, glue handwritten MNIST digits into a mosaic, inject k misread digits,
                 let the system read / repair / solve it.
Upload mode    : upload a picture of a 9x9 handwritten grid (see the hints on the page).
Needs the trained model of results/seed{S}/std (the zip contains seed 1)."""
import io
import random
import sys
from pathlib import Path

# Ensure local 'p7' package is discoverable
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

import matplotlib
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

from p7 import nn
from p7.csp import conflicts_of, make_puzzle, solve
from p7.data import Bank, Mosaic, build_mosaic, read_mosaic, run_dir
from p7.integration import solve_handwritten

st.set_page_config(page_title='Handwritten Sudoku Reader and Solver', layout='wide')


# ------------------------------------------------------------------ cached resources
@st.cache_resource(show_spinner='Loading digit bank ...')
def get_bank(seed):
    return Bank(seed, 'std')


@st.cache_resource(show_spinner='Loading the CNN (TensorFlow) ...')
def get_model(seed):
    return nn.load_model(run_dir(seed, 'std'))


@st.cache_data(show_spinner='Generating a unique puzzle ...')
def get_puzzle(seed, givens):
    return make_puzzle(random.Random(seed), givens)


def available_seeds():
    from p7.data import RES
    return sorted(int(p.parent.parent.name[4:]) for p in RES.glob('seed*/std/bank.npz')) or [1]


# ------------------------------------------------------------------ drawing helpers
def draw_grid(ax):
    for t in range(10):
        ax.axhline(t - .5, color='k' if t % 3 == 0 else 'lightgray', lw=1.8 if t % 3 == 0 else .5)
        ax.axvline(t - .5, color='k' if t % 3 == 0 else 'lightgray', lw=1.8 if t % 3 == 0 else .5)
    ax.set_xlim(-.5, 8.5)
    ax.set_ylim(8.5, -.5)
    ax.axis('off')


def digits(ax, grid, hi=(), hi_color='red'):
    for i, v in enumerate(grid):
        if v:
            ax.text(i % 9, i // 9, v, ha='center', va='center', fontsize=13,
                    color=hi_color if i in hi else 'black', fontweight='bold' if i in hi else 'normal')


def fig_mosaic(m, mark=()):
    f, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(1 - m.image(), cmap='gray', extent=(-.5, 8.5, 8.5, -.5))
    for c in mark:
        ax.add_patch(plt.Rectangle((c % 9 - .5, c // 9 - .5), 1, 1, fill=False, ec='red', lw=2.5))
    draw_grid(ax)
    f.tight_layout()
    return f


def fig_reading(grid, P, conflict_cells):
    f, ax = plt.subplots(figsize=(4.2, 4.2))
    cm = np.array([P[i].max() if grid[i] else np.nan for i in range(81)]).reshape(9, 9)
    ax.imshow(cm, cmap='RdYlGn', vmin=0.3, vmax=1)
    digits(ax, grid, hi=conflict_cells)
    draw_grid(ax)
    f.tight_layout()
    return f


def fig_solution(sol, corrected_cells):
    f, ax = plt.subplots(figsize=(4.2, 4.2))
    ax.imshow(np.ones((9, 9)), cmap='Greens', vmin=0, vmax=3)
    digits(ax, sol, hi=corrected_cells)
    draw_grid(ax)
    f.tight_layout()
    return f


def cell_name(c):
    return f'r{c // 9 + 1}c{c % 9 + 1}'


# ------------------------------------------------------------------ upload preprocessing
def image_to_cells(img, blank_thr):
    """Grayscale -> 252x252 -> 81 cells of 28x28, white digit on black (MNIST style)."""
    a = np.asarray(img.convert('L').resize((252, 252)), dtype='float32') / 255.0
    if a.mean() > 0.5:
        a = 1 - a
    lo, hi = np.percentile(a, 1), np.percentile(a, 99.5)
    a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    cells = a.reshape(9, 28, 9, 28).transpose(0, 2, 1, 3).reshape(81, 28, 28)
    blank = cells.reshape(81, -1).mean(1) < blank_thr
    cells[blank] = 0
    return cells.astype('float32')


# ------------------------------------------------------------------ sidebar
st.title('Handwritten Sudoku Reader and Solver')
st.caption('CNN reads the digits → AC-3 finds the conflict → search (UCS / Greedy / A*) proposes the cheapest corrections '
           '→ CSP (BT + FC + AC-3) tests each corrected puzzle.')

seeds = available_seeds()
with st.sidebar:
    st.header('Settings')
    seed = st.selectbox('Network / digit-bank seed', seeds)
    mode = st.radio('Input', ['Generated puzzle', 'Upload image'])
    algo = st.radio('Search algorithm', ['ucs', 'astar', 'greedy'], format_func={'ucs': 'UCS', 'astar': 'A*', 'greedy': 'Greedy'}.get,
                    help='UCS and A* return the most likely correction set; Greedy is faster but may be more expensive.')
    goal = st.radio('Goal', ['solvable', 'unique'], format_func={'solvable': 'puzzle solvable', 'unique': 'exactly one solution'}.get)
    max_corr = st.slider('Max corrections', 0, 3, 3)
    live_cnn = st.checkbox('Read with the live CNN (slower)', value=(mode == 'Upload image'), disabled=(mode == 'Upload image'))

bank = get_bank(seed)
mosaic, truth, solution_true = None, None, None

# ------------------------------------------------------------------ input
if mode == 'Generated puzzle':
    c1, c2, c3 = st.columns(3)
    pseed = c1.number_input('Puzzle seed', 0, 10_000, 7)
    givens = c2.slider('Givens', 25, 40, 30)
    k = c3.slider('Injected misreads', 0, 5, 2)
    puz, sol = get_puzzle(int(pseed), givens)
    mosaic = build_mosaic(puz, sol, bank, random.Random(int(pseed) + k), k)
    truth, solution_true = puz, sol
    if live_cnn:
        model = get_model(seed)
        grid, P = read_mosaic(mosaic, predict_fn=lambda X: nn.softmax(nn.predict_logits(model, X), bank.T))
    else:
        grid, P = read_mosaic(mosaic)
    buf = io.BytesIO()
    Image.fromarray((255 - mosaic.image() * 255).astype('uint8')).save(buf, format='PNG')
    st.sidebar.download_button('Download this mosaic as PNG (to test Upload)', buf.getvalue(), 'mosaic.png', 'image/png')
else:
    st.info('Upload a **square-ish picture of a 9×9 grid**, one handwritten digit per cell, no grid lines, dark ink on a light '
            'background (or the reverse). The picture is resized to 252×252 and cut into 81 cells of 28×28. '
            'The network was trained on MNIST digits, so photos with lines, shadows or thick pens will be misread. '
            'You can download a mosaic from the “Generated puzzle” mode to try it.')
    up = st.file_uploader('Image of a handwritten Sudoku', type=['png', 'jpg', 'jpeg'])
    thr = st.slider('Blank-cell brightness threshold', 0.0, 0.2, 0.03, 0.005,
                    help='A cell whose mean brightness is below this value is treated as blank (brightness rule).')
    if up is None:
        st.stop()
    cells = image_to_cells(Image.open(up), thr)
    mosaic = Mosaic(cells, np.zeros((81, 9)), np.zeros((81, 9)), [0] * 81, [0] * 81, [])
    model = get_model(seed)
    grid, P = read_mosaic(mosaic, predict_fn=lambda X: nn.softmax(nn.predict_logits(model, X), bank.T))
    # read_mosaic uses its own fixed threshold; apply the slider to the blank decision instead
    blank = cells.reshape(81, -1).mean(1) < thr
    P[blank] = 0
    grid = [0 if blank[i] else int(P[i].argmax()) + 1 for i in range(81)]

# ------------------------------------------------------------------ pipeline
confs = conflicts_of(grid)
conflict_cells = {c for cs in confs for c in cs}
res = solve_handwritten(grid, P, algo=algo, goal=goal, max_corr=max_corr)
corrected_cells = {c['cell'] for c in res['corrections']}

col1, col2, col3 = st.columns(3)
with col1:
    st.subheader('1. Handwritten input')
    st.pyplot(fig_mosaic(mosaic, mosaic.injected))
    if mosaic.injected:
        st.caption('Red boxes = injected hard digits: ' + ', '.join(cell_name(c) for c in mosaic.injected))
with col2:
    st.subheader('2. What the CNN read')
    st.pyplot(fig_reading(grid, P, conflict_cells))
    st.caption('Colour = confidence (red = unsure). Red digits = cells in a duplicate-digit conflict.')
    if confs:
        st.warning('AC-3 / conflict: ' + '; '.join('{' + ', '.join(cell_name(c) for c in cs) + '}' for cs in confs))
    else:
        st.success('No duplicate-digit conflict in the reading.')
with col3:
    st.subheader('3. Result')
    if res['status'] == 'solved':
        st.pyplot(fig_solution(res['solution'], corrected_cells))
        st.caption('Red = corrected cells.')
    else:
        st.error(res['message'])
        st.write('Least confident cells:')
        st.dataframe(pd.DataFrame(res['least_confident']), hide_index=True)

if res['status'] == 'solved':
    st.success(f"Solved with {len(res['corrections'])} correction(s), total cost {res['cost']:.3f}"
               + (f"  ·  exact solution recovered: {res['solution'] == solution_true}" if solution_true else ''))
    if res['corrections']:
        st.dataframe(pd.DataFrame(res['corrections'])[['row', 'col', 'read', 'new', 'cost']].rename(
            columns={'read': 'read digit', 'new': 'corrected to'}), hide_index=True)
    else:
        st.write('No correction was needed.')
    if solution_true and res['solution'] != solution_true:
        st.warning('The puzzle is solvable but the solution differs from the true one: the misread left another valid puzzle. '
                   'Try the “exactly one solution” goal.')

st.subheader('Search counters')
s = res['stats']
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric('Nodes expanded', s['expanded'])
m2.metric('Nodes generated', s['generated'])
m3.metric('Goal tests', s['goal_tests'])
m4.metric('CSP calls', s['solver_calls'])
m5.metric('Time (s)', f"{s['time']:.3f}")

with st.expander('Search trace (order in which correction sets were tested)'):
    tr = []
    res_tr = solve_handwritten(grid, P, algo=algo, goal=goal, max_corr=max_corr, trace=tr)
    st.dataframe(pd.DataFrame([{'order': t['order'], 'corrections': ', '.join(f'{cell_name(c)}→{d}' for c, d in t['state']) or '(none)',
                                'g (cost)': round(t['g'], 3), 'goal reached': t['goal']} for t in tr[:300]]), hide_index=True)
    if len(tr) > 300:
        st.caption(f'first 300 of {len(tr)} tested states')

if res['status'] == 'solved':
    with st.expander('CSP pruning: BT vs FC vs AC-3 on the corrected puzzle'):
        corrected = res['problem'].grid_of(frozenset((c['cell'], c['new']) for c in res['corrections']))
        rows = []
        for mode_, name in (('bt', 'BT'), ('bt_ac3', 'BT + AC-3'), ('fc', 'BT + FC'), ('fc_ac3', 'BT + FC + AC-3'), ('mac', 'BT + FC + MAC')):
            _, stt, _ = solve(corrected, mode=mode_, limit=1)
            rows.append({'variant': name, 'assignments': stt['assignments'], 'backtracks': stt['backtracks'],
                         'AC-3 prunings': stt['ac3_prunings'], 'FC prunings': stt['fc_prunings']})
        st.dataframe(pd.DataFrame(rows), hide_index=True)

if solution_true is not None:
    with st.expander('NN on vs off'):
        P_off = np.where(P > 0, 1 / 9 + np.random.RandomState(0).uniform(0, 1e-3, P.shape), 0)
        for i, v in enumerate(grid):
            if v:
                P_off[i, v - 1] += 0.01
        off = solve_handwritten(grid, P_off, algo=algo, goal=goal, max_corr=max_corr)
        st.table(pd.DataFrame([
            {'system': 'no correction', 'status': 'solved' if not conflicts_of(grid) and solve(grid, mode='fc_ac3')[0] else 'no solution',
             'exact solution': '-', 'goal tests': 1},
            {'system': 'NN OFF (flat costs)', 'status': off['status'], 'exact solution': str(off['solution'] == solution_true),
             'goal tests': off['stats']['goal_tests']},
            {'system': 'NN ON (probability costs)', 'status': res['status'], 'exact solution': str(res['solution'] == solution_true),
             'goal tests': res['stats']['goal_tests']}]))
