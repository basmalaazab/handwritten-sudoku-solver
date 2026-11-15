# P7 — Handwritten Sudoku Reader and Solver

A CNN reads a handwritten Sudoku (MNIST digits). If it misreads a digit the puzzle breaks. The system finds the
most likely misread digits, corrects them and solves the puzzle:

```
mosaic ──CNN──► digits + probabilities ──AC-3──► conflict? ──search (UCS/Greedy/A*)──► cheapest corrections
                                                                      ▲                         │
                                                      cost = log p(read) − log p(new)           ▼
                                                                           CSP (BT/FC/AC-3) tests each corrected puzzle
                                                                           first solvable one wins, else "unreadable"
```

## Layout (checklist item 1)

| module | content |
|---|---|
| `p7/data.py` | MNIST download/verify/load, seeded split, hard-digit rule, digit bank, mosaic builder + reader (brightness rule) |
| `p7/nn.py` | CNN (std + over-confident variant), logistic-regression baseline, ECE, temperature scaling |
| `p7/search.py` | UCS / Greedy / A* over correction sets, random baseline, counters (expanded / generated / goal tests) |
| `p7/csp.py` | BT, forward checking, AC-3 (none / pre / maintained), MRV as **separate switches**, counters, generator |
| `p7/integration.py` | the whole pipeline + "unreadable" report |
| `p7/eval.py` | Exp 1-4, 5 seeds, mean ± std, tables, plots |
| `tests/` | 31 pytest tests (normal / edge / impossible) |
| `make_data.py`, `run_eval.py`, `run_demo.py` | entry points |
| `docs/` | `DATA_DICTIONARY.md`, `DEMO_SCRIPT.md`, `P7_report.pdf` (`python docs/build_report.py`), slides: https://claude.ai/artifact/FYzXHhtzQyrs4QqtSCppVD |

## Setup (checklist item 8)

Python 3.12, CPU only (about 2 GB RAM).

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # pinned versions
python make_data.py --seed 1             # downloads MNIST (MD5-checked), seeded split, seeded puzzles
python -m pytest tests -q                # 31 tests
python run_demo.py --seed 1 --k 2        # one integrated demo + results/demo.png (trains seed-1 model on first use, ~2.5 min)
python run_eval.py --seeds 1 2 3 4 5     # all experiments, 5 seeds (~40 min on 1 CPU core) -> results/tables.md
```

If `yann.lecun.com` is unreachable `make_data.py` falls back to two mirrors and checks the same MD5 values.
Put the four `.gz` files in `data/` by hand if you are offline.

## Web app (Streamlit)

```bash
streamlit run app.py        # Windows: run_app.bat / run_app.ps1 (one folder above this one)
```
Generated mode: choose puzzle seed, givens, injected misreads, UCS/A*/Greedy, solvable/unique goal; the page shows the mosaic,
the CNN reading and its confidence, the AC-3 conflict, the corrections, the solution, the search counters and trace,
the CSP pruning table and NN on/off. Upload mode: upload a picture of a 9x9 handwritten grid (one digit per cell, no lines).
Needs the model of `results/seed1/std` (included in the zip).

## Switches of the CSP (checklist item 6)

```python
from p7.csp import solve
solve(grid, fc=False, ac3_mode='none')   # plain backtracking
solve(grid, fc=False, ac3_mode='pre')    # BT + AC-3 preprocessing
solve(grid, fc=True,  ac3_mode='none')   # BT + forward checking (MRV on)
solve(grid, fc=True,  ac3_mode='pre')    # BT + FC + AC-3 once
solve(grid, fc=True,  ac3_mode='mac')    # BT + FC + AC-3 after every assignment (MAC)
# returns (solutions, stats, hit_limit); stats = assignments, backtracks, ac3_prunings, fc_prunings
```

## Results

See `results/tables.md` (mean ± std over seeds 1-5), the plots `results/exp1.png`, `exp2.png`, `exp3.png`,
`results/demo.png` and the report in `docs/`.

## Honest limits

* Mosaics are synthetic (MNIST digits glued into a grid, hard digits partly augmented), not photographed Sudoku.
* The correct digit is always among the CNN's top 3 choices for injected digits (part of the injection rule).
* Only the puzzle/mosaic/NN seeds vary across the 5 runs; the generator and the injection rule are fixed.
