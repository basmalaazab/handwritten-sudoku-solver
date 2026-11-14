# 10-minute talk + 5-minute live demo (plan)

## Talk (10 min, 7 slides)
1. Problem and why it is hard (1.0 min): handwritten Sudoku, one misread digit breaks the puzzle.
2. Architecture (1.5 min): NN -> AC-3 -> search -> CSP; the NN probability becomes the cost of changing a digit.
3. Neural network (1.5 min): CNN vs logistic regression; calibration (ECE, temperature scaling) and why it matters for costs.
4. Search (1.5 min): state/action/cost, UCS vs Greedy vs A*, admissible heuristic (disjoint conflicts), counters.
5. CSP (1.5 min): BT, FC, AC-3, MAC with separate switches; assignments/backtracks.
6. Experiments (2.0 min): Exp1-3 tables with mean +- std over 5 seeds; extensions.
7. Limits and conclusions (1.0 min).

## Live demo (5 min) — all commands from the project root
1. `python run_demo.py --k 2` (1.5 min). Point at: the NN reading, the AC-3 conflict list, the UCS trace
   (states tested in cost order, last one = GOAL), UCS/Greedy/A* counters.
2. Same output, section 5 (1 min): CSP pruning — BT vs BT+AC3 vs BT+FC vs BT+FC+AC3 vs MAC on the corrected puzzle
   (assignments/backtracks). Open `results/demo.png`: domain sizes before/after AC-3 + bar chart.
3. Same output, section 6 (1 min): NN ON vs OFF: no correction = no solution; flat costs (NN off) solve it only by
   luck/with many more goal tests and may give a wrong solution; NN costs find the right one fast.
4. `python run_demo.py --k 5 --puzzle-seed 7` (0.5 min): the "unreadable" case with the clear no-solution message
   and the five least-confident cells.
5. `python -m pytest tests -q` (0.5 min): 31 tests green, incl. 2+ impossible cases with the message.

Backup if something fails: `results/demo.png`, `results/tables.md`.
