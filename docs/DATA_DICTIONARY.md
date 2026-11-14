# Data dictionary (P7 Handwritten Sudoku Reader and Solver)

Legend: **REAL** = comes from MNIST unchanged. **GENERATED** = produced by our code (seeded, reproducible).

## 1. MNIST files (REAL) — `data/*.gz`, MD5-verified by `make_data.py`

| file | content | MD5 |
|---|---|---|
| train-images-idx3-ubyte.gz | 60,000 images 28x28, uint8 | f68b3c2dcbeaaa9fbdd348bbdeb94873 |
| train-labels-idx1-ubyte.gz | 60,000 labels 0-9 | d53e105ee54ea40749a09fcbcd1e9432 |
| t10k-images-idx3-ubyte.gz | 10,000 test images | 9fb629c4189551a2d022fa330f9573f3 |
| t10k-labels-idx1-ubyte.gz | 10,000 test labels | ec29112dd5afa0611ce80d1b7f02629c |

Training and test writers are different people (MNIST property). Public data.

## 2. Preprocessed arrays — `data/processed/mnist_1to9_seed{S}.npz` (REAL pixels, GENERATED split)

| array | shape | type | meaning |
|---|---|---|---|
| X_train / X_val / X_test | (N,28,28) | float32 in [0,1] | pixels / 255 |
| y_train / y_val / y_test | (N,) | int64 0..8 | digit - 1 (digit 0 removed) |

Split: seeded permutation of the 60,000 training images -> 50,000 train / 10,000 validation, then digit 0 is dropped
(45,081 / 8,996 for seed 1). The test set (9,020 images of digits 1-9) is untouched.

## 3. Sudoku puzzles — `data/processed/puzzles_seed{S}_g{G}.json` (GENERATED)

| field | type | meaning |
|---|---|---|
| puzzle | list[81] int | row-major 9x9, 0 = blank cell, 1-9 = given |
| solution | list[81] int | the unique solution |

How: full grid by randomised backtracking (forward checking + random value order); then cells are removed in random
order and a removal is kept only if the puzzle still has exactly one solution (checked by counting solutions up to 2),
until G givens remain (G = 30 in Exp 1/3/4; 25/30/35 in Exp 2). `random.Random(seed)`.

## 4. Digit bank — `results/seed{S}/{variant}/bank.npz` (REAL images, GENERATED hard images)

| array | shape | REAL/GENERATED | meaning |
|---|---|---|---|
| Xte, yte | (9020,28,28), (9020,) | REAL | MNIST test images of digits 1-9, label 0..8 |
| P_cal / P_raw | (9020,9) | GENERATED (NN output) | temperature-scaled / raw softmax probabilities |
| H_img | (H,28,28) | GENERATED | hard digit images (see rule) |
| H_lab | (H,) | GENERATED | true digit-1 of each hard image |
| H_cal / H_raw | (H,9) | GENERATED (NN output) | calibrated / raw probabilities of the hard images |
| H_src | (H,) | GENERATED | 0 = natural test image that satisfies the rule, 1 = augmented copy |
| T | scalar | GENERATED | temperature fitted on the validation set |

**Hard-digit rule** (documented in `p7/data.py`): for each digit d, candidates are natural test images of d and
augmented copies (rotation 15-45 deg either way, shear +-0.3, scale 0.85-1.1, 8x8 erase w.p. 0.6, gaussian noise
sigma 0.1 w.p. 0.5). A candidate is accepted iff the calibrated CNN has argmax != d, d is the 2nd or 3rd choice,
max probability <= 0.80 and p(d) >= 0.05. 60 accepted images per digit (about 15% natural, 85% augmented).
"Easy" background digits = test images the calibrated CNN reads correctly (so misread level = injected digits exactly).

## 5. Mosaic (GENERATED, in memory; `p7.data.Mosaic`)

| field | shape | meaning |
|---|---|---|
| images | (81,28,28) | cell images, row-major; blank cells = |N(0,0.002)| noise (mean << BLANK_THR) |
| probs_cal / probs_raw | (81,9) | probabilities cached from the NN for each cell image (zeros for blank cells) |
| puzzle / solution | 81 ints | ground truth |
| injected | list of cell indices | cells whose image is an injected hard digit |
| image() | (252,252) | the full handwritten Sudoku picture |

Blank cells are found with the **brightness rule**: mean pixel < 0.015 (every MNIST digit image 1-9 is brighter;
checked in tests). Not by the NN. This is **not** photographed real-world Sudoku.

## 6. Output files

| file | columns |
|---|---|
| results/seed{S}/std/correction_log.csv | seed, level (injected misreads), puzzle, row, col (1-9), read_digit, new_digit, cost = log p(read) - log p(new), injected_here (was this a truly injected cell), exact (final solution equals the true one) |
| results/seed{S}/std/puzzle_summary.csv | seed, level, algo, puzzle, status (solved/unreadable), exact, n_corrections, cost, ac3_conflict_on_read_grid, injected cells, least_confident (JSON list for unreadable puzzles) |
| results/seed{S}/*/classifier_report.json | accuracy, ECE, NLL for CNN raw / temperature-scaled / logistic regression, temperature, confusion matrix |
| results/seed{S}/std/experiments.json | per-seed results of Exp 1-4 |
| results/summary_mean_std.json, tables.md | mean and standard deviation over seeds |
