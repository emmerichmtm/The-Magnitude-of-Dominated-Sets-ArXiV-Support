# Magnitude gradient-method reproducibility package

This flat package reproduces the numerical examples in the report by **running** the projected gradient methods described in the paper.

There are **no subdirectories** in the zip: every file is at one directory level.

## Files

- `reproduce_magnitude_gradient_examples.py` — main runnable Python script
- `README.md` — usage notes
- `magnitude_report_current.tex` — current LaTeX source used as reference for the numerical examples

## Requirements

The script uses:
- Python 3.10+
- `numpy`
- `matplotlib`

## Default usage

Run:

```bash
python3 reproduce_magnitude_gradient_examples.py
```

By default, this generates **all** outputs in the current directory:
- exact introductory three-point example,
- Problem 1 branch-based 2D runs,
- Problem 2 pull-back 2D runs,
- 3D simplex runs for 6, 9, and 10 points,
- convergence CSV files,
- `example_results.json`,
- TikZ coordinate files,
- matplotlib PNG plots.

## Command-line interface

Generate only the 2D examples:

```bash
python3 reproduce_magnitude_gradient_examples.py --only-2d
```

Generate only the 3D examples:

```bash
python3 reproduce_magnitude_gradient_examples.py --only-3d
```

Disable PNG plots:

```bash
python3 reproduce_magnitude_gradient_examples.py --no-png
```

Disable TikZ coordinate files:

```bash
python3 reproduce_magnitude_gradient_examples.py --no-tikz
```

Write outputs to a separate directory:

```bash
python3 reproduce_magnitude_gradient_examples.py --outdir results
```

Options can be combined, for example:

```bash
python3 reproduce_magnitude_gradient_examples.py --only-3d --no-tikz --outdir simplex_results
```

## Implemented methods

- exact inclusion--exclusion oracle for anchored-box hypervolume,
- exact inclusion--exclusion oracle for anchored-box magnitude,
- exact tie-shared subgradients for both indicators,
- normalized projected gradient ascent,
- pull-back through the Jacobian for the 2D decision-space examples,
- Euclidean projection onto the simplex for the 3D runs,
- Das--Dennis initialization for the 3D simplex runs.

## Numerical scope

The script reproduces the numerical plots and terminal populations for:
- the introductory 3-point dominated-set example,
- Problem 1 (2D branch parameterization),
- Problem 2 (2D decision-space pull-back method),
- 6-point analytic simplex configurations,
- 9-point and 10-point 3D simplex runs from Das--Dennis starts.
