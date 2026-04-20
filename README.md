# Magnitude of Dominated Sets

This repository contains code and reproducibility material for the paper on **magnitude as a quality indicator for Pareto front approximations**. The focus is on exact small- and medium-scale computations for **multiobjective maximization** with a common **anchor point**, together with projected gradient methods that reproduce the numerical examples from the paper.

The repository has two complementary purposes:

1. **Exact indicator computations.** It provides exact routines for 2D and 3D hypervolume and magnitude, together with tie-shared inclusion--exclusion subgradients.
2. **Reproducibility of the numerical examples.** It reproduces the 2D and 3D gradient-based experiments reported in the paper by actually running the projected gradient methods described there.

## Main files

- `exact_magnitude_hypervolume.py` — exact 2D/3D hypervolume and magnitude routines, gradients, and helper functions.
- `reproduce_magnitude_gradient_examples.py` — main script to rerun the numerical examples and regenerate data and plots.
- `magnitude_report_current.tex` — LaTeX source of the current report used as the reference for the numerical examples.
- `README.md` — repository overview and usage instructions.

## Mathematical setting

Let `points = [p^(1), ..., p^(n)]` be objective vectors in `R^d`, and let `anchor = a` be a common anchor point. We consider **maximization**, so each point must weakly dominate the anchor componentwise.

After translation by the anchor,

\[
q^{(i)} = p^{(i)} - a \ge 0,
\]

the dominated region is

\[
D(P) = \bigcup_{i=1}^n [0, q^{(i)}].
\]

### Hypervolume

The exact hypervolume is computed by inclusion--exclusion:

\[
HV_d(P;a) = \sum_{\emptyset \neq S \subseteq \{1,\dots,n\}}
(-1)^{|S|+1}\prod_{k=1}^d \min_{i\in S} q^{(i)}_k.
\]

### Magnitude in 2D

\[
Mag_2(P;a) = 1 + \frac{L_x + L_y}{2} + \frac{HV_2(P;a)}{4},
\]

where `L_x` and `L_y` are the axis projection lengths.

### Magnitude in 3D

\[
Mag_3(P;a)
= 1 + \frac{L_x+L_y+L_z}{2}
  + \frac{A_{xy}+A_{xz}+A_{yz}}{4}
  + \frac{HV_3(P;a)}{8},
\]

where `A_xy`, `A_xz`, and `A_yz` are the exact 2D dominated areas of the coordinate projections.

## Implemented methods

The code includes:

- exact inclusion--exclusion oracles for anchored-box hypervolume,
- exact inclusion--exclusion oracles for anchored-box magnitude,
- exact tie-shared subgradients for both indicators,
- normalized projected gradient ascent,
- pull-back through the Jacobian for the 2D decision-space examples,
- Euclidean projection onto the simplex for the 3D runs,
- Das--Dennis initialization for the 3D simplex runs.

Because the indicators are not differentiable at ties, the returned gradients are consistent **subgradient selections**. Tied contributions are shared equally among the tied points.

## Exact API

### Hypervolume
- `hypervolume_2d_max(points, anchor=(0.0, 0.0))`
- `hypervolume_3d_max(points, anchor=(0.0, 0.0, 0.0))`
- `hypervolume_gradient_2d_max(points, anchor=(0.0, 0.0))`
- `hypervolume_gradient_3d_max(points, anchor=(0.0, 0.0, 0.0))`

### Magnitude
- `magnitude_2d_max(points, anchor=(0.0, 0.0))`
- `magnitude_3d_max(points, anchor=(0.0, 0.0, 0.0))`
- `magnitude_gradient_2d_max(points, anchor=(0.0, 0.0))`
- `magnitude_gradient_3d_max(points, anchor=(0.0, 0.0, 0.0))`

### Helpers
- `normalize_rows(grad)`
- `simplex_project_rows(points, total=1.0)`
- `projected_gradient_step(points, gradient_fn, anchor, step_size, projector=None, normalize_pointwise=True)`

## Requirements

- Python 3.10+
- `numpy`
- `matplotlib`

## Run the numerical examples

Default usage:

```bash
python3 reproduce_magnitude_gradient_examples.py
```

By default, this generates **all** outputs in the chosen output directory:

- the exact introductory three-point dominated-set example,
- Problem 1 branch-based 2D runs,
- Problem 2 pull-back 2D runs,
- 3D simplex runs for 6, 9, and 10 points,
- convergence CSV files,
- `example_results.json`,
- TikZ coordinate files,
- matplotlib PNG plots.

### Command-line interface

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

## Example: exact 2D indicator values

```python
import numpy as np
from exact_magnitude_hypervolume import (
    hypervolume_2d_max, magnitude_2d_max,
    hypervolume_gradient_2d_max, magnitude_gradient_2d_max,
)

P = np.array([(1.0, 3.0), (3.0, 2.0), (5.0, 1.0)])
a = (0.0, 0.0)

print(hypervolume_2d_max(P, a))
print(magnitude_2d_max(P, a))
print(hypervolume_gradient_2d_max(P, a))
print(magnitude_gradient_2d_max(P, a))
```

## Numerical scope

The scripts reproduce the numerical plots and terminal populations for:

- the introductory 3-point dominated-set example,
- Problem 1 (2D branch parameterization),
- Problem 2 (2D decision-space pull-back method),
- 6-point analytic simplex configurations,
- 9-point and 10-point 3D simplex runs from Das--Dennis starts.

## Complexity note

The exact indicator routines use **inclusion--exclusion** and are therefore exponential in the number of points. This is appropriate for exact theoretical investigations and small- to medium-scale numerical experiments, but not for large populations.

## Reproducibility note

The numerical examples are generated by actually running the gradient methods from the paper, not by replaying pre-stored point sets. The repository is intended as a compact reproducibility package for the results reported in the manuscript.

## Author / project

This code accompanies the paper on *The Magnitude of Dominated Sets: A Quality Indicator for Pareto Front Approximations* and is part of the ongoing work collected on Michael Emmerich's GitHub profile:

- <https://github.com/emmerichmtm>
