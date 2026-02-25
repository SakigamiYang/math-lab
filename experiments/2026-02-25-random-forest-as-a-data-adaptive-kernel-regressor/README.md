# Random Forest as a Data-Adaptive Kernel Regressor

## Goal
To verify numerically that a Random Forest regressor (with `bootstrap=False`) is exactly equivalent to a data-adaptive kernel smoother, i.e., its prediction can be written as a weighted linear combination of training targets.

## Mathematical setup
Let the training dataset be

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n.
$$

Consider a Random Forest with $ B $ trees, trained with:

- `bootstrap = False`
- CART regression trees
- leaf prediction equal to the empirical mean of targets in the leaf

For tree $ b $, let:

- $ L_b(x) $ denote the leaf index of input $ x $
- $ S_b(x) = \{ i : L_b(x_i) = L_b(x) \} $

Then the tree-level prediction is

$$
\hat f_b(x) = \frac{1}{|S_b(x)|} \sum_{i \in S_b(x)} y_i.
$$

The forest prediction is

$$
\hat f_{\mathrm{RF}}(x) = \frac{1}{B} \sum_{b=1}^B \hat f_b(x).
$$

Rewriting:

$$
\hat f_{\mathrm{RF}}(x) = \sum_{i=1}^n w_i(x) y_i,
$$

where

$$
w_i(x) =  \frac{1}{B} \sum_{b=1}^B \frac{\mathbf{1}\{L_b(x_i) = L_b(x)\}}{|S_b(x)|}.
$$

Thus:

- $ w_i(x) \ge 0 $
- $ \sum_i w_i(x) = 1 $

This shows that Random Forest regression is a **kernel smoother** with a data-adaptive kernel induced by tree partitions.

Define the induced proximity kernel:

$$
K(x, x') = \frac{1}{B} \sum_{b=1}^B \mathbf{1}\{L_b(x) = L_b(x')\}.
$$

The regression weights are a leaf-normalized version of this kernel.

## Experiment design
### Data

- Synthetic regression dataset (`sklearn.make_regression`)
- Moderate dimension
- Fixed random seed for reproducibility

### Model

- `RandomForestRegressor`
- `bootstrap = False`
- `n_estimators = 200`
- `min_samples_leaf > 1`
- fixed `random_state`

### Procedure

1. Train the forest.
2. Extract leaf indices for training and test points.
3. Construct weights $ w_i(x) $ explicitly from leaf structure.
4. Reconstruct predictions:

   $$
   \hat f_{\mathrm{recon}}(x) = \sum_i w_i(x) y_i.
   $$

5. Compare reconstructed predictions with `rf.predict`.

### Metrics

- `max_abs_err`
- `mean_abs_err`
- row-sum deviation $ \max |\sum_i w_i(x) - 1| $

Expected outcome:

- Prediction error ≈ machine precision
- Row sums ≈ 1 up to floating-point error

## How to run
```bash
uv run python run.py
```

## Outputs

- `artifacts/metadata.json`

## Results

Numerical reconstruction accuracy:

- `max_abs_err = 1.990e-13`
- `mean_abs_err = 2.614e-14`
- `max_row_sum_err = 2.220e-16`

### Interpretation

1. **Exact prediction equivalence**

   The reconstructed predictions match `rf.predict` up to floating-point precision.  
   The maximum deviation (~1e−13) is at the level of numerical rounding error, confirming:

   $$
   \hat f_{\mathrm{RF}}(x) = \sum_{i=1}^n w_i(x)\, y_i
   $$

   holds exactly under `bootstrap = False`.

2. **Probability structure of weights**

   The row-sum deviation (~1e−16) verifies:

   $$
   \sum_{i=1}^n w_i(x) = 1
   $$

   up to machine precision.  
   Hence each prediction is a convex combination of training targets.

3. **Structural consequence**

   - The Random Forest regressor is **linear in the training targets**.
   - All nonlinearity lies in the data-dependent weights $ w_i(x) $.
   - The model is therefore a **data-adaptive kernel smoother**.
   - The induced geometry depends solely on tree partitions.

### Conclusion

Under deterministic training data and `bootstrap=False`, Random Forest regression admits an exact kernel-weight representation. The numerical experiment confirms the mathematical derivation to floating-point precision.

## Notes

### Open questions

- Does the induced kernel converge as $ B \to \infty $ ?
- Is the proximity kernel positive semi-definite?
- Does the induced metric satisfy the triangle inequality?
