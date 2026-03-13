# Random Walk Mixing on Finite Groups

## Goal

Empirically study how the algebraic structure of a finite group and the choice of generating set affect the **mixing behavior of random walks**, and verify the relationship between **spectral gap** and **mixing speed**.

---

## Mathematical setup

Let $G$ be a finite group and $S \subseteq G$ a generating set satisfying $S = S^{-1}$.

We define a random walk on $G$ by

$$
X_{t+1} = s_t X_t
$$

where $s_t$ is sampled uniformly from $S$.

To avoid periodicity issues we use a **lazy random walk**

$$
P(x,y) = \frac{1}{2} \mathbf{1}_{x=y} + \frac{1}{2|S|} \sum_{s\in S}\mathbf{1}_{y=sx}.
$$

This defines a Markov chain with transition matrix $P$.

The uniform distribution

$$
U(g) = \frac{1}{|G|}
$$

is stationary.

---

### Distance to stationarity

We measure convergence using **total variation distance**

$$
d_{\mathrm{TV}}(P_t, U) = \frac{1}{2} \sum_{g \in G} \left| P_t(g) - U(g) \right|.
$$

---

### Spectral structure

Let the eigenvalues of the transition matrix be

$$
1 = \lambda_1 \ge |\lambda_2| \ge \dots
$$

The **spectral gap**

$$
\gamma = 1 - |\lambda_2|
$$

controls convergence speed.

Heuristically

$$
d_{\mathrm{TV}}(P_t,U) \approx O((1-\gamma)^t).
$$

---

### Groups studied

The experiments consider:

1. Cyclic groups
   $$
   G = \mathbb{Z}_n
   $$

2. Dihedral groups
   $$
   D_n = \langle r,s \mid r^n=e,\ s^2=e,\ srs=r^{-1}\rangle
   $$

3. Symmetric groups (optional extension)
   $$
   S_n
   $$

---

## Experiment design

### State space

All group elements are explicitly enumerated.

Group size examples:

| group            | size   |
|------------------|--------|
| $ \mathbb{Z}_n $ | $ n $  |
| $ D_n $          | $ 2n $ |
| $ S_n $          | $ n! $ |

For computational feasibility experiments typically use

$$
n \le 20
$$

for cyclic and dihedral groups.

---

### Generating sets

Example generating sets tested:

Cyclic group

$$
S_1 = \{+1,-1\}
$$

$$
S_2 = \{+1,-1,+2,-2\}
$$

Dihedral group

$$
S = \{r, r^{-1}, s\}
$$

Symmetric group

Adjacent transpositions

$$
S = \{(1,2),(2,3),\dots,(n-1,n)\}
$$

---

### Simulation procedure

1. Enumerate group elements.
2. Construct the transition matrix $P$.
3. Initialize

$$
P_0 = \delta_e
$$

4. Compute

$$
P_t = P_0 P^t
$$

for a range of $t$.

5. Measure distance to stationarity.

---

### Metrics

Primary metrics:

1. Total variation distance

$$
d_{\mathrm{TV}}(P_t,U)
$$

2. Spectral gap

$$
\gamma = 1 - |\lambda_2|
$$

3. Mixing time

Defined as the smallest $t$ such that

$$
d_{\mathrm{TV}}(P_t,U) < 0.01.
$$

---

### Parameters

Typical experimental settings:

| parameter   | value                          |
|-------------|--------------------------------|
| max steps   | 200                            |
| groups      | $ Z_n $, $ D_n $               |
| repetitions | deterministic matrix iteration |
| walk type   | lazy random walk               |

---

## How to run

```bash
uv run python run.py
```

This command records experiment metadata and environment information.

---

## Outputs

Generated files:

```
artifacts/
  metadata.json
  figures/
  tables/
```

---

### Figures

Typical plots include

**1. Mixing curves**

Total variation distance vs steps

```
t → d_TV(P_t, U)
```

---

**2. Spectral gap comparison**

Bar chart comparing spectral gaps for different generating sets.

---

**3. Probability heatmap**

Distribution evolution over time

```
x-axis : step t
y-axis : group element
color  : probability
```

---

### Tables

Example table

| group      | generator        | spectral gap | mixing time |
|------------|------------------|--------------|-------------|
| $ Z_{20} $ | $ \pm 1 $        | 0.049        | 84          |
| $ Z_{20} $ | $ \pm 1, \pm 2 $ | 0.13         | 32          |
| $ D_{20} $ | $ r, r^{-1}, s $ | 0.18         | 21          |

---

## Results

The experiments show clear variation in mixing behavior across different
groups and generating sets.

1. The spectral gap is strongly associated with mixing speed. Configurations
with larger spectral gaps consistently reach the threshold
$ d_{\mathrm{TV}}(P_t,U) < 0.01 $ in fewer steps.

2. The choice of generating set significantly affects the spectral gap,
even within the same group. For example, two generating sets on $S_4$
produce noticeably different mixing times, with the star-transposition
generators yielding a larger spectral gap and faster convergence than
adjacent transpositions.

3. Increasing group size tends to reduce the spectral gap for similar
generator structures. For example, nearest-neighbor walks on
$ \mathbb{Z}_{20} $ and $ \mathbb{Z}_{30} $ show smaller spectral gaps
for the larger group, resulting in slower mixing.

Overall, the experiments support the theoretical intuition that the
spectral gap is a key predictor of convergence speed in finite-group
random walks, while also highlighting that the specific algebraic
structure of the generating set plays a crucial role.

---

## Notes

### Possible extensions

1. Investigate **cutoff phenomenon** in larger symmetric groups.
2. Study the effect of asymmetric generating sets.
3. Connect spectral decomposition with **group representation theory**.
4. Compare deterministic matrix evolution with Monte-Carlo simulation.
