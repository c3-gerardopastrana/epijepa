# Theory: a robust linear-sufficiency objective (z as a bounded observer)

This note defines the objective behind the `z2feat` / `minmax_random` experiments and
situates it precisely relative to the epiplexity estimator of Zhang & Levin,
*Intelligence from Learnable Novelty* (arXiv:2607.18433), which in turn builds on Finzi
et al., *From Entropy to Epiplexity* (arXiv:2601.03220).

## 1. Background: epiplexity and its estimator

Epiplexity (Finzi et al. 2026; "learnable novelty" in Zhang & Levin) is an MDL quantity:
for data explained by a **computationally bounded learner**, the total description length
splits into a residual part (what the learner can never compress) and a **program length**
of the best bounded model. Epiplexity is that program-length part — *the structure a
finite-compute learner can actually extract*.

Crucially, the definition **does not fix what the bounded learner is**, only that it
learns (Zhang & Levin, §3: "the definition does not prescribe a particular learner
architecture; it only requires a system with bounded learning capacity").

Zhang & Levin make this cheap and differentiable with a **reservoir** learner: a fixed
random nonlinear map `phi` producing features `H = phi(X)`, followed by a single linear
readout `W` (all learnable capacity is in `W`). With Gaussian residual noise, the model
program length is priced by a **spectral description length** (their Eq. 6):

    C_spec(W) = alpha * log2 det( I + eta W Wᵀ ) = alpha * sum_i log2( 1 + eta s_i(W)^2 )

and, approximating the MDL minimizer by ridge regression (Eqs. 7–9):

    W_lambda = (H̃ᵀ H̃ + lambda I)⁻¹ H̃ᵀ Ỹ
    S^phi(Y | X) = ½ log2 det( I + eta W_lambda W_lambdaᵀ )

with per-column standardized features H̃ and centered/scaled target Ỹ.

**What the "guarantees" are.** There is *no proven bound* tying this estimator to true
epiplexity — the authors (and the Pith review) are explicit that Eq. 9 is "a sequence of
approximations and analogies." What it has are **MDL-grounded properties**: (i) the
log-det is a genuine codelength (derivable from a hierarchical Gaussian prior); (ii)
scaling a readout costs only *logarithmically*, and redundant/duplicate directions add
~nothing, so the score measures *independent learnable structure*, not magnitude; (iii)
it cleanly separates learnable structure from noise (noisy-TV -> all residual -> 0;
dark-room -> no surprise -> 0); (iv) empirically stable where raw-magnitude bonuses
collapse. These are properties, not theorems.

## 2. Two directions, both valid epiplexity instances

Let `H` = (standardized) random-reservoir features, `Z_c` = centered embeddings
`z = E_theta(x)`. The score `S(·,·) = ½ log2 det(I + eta W Wᵀ)` can be formed in two ways,
differing in **which representation is the bounded learner** and **which is the data**:

| | bounded learner | data explained | ridge | "epiplexity of…" | what's optimized |
|---|---|---|---|---|---|
| **feat2z** (Zhang & Levin repr. expt) | reservoir + linear readout | the embedding `z` | `H -> Z_c` | how much of `z` a fixed observer extracts | the **data source** `E_theta` |
| **z2feat** (this work) | `z` + linear readout | the random features `H` | `Z_c -> H` | how much of `H` a `z`-bounded observer extracts | the **observer** `z` |

Because the definition only requires *a* bounded learner, **both are legitimate epiplexity
estimators** — they just measure the epiplexity of different objects. The paper's
representation experiment maximizes `S^phi(Z | X)` with a *fixed* reservoir observer and an
*optimized* data source `z`. This work does the dual: the observer **is** `z` (bounded to a
linear readout), the data is a fixed random feature set `H`, and we optimize the observer.

Both prevent collapse the same way: a constant `z` gives `Z_c = 0`, so the ridge weights
and the score are zero — a bounded learner with nothing to learn.

## 3. The reframing: z as a bounded observer learning "true features"

Reading z2feat directly: **`z` is a computationally bounded observer** (bounded because it
may only apply a *linear* readout) that tries to **explain a set of image features `H`**.
If we treat the random features `H` as a proxy for the image's "true" structure, then a
high score means *`z` has captured that structure in a linearly accessible form* — exactly
the linear-probe criterion, turned into a training objective.

The honest caveat is not about the direction (that is sanctioned by the definition) but
about `H`: a *single* random reservoir is a biased, incomplete proxy for "true features."
That is what motivates the max-min.

## 4. The objective: robust linear sufficiency (max-min over observers' targets)

Any one random feature map `g_R` is a shaky stand-in for "true structure." So we require
`z` to be a good bounded observer of the **worst-case** target over a family of random maps
`R ~ R_dist`:

    max_theta  min_{R ~ R_dist}  S( z = E_theta(x) ;  y_R = g_R(x) )

Equivalently in risk form, with the inner ridge made explicit (three nested problems):

    min_theta  max_R  min_W  E_x || y_R - W z ||²          (readout, adversary, encoder)

- **inner (W):** ridge readout from `z` — closed form.
- **middle (R):** the adversary picks the random target `z` currently explains worst.
- **outer (theta):** the encoder makes `z` a good bounded observer of *every* `R`.

**Why worst-case, not average.** `min_theta E_R[risk]` lets easy targets compensate for
hard ones — `z` can over-serve some directions and ignore a few. The **max-min forbids
compensation**: a single poorly-explained target sets the loss, forcing `z` to be a
*uniformly* good linear-sufficient statistic over the family — robust to which random
"true feature" proxy it faces.

## 5. Practical relaxations (what the code implements)

1. **Sample the family:** draw `K` maps `R_1..R_K`, replace `max_R` by a max over the
   sample. **Resample every step** (`--reservoir minmax_random`) so the worst-case tracks
   the distribution and *moves as `z` improves* — a dynamic adversary, not a frozen set.
   (A frozen `K`-set makes the max ~= the mean; this was confirmed empirically.)
2. **Soft-min (temperature `tau`):** `min_k S_k ≈ -tau log sum_k exp(-S_k / tau)`
   (`--multi_agg max --softmin_tau`), interpolating average-case (`tau→∞`) to hard
   worst-case (`tau→0`).
3. **Diverse observer family:** the family must have *complementary blind spots* or the
   worst-case is no harder than the average. iid random CNNs share one NNGP kernel (bad);
   multi-bandwidth/-family RFF (`multi_rff`) and multi-scale pooling (`multiscale`) give
   genuinely different views (good).
4. **`z`-as-regressor determinant form:** with `z` the regressor, `W` is `d × m`, so the
   score uses the small `d × d` form `½ log2 det(I + eta W Wᵀ)` — stacking many targets
   stays cheap.

## 6. Naming (to stay honest)

Call it **"maximize the epiplexity of random image features under a `z`-bounded linear
observer, robustified by max-min over the observer–target family."** It *is* epiplexity by
Zhang & Levin's definition (a bounded learner — here `z` — extracting structure), but it is
the **dual** of their representation experiment: we optimize the *observer* `z` rather than
the *data source*, and we take a worst-case over a family of targets. It should not be
presented as inheriting any bound the reservoir estimator is proven to satisfy — there are
no such bounds even for the original direction; both rest on the MDL-grounded *properties*
of the log-det codelength (§1).
