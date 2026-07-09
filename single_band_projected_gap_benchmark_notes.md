# Single-band projected Lieb gap equation: conventions and benchmarks

This note records the conventions currently used in
`symmetrized_lieb_gap_solver.py` and the benchmark points used to compare with
the PDF/notebook results.

## Main conclusion

There are two separate issues that should not be mixed.

1. `kernel_sign` should not be used as a fitting knob.  After fixing the
   reciprocal-gauge susceptibility ordering, the projected formula should use
   `kernel_sign = +1`.  The earlier need for `kernel_sign = -1` came from using
   the opposite `J <-> -J` ordering in the chi combination.

2. `s_mu s_nu = +1` is not an identity of the full projected equation.  In the
   code this is the `mu_selection="positive_product"` option, i.e. an
   attractive-only sector restriction.  It is the correct setting when we want
   to compare the single-band projected solver with the paper's attractive
   Gamma-channel statements.  If `mu_selection="all"` is used, the solver is
   solving a different fixed-spin-channel single-band problem and should not be
   expected to reproduce the paper's Gamma-sector phase labels point by point.

## Current projected formula

The fixed-`nu` single-band equation is solved as

```text
Omega_nu(p) = int_k K_nu(p,k) W(k) Omega_nu(k),
```

with

```text
K_nu(p,k)
 = g^2/32 sum_mu s_mu s_nu g_mu*(p) g_mu(k) C_{mu,nu}(p,k).
```

The corrected reciprocal-gauge/parity-symmetrized chi combination is

```text
C_{mu,nu}(p,k)
 = chi_{p,k}(-J)
 + s_mu chi_{p,k}(+J)
 + eta_nu chi_{p,-k}(-J)
 + eta_nu s_mu chi_{p,-k}(+J).
```

This `chi(-J) + s_mu chi(+J)` ordering reproduces the Route-I sector kernels:

```text
Gamma1-4 sector -> chi1234
Gamma5/6 sector -> chi56
Gamma7/8 sector -> chi78
```

Using the old ordering `chi(+J) + s_mu chi(-J)` flips the Gamma5/Gamma7 sector
relative to Gamma1-4, and then an artificial global `kernel_sign=-1` can make
some benchmark points look correct while failing near the Gamma1-4 p-wave
region.

## Fixed conventions

Momentum grid:

```text
kx, ky in [-pi, pi), endpoint excluded.
```

For the PDF-final half-angle chi formula we do not wrap the susceptibility
momentum back to the BZ:

```text
wrap_susceptibility_momentum = False
```

The projected two-orbital Lieb Hamiltonian uses

```text
d_x(k) = -4 t cos(kx/2) cos(ky/2)
d_z(k) = -(tp_a - tp_b) (cos kx - cos ky)
E_k = sqrt(d_x(k)^2 + d_z(k)^2)
```

With the default values `t=1`, `tp_a=-0.3`, `tp_b=0.2`,

```text
d_z(k) = 0.5 (cos kx - cos ky).
```

The form factors are

```text
g_0(k) = 1
g_x(k) = d_x(k) / E_k
g_y(k) = 0
g_z(k) = d_z(k) / E_k
```

The code enforces the inversion-symmetric gauge

```text
u(-k) = u(k),
```

so `g_mu(-k)=g_mu(k)` holds numerically on the grid.

The Pauli signs are

```text
s_mu = (+1, -1, -1, +1) for mu = 0, x, y, z
s_nu = {0:+1, x:-1, y:-1, z:+1}
eta_nu = {0:-1, x:-1, y:+1, z:-1}
```

Here `eta_nu` is the momentum parity required by fermionic antisymmetry of
`d_{-k}^T sigma_nu d_k`: `nu=y` is even, while `nu=0,x,z` are odd.

## Gamma-label mapping

The paper's eight Gamma labels are used only as a diagnostic language for
classifying single-band modes.  The scalar projected solver still solves for
`Omega_nu(k)` in a fixed spin channel; it does not reconstruct the full
orbital HS field `Delta_{mu,nu}(k)`.

| Gamma | orbital mu | spin nu | use in projected diagnostic |
|---|---|---|---|
| Gamma1 | 0 | 0 | label for the fixed-`nu=0` mode with orbital label `mu=0` |
| Gamma2 | z | z | label for the fixed-`nu=z` mode with orbital label `mu=z` |
| Gamma3 | z | 0 | label for the fixed-`nu=0` mode with orbital label `mu=z` |
| Gamma4 | 0 | z | label for the fixed-`nu=z` mode with orbital label `mu=0` |
| Gamma5 | x | y | label for the fixed-`nu=y` mode with orbital label `mu=x` |
| Gamma6 | y | x | label for the fixed-`nu=x` mode with orbital label `mu=y` |
| Gamma7 | x | x | label for the fixed-`nu=x` mode with orbital label `mu=x` |
| Gamma8 | y | y | label for the fixed-`nu=y` mode with orbital label `mu=y` |

Because `g_y=0` in this projected Lieb model, Gamma6 and Gamma8 vanish.

These labels are useful for comparing the projected scalar output with the
paper notation, but they should not be identified with full orbital
Hubbard-Stratonovich eigenvectors.  Full Gamma-basis HS fields must be computed
with the full Gamma-space kernel.

## Attractive-only comparison

For comparison with the paper's leading attractive channels, use

```python
GapEquationParams(
    susceptibility_type="projected_formula",
    kernel_sign=1.0,
    mu_selection="positive_product",
    wrap_susceptibility_momentum=False,
    paper_chi_prefactor=1.0,
)
```

`mu_selection="positive_product"` keeps only terms with

```text
s_mu s_nu > 0.
```

This gives the paper-like attractive sectors:

```text
nu = y: keep mu = x, y -> effectively Gamma5 because g_y=0
nu = x: keep mu = x, y -> effectively Gamma7 because g_y=0
nu = 0,z: keep mu = 0,z -> Gamma1-4 sector
```

Without this restriction, the fixed-`nu` single-band kernel includes additional
orbital components in the same spin channel.  That is a valid projected
equation, but it is not the same benchmark as the paper's Gamma-channel
attractive-sector comparison.

## Benchmarks

All benchmark numbers below were computed with

```text
Nk = 31
T = 0.05
J = 2/3
r = (r/J) * J
band_sign = +1
mu_selection = "positive_product"
classify_object = "projected_component"
overlap_weight_mode = "pair_factor"
```

The eigenvalues are discretization-dependent; the channel and form factor are
the main benchmark.

| chemical potential | r/J | leading sector | expected pairing form | numerical leading result |
|---:|---:|---|---|---|
| 0.5 | 1.5 | Gamma5 = tau_x sigma_y | `d' ~ sin(kx/2) sin(ky/2)` | `nu=y`, `lambda=3.07898e-02`, overlap `-0.880` |
| 1.5 | 1.5 | Gamma7 = tau_x sigma_x | twofold `p'`, `u_+`/`u_-` | `nu=x`, `lambda=2.67935e-02`, representative `u_-`, overlap `+0.836` |
| 2.5 | 1.5 | Gamma5 = tau_x sigma_y | `s' ~ cos(kx/2) cos(ky/2)` | `nu=y`, `lambda=2.94729e-02`, overlap `+0.998` |
| 0.75 | 1.0753 | Gamma1-4 sector | fourfold p-wave, `f1/f2` | `nu=0,z`, `lambda=1.50448e-01`, overlap with `f1/f2` about `0.94` |

For the near-critical p-wave point,

```text
f1(k) = sin kx - sin kx cos ky = sin kx (1 - cos ky)
f2(k) = sin ky - sin ky cos kx = sin ky (1 - cos kx)
```

The fourfold degeneracy appears as:

1. two spin-channel degeneracies, `nu=0` and `nu=z`;
2. two momentum-form-factor degeneracies, `f1` and `f2`.

Because the eigensolver can return any orthogonal basis inside a degenerate
subspace, a raw eigenvector may look like a rotated combination of `f1` and
`f2`.  The correct comparison is the whole degenerate subspace.  Projecting the
`nu=0,z` top-two subspace onto `f1/f2` gives overlap about `0.9375` for
`Omega` and about `0.7528` for the `g_z Omega` components on the `Nk=41`
check.

## What goes wrong without the benchmark conventions

With the corrected chi ordering but `mu_selection="all"`, some points still
look reasonable, but the comparison is no longer the paper's attractive-sector
benchmark.  For example, at `mu=2.5`, `r/J=1.5`, the full fixed-spin-channel
sum can choose a `Gamma7`-like `p'` channel instead of the paper's `Gamma5`
`s'` channel.  This does not by itself mean the numerics are wrong; it means
the object being diagonalized is different.

The explicit `mu_selection="all"` check, using the same corrected
reciprocal-gauge-symmetrized kernel and the same `projected_component`
diagnostic classifier,
is:

```text
Nk = 31
T = 0.05
J = 2/3
kernel_sign = +1
mu_selection = "all"
classify_object = "projected_component"
overlap_weight_mode = "pair_factor"
```

| chemical potential | r/J | leading component | pairing label | spin channel | lambda | overlap | runner-up lambda |
|---:|---:|---|---|---|---:|---:|---:|
| 0.5 | 1.5 | Gamma5 = tau_x sigma_y | `d' ~ sin(kx/2) sin(ky/2)` | `nu=y` | `2.69972e-02` | `+0.880` | `1.70483e-02` |
| 1.5 | 1.5 | Gamma7 = tau_x sigma_x | `u_- ~ sin((kx-ky)/2)` | `nu=x` | `1.69159e-02` | `-0.668` | `1.39259e-02` |
| 2.5 | 1.5 | Gamma7 = tau_x sigma_x | `u_- ~ sin((kx-ky)/2)` | `nu=x` | `1.11473e-02` | `+0.687` | `6.10489e-03` |
| 0.75 | 1.0753 | Gamma1 = tau_0 sigma_0 | `f1 ~ sin kx (1 - cos ky)` | `nu=0` | `1.37393e-01` | `-0.862` | `1.37393e-01` |

Thus the full fixed-`nu` projected kernel is not wildly unrelated to the
attractive-only benchmark, but it is not identical:

1. `mu=0.5, r/J=1.5` remains Gamma5 `d'`.
2. `mu=1.5, r/J=1.5` remains Gamma7 `p'`, though the eigenvalue is smaller.
3. `mu=2.5, r/J=1.5` changes from the paper-like Gamma5 `s'` result to a
   Gamma7 `p'` result.
4. `mu=0.75, r/J=1.0753` remains in the Gamma1-4 p-wave sector, with the
   expected twofold spin-channel degeneracy and twofold `f1/f2` momentum
   degeneracy.

The clean benchmark prescription is therefore:

```text
fixed chi ordering: chi(-J) + s_mu chi(+J)
fixed kernel_sign: +1
paper comparison: mu_selection="positive_product"
diagnostic Gamma label: inspect C^{proj}_{mu,nu}=g_mu Omega_nu, not only Omega_nu
```

## Check: dropping reciprocal-gauge symmetrization

We also tested the parity-only kernel

```text
K_nu^(one-sided)(p,k)
 = g^2/16 sum_mu s_mu s_nu g_mu*(p) g_mu(k)
   [chi_{p,k}(J) + eta_nu chi_{p,-k}(J)].
```

This is the formula obtained after using only inversion
`g_mu(-k)=g_mu(k)`, but before the reciprocal-lattice gauge average over
`k` and `k+G_l`.  It is therefore not the same object as the PDF-final
reciprocal-gauge-covariant kernel.

Keeping the same benchmark settings,

```text
Nk = 31
T = 0.05
J = 2/3
mu_selection = "positive_product"
classify_object = "projected_component"
```

the one-sided result is:

| chemical potential | r/J | reciprocal-gauge-sym result | one-sided result with `chi(+J)` |
|---:|---:|---|---|
| 0.5 | 1.5 | Gamma5, `d'` | Gamma5, `s'`, `lambda=8.05466e-02` |
| 1.5 | 1.5 | Gamma7, `p'` | Gamma5, `s'`, `lambda=9.83389e-02` |
| 2.5 | 1.5 | Gamma5, `s'` | Gamma5, `s'`, `lambda=7.94761e-02` |
| 0.75 | 1.0753 | Gamma1-4, `f1/f2` p-wave | Gamma5, `s'`, `lambda=1.46003e-01` |

The convention ambiguity in the one-sided `chi(J)` label was also checked by
using `chi(-J)` instead of `chi(+J)`.  It still drives the same four benchmark
points to a leading Gamma5 `s'`-like state, so the mismatch is not fixed by
flipping the one-sided `J` sign.

This means reciprocal-gauge symmetrization is not a harmless cosmetic
improvement in this reduced single-band implementation.  It is required to
recover the paper-sector kernel structure.  The reason is that the projected
form factors are inversion-even, but they are not all periodic under
`k -> k+G_l`:

```text
g_0(k+G_l) = +g_0(k)
g_z(k+G_l) = +g_z(k)
g_x(k+G_l) = -g_x(k)
g_y(k+G_l) = 0
```

At the same time the half-angle susceptibility changes branch under the same
reciprocal shift, effectively exchanging `J` and `-J`.  The reciprocal-gauge
average is what combines these two sign changes into the sector kernels
`chi1234`, `chi56`, and `chi78`.
