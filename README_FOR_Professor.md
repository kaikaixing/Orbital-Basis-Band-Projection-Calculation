# Pairing Kernel Codes: Orbital Basis and Band Projection

This repository contains two related but different calculations for the Lieb-lattice pairing problem.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## 1. Full Spin-Orbital Gamma-Basis Calculation

Main files:

- `benchmarks/full_16_gamma_pairing_solver.py`
- `benchmarks/run_full_16_gamma_benchmarks.py`

This route keeps the pairing field as a vector in the full matrix basis

```text
Gamma_a = tau_mu sigma_nu,  a = 1,...,16.
```

The eigenvector therefore contains both momentum dependence and Gamma-component dependence.  This is the appropriate code path for checking whether including repulsive channels changes the leading pairing qualitatively.

Run:

```bash
python benchmarks/run_full_16_gamma_benchmarks.py --Nk-base 8 --Nw 8
```

The script compares:

- `attractive8 / paper blocks`: old paper-style attractive subset;
- `all16 / SSG blocks`: all 16 Gamma matrices, grouped by symmetry characters.

Important output columns:

- `leading Gamma`: dominant `tau_mu sigma_nu` component;
- `sign (eta,rho,c)`: gap parity, reciprocal-gauge character, and coupling sign;
- `weight`: fraction of the eigenvector in the dominant Gamma component;
- `harmonic`: best simple momentum harmonic for that dominant component.

## 2. Single-Band Projected Calculation

Main files:

- `symmetrized_lieb_gap_solver.py`
- `scan_symmetrized_single_band_phase_diagram.py`
- `benchmarks/run_projected_gap_benchmarks.py`

This route first projects to one band and solves a scalar gap equation

```text
Omega_nu(p) = sum_k K_nu^sym(p,k) W(k) Omega_nu(k).
```

The unknown is only `Omega_nu(k)` for a fixed spin channel `nu`.  Orbital structure enters through the projected form factors

```text
g_mu(k) = u_n^T(-k) tau_mu u_n(k).
```

For a lightweight Gamma-label diagnostic, inspect the rank-one projected
pair-amplitude lift

```text
C_{mu,nu}^{proj}(k) = g_mu(k) Omega_nu(k).
```

This diagnostic should not be identified with the full orbital HS eigenvector
`Delta_{mu,nu}(k)`.  For a full Gamma-basis HS field, the band-projected gap is
instead

```text
Omega_nu^{band}(k) = sum_mu g_mu^*(k) Delta_{mu,nu}(k).
```

Run:

```bash
python benchmarks/run_projected_gap_benchmarks.py --Nk 31
```

This prints both:

- `mu_selection="positive_product"`: paper-like attractive-sector comparison;
- `mu_selection="all"`: full fixed-spin-channel projected kernel.

## 3. Gamma-Component Report for Projected Modes

Main file:

- `benchmarks/gamma_component_report.py`

This script solves the single-band projected equation and then reports
`g_mu(k) Omega_nu(k)` as a diagnostic Gamma-component shape.  It is not a
full Gamma-basis HS solver and should not be used as evidence that the full
orbital HS field satisfies `Delta_{mu,nu}=g_mu Omega_nu`.

Run:

```bash
python benchmarks/gamma_component_report.py --Nk 21 --mu-selection positive_product
python benchmarks/gamma_component_report.py --Nk 21 --mu-selection all
```

## Current Interpretation Boundary

The full Gamma-basis solver and the scalar band-projection solver do not solve the identical eigenproblem.  They should be compared at the level of:

- leading symmetry sector;
- dominant Gamma component or projected `g_mu Omega_nu` diagnostic component;
- dominant momentum harmonic;
- robustness under `positive_product` versus `all` channels.

The scalar labels `s'`, `d'`, `p`, and `p'` are harmonic classifications of the computed eigenfunction.  They should not be read as symmetry input imposed before solving.
