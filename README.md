# Orbital-Basis and Band-Projection Pairing Calculations

This repository is a compact code bundle for comparing two pairing-kernel
calculations in the Lieb-lattice model:

1. a full spin-orbital Gamma-basis calculation;
2. a single-band projected scalar calculation.

Start with `README_FOR_TEACHER.md` for the code map, physical meaning of each
script, and recommended run commands.

Install dependencies:

```bash
pip install -r requirements.txt
```

Main entry points:

```bash
python benchmarks/run_full_16_gamma_benchmarks.py --Nk-base 8 --Nw 8
python benchmarks/run_projected_gap_benchmarks.py --Nk 31
python benchmarks/gamma_component_report.py --Nk 21 --mu-selection positive_product
```

Generated numerical outputs are ignored by git.
