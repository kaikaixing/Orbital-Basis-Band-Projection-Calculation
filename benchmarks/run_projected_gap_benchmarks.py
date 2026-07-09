from __future__ import annotations

import argparse
import json
import sys
from argparse import Namespace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scan_symmetrized_single_band_phase_diagram import solve_scan_point


POINTS = [(0.5, 1.5), (1.5, 1.5), (2.5, 1.5), (0.75, 1.0753)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run projected Lieb gap benchmark points."
    )
    parser.add_argument("--Nk", type=int, default=31)
    parser.add_argument("--T", type=float, default=0.05)
    parser.add_argument("--J", type=float, default=2.0 / 3.0)
    parser.add_argument(
        "--out",
        default="benchmarks/output/projected_gap_benchmarks.json",
        help="JSON output path, relative to repo root unless absolute.",
    )
    return parser.parse_args()


def solver_args(mu_selection: str, args: argparse.Namespace) -> Namespace:
    return Namespace(
        Nk=args.Nk,
        T=args.T,
        g=1.0,
        J=args.J,
        susceptibility_type="projected_formula",
        wrap_susceptibility_momentum=False,
        paper_chi_prefactor=1.0,
        mu_selection=mu_selection,
        overlap_weight_mode="pair_factor",
        fs_cutoff=0.2,
        classify_object="projected_component",
        component_set="paper_gamma",
    )


def run_suite(mu_selection: str, args: argparse.Namespace) -> list[dict[str, object]]:
    point_args = solver_args(mu_selection, args)
    rows = []
    for mu, ratio in POINTS:
        row = solve_scan_point(mu, ratio, point_args)
        rows.append(row)
    return rows


def compact_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "mu": row["mu"],
        "r_over_J": row["r_over_J"],
        "winner_phase": row["winner_phase"],
        "winner_channel": row["winner_channel"],
        "winner_component_gamma": row["winner_component_gamma"],
        "winner_basis": row["winner_basis"],
        "winner_basis_overlap": row["winner_basis_overlap"],
        "lambda_winner": row["lambda_winner"],
        "lambda_runner_up": row["lambda_runner_up"],
        "lambda_margin": row["lambda_margin"],
    }


def print_table(label: str, rows: list[dict[str, object]]) -> None:
    print(f"\n[{label}]")
    for row in rows:
        print(
            "mu={mu:g}, r/J={ratio:g}: {gamma}, {basis}, "
            "nu={nu}, lambda={lam:.6e}, overlap={overlap:+.3f}".format(
                mu=float(row["mu"]),
                ratio=float(row["r_over_J"]),
                gamma=row["winner_component_gamma"],
                basis=row["winner_basis"],
                nu=row["winner_channel"],
                lam=float(row["lambda_winner"]),
                overlap=float(row["winner_basis_overlap"]),
            )
        )


def main() -> None:
    args = parse_args()
    output_path = Path(args.out)
    if not output_path.is_absolute():
        output_path = ROOT / output_path

    result = {
        "settings": {
            "Nk": args.Nk,
            "T": args.T,
            "J": args.J,
            "susceptibility_type": "projected_formula",
            "classify_object": "projected_component",
        },
        "positive_product": [compact_row(row) for row in run_suite("positive_product", args)],
        "all": [compact_row(row) for row in run_suite("all", args)],
    }

    print_table("positive_product", result["positive_product"])
    print_table("all", result["all"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"\nwrote {output_path}")


if __name__ == "__main__":
    main()
