from __future__ import annotations

"""Report scalar band-pairing modes from the single-band projected equation.

The reported object is the band gap function Omega_nu(k) for a fixed spin
channel nu.  Projected orbital form factors are still used to build the kernel,
but this script does not rank or label g_mu(k) Omega_nu(k) as Gamma components.
"""

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scan_symmetrized_single_band_phase_diagram import classify_basis  # noqa: E402
from symmetrized_lieb_gap_solver import (  # noqa: E402
    CHANNELS,
    ETA_NU,
    GapEquationParams,
    build_kernel_for_channel,
    compute_form_factors_on_grid,
    diagonalize_lieb_band,
    harmonic_overlaps,
    infer_s_mu_from_tau_z,
    pauli_basis_2,
)


DEFAULT_POINTS = "0.5:1.5,1.5:1.5,2.5:1.5,0.75:1.0753"


@dataclass(frozen=True)
class PointData:
    params: GapEquationParams
    band: Any
    g_mu: np.ndarray
    s_mu: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report single-band Omega_nu(k) pairing modes."
    )
    parser.add_argument("--Nk", type=int, default=21)
    parser.add_argument("--T", type=float, default=0.05)
    parser.add_argument("--J", type=float, default=2.0 / 3.0)
    parser.add_argument("--top-modes", type=int, default=4)
    parser.add_argument(
        "--points",
        default=DEFAULT_POINTS,
        help="Comma-separated mu:r_over_J pairs, for example '2.5:1.5'.",
    )
    parser.add_argument(
        "--mu-selection",
        default="all",
        choices=("all", "positive_product"),
        help="Kernel mu-selection used for the modes being reported.",
    )
    parser.add_argument(
        "--out-dir",
        default="symmetrized_lieb_gap_outputs/band_pairing_report",
    )
    return parser.parse_args()


def parse_points(points_text: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for item in points_text.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Point {item!r} must have form mu:r_over_J")
        mu_text, ratio_text = item.split(":", maxsplit=1)
        points.append((float(mu_text), float(ratio_text)))
    if not points:
        raise ValueError("At least one point is required")
    return points


def make_params(
    *,
    args: argparse.Namespace,
    mu: float,
    r_over_J: float,
) -> GapEquationParams:
    return GapEquationParams(
        Nk=args.Nk,
        mu_F=float(mu),
        T=args.T,
        g=1.0,
        J=args.J,
        r=float(r_over_J * args.J),
        susceptibility_type="projected_formula",
        kernel_sign=1.0,
        mu_selection=args.mu_selection,
        wrap_susceptibility_momentum=False,
        paper_chi_prefactor=1.0,
        make_plots=False,
        dense_eig_max_dim=1_000_000,
        overlap_weight_mode="pair_factor",
        harmonic_basis_set="paper",
    )


def prepare_point(params: GapEquationParams) -> PointData:
    band = diagonalize_lieb_band(params)
    tau_list, _ = pauli_basis_2()
    s_mu = infer_s_mu_from_tau_z(tau_list, tau_list[3])
    g_mu = compute_form_factors_on_grid(band.eigenvectors, tau_list, params.Nk)
    return PointData(params=params, band=band, g_mu=g_mu, s_mu=s_mu)


def top_channel_modes(data: PointData, channel: str, top_modes: int) -> list[dict[str, Any]]:
    kernel = build_kernel_for_channel(
        channel,
        data.params,
        data.band,
        data.g_mu,
        data.s_mu,
    )
    values, vectors = np.linalg.eig(kernel)
    order = np.argsort(np.real(values))[::-1][:top_modes]
    modes = []
    for channel_rank, idx in enumerate(order, start=1):
        omega = vectors[:, idx]
        omega_best = harmonic_overlaps(omega, data.band, data.params, top=1)[0]
        modes.append(
            {
                "channel": channel,
                "eta_nu": ETA_NU[channel],
                "channel_rank": channel_rank,
                "lambda": values[idx],
                "omega_basis": str(omega_best["name"]),
                "omega_phase": classify_basis(str(omega_best["name"])),
                "omega_basis_overlap": float(omega_best["overlap"]),
                "omega_abs_basis_overlap": float(omega_best["abs_overlap"]),
            }
        )
    return modes


def point_label(mu: float, r_over_J: float) -> str:
    return f"mu={mu:.8g},r_over_J={r_over_J:.8g}"


def mode_row(
    *,
    point: str,
    params: GapEquationParams,
    mode: dict[str, Any],
    overall_rank: int,
) -> dict[str, Any]:
    return {
        "point": point,
        "mu_F": params.mu_F,
        "r_over_J": params.r / params.J,
        "mode_overall_rank": overall_rank,
        "mode_channel": mode["channel"],
        "eta_nu": mode["eta_nu"],
        "mode_channel_rank": mode["channel_rank"],
        "mode_lambda_real": float(np.real(mode["lambda"])),
        "mode_lambda_imag": float(np.imag(mode["lambda"])),
        "omega_phase": mode["omega_phase"],
        "omega_basis": mode["omega_basis"],
        "omega_basis_overlap": mode["omega_basis_overlap"],
        "omega_abs_basis_overlap": mode["omega_abs_basis_overlap"],
    }


def run_point(
    *,
    args: argparse.Namespace,
    mu: float,
    r_over_J: float,
) -> list[dict[str, Any]]:
    data = prepare_point(make_params(args=args, mu=mu, r_over_J=r_over_J))
    modes: list[dict[str, Any]] = []
    for channel in CHANNELS:
        modes.extend(top_channel_modes(data, channel, args.top_modes))
    modes.sort(key=lambda mode: float(np.real(mode["lambda"])), reverse=True)

    point = point_label(mu, r_over_J)
    return [
        mode_row(
            point=point,
            params=data.params,
            mode=mode,
            overall_rank=overall_rank,
        )
        for overall_rank, mode in enumerate(modes, start=1)
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, Any]], max_modes_per_point: int = 8) -> None:
    print("\nBand pairing mode report")
    print("------------------------")
    by_point: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_point.setdefault(str(row["point"]), []).append(row)
    for point, point_rows in by_point.items():
        print(point)
        for row in point_rows[:max_modes_per_point]:
            print(
                "  #{rank:02d} nu={nu} eta={eta:+d} lambda={lam:+.6e} "
                "{phase} basis={basis} overlap={overlap:+.3f}".format(
                    rank=int(row["mode_overall_rank"]),
                    nu=row["mode_channel"],
                    eta=int(row["eta_nu"]),
                    lam=float(row["mode_lambda_real"]),
                    phase=row["omega_phase"],
                    basis=row["omega_basis"],
                    overlap=float(row["omega_basis_overlap"]),
                )
            )


def main() -> None:
    args = parse_args()
    points = parse_points(args.points)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for mu, ratio in points:
        rows.extend(run_point(args=args, mu=mu, r_over_J=ratio))

    write_csv(out_dir / "band_pairing_report.csv", rows)

    json_summary = {
        "settings": {
            "Nk": args.Nk,
            "T": args.T,
            "J": args.J,
            "top_modes_per_channel": args.top_modes,
            "mu_selection": args.mu_selection,
            "susceptibility_type": "projected_formula",
            "overlap_weight_mode": "pair_factor",
            "basis": "single-band Omega_nu(k)",
            "interpretation": (
                "Rows classify the scalar band pairing eigenfunction Omega_nu(k). "
                "Projected orbital form factors enter only through the kernel."
            ),
        },
        "mode_summary": rows,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_summary, indent=2), encoding="utf-8"
    )

    print_summary(rows)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
