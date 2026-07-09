from __future__ import annotations

"""Report diagnostic Gamma labels for scalar band-projected modes.

This diagnostic sits between the two main routes.  It first solves the scalar
band-projected equation for Omega_nu(k), then ranks paper-style
Gamma_a=tau_mu sigma_nu labels using the projected orbital form-factor
information.

Important: these labels are only a bookkeeping bridge between the projected
scalar calculation and the paper's Gamma notation.  This script is not a full
Gamma-basis HS solver and does not reconstruct Delta_{mu,nu}(k).
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

from scan_symmetrized_single_band_phase_diagram import (  # noqa: E402
    classify_basis,
    component_harmonic_overlaps,
)
from symmetrized_lieb_gap_solver import (  # noqa: E402
    CHANNELS,
    ETA_NU,
    GapEquationParams,
    build_kernel_for_channel,
    compute_form_factors_on_grid,
    diagonalize_lieb_band,
    harmonic_overlap_weight,
    harmonic_overlaps,
    infer_s_mu_from_tau_z,
    pauli_basis_2,
)


DEFAULT_POINTS = "0.5:1.5,1.5:1.5,2.5:1.5,0.75:1.0753"
MU_LABELS = ("0", "x", "y", "z")


GAMMA16 = [
    (1, "0", "0"),
    (2, "z", "z"),
    (3, "z", "0"),
    (4, "0", "z"),
    (5, "x", "y"),
    (6, "y", "x"),
    (7, "x", "x"),
    (8, "y", "y"),
    (9, "y", "0"),
    (10, "y", "z"),
    (11, "x", "0"),
    (12, "x", "z"),
    (13, "0", "x"),
    (14, "0", "y"),
    (15, "z", "x"),
    (16, "z", "y"),
]


S_MU = {"0": +1, "x": -1, "y": -1, "z": +1}
S_NU = {"0": +1, "x": -1, "y": -1, "z": +1}
ETA_NU_LABEL = {"0": -1, "x": -1, "y": +1, "z": -1}


def gamma_label(index: int, mu: str, nu: str) -> str:
    return f"Gamma{index} tau{mu} sigma{nu}"


def sign_text(value: int | float) -> str:
    return "+" if value > 0 else "-"


def pair_characters(mu: str, nu: str) -> dict[str, str]:
    # Pair-bilinear characters using the same minimal generators as
    # ssg_pairing_symmetry_check.py.  The physical spin pi rotations include
    # the SU(2) spinor phase, so the pair character differs from vertex
    # conjugation.
    g_tau_z = S_MU[mu]
    s_pi_z = -S_NU[nu]
    tau_x = {"0": +1, "x": +1, "y": -1, "z": -1}[mu]
    spin_x_conjugation = {"0": +1, "x": +1, "y": -1, "z": -1}[nu]
    c_tau_x_s_pi_x = -tau_x * spin_x_conjugation
    return {
        "pair_char_G_tau_z": sign_text(g_tau_z),
        "pair_char_S_pi_z": sign_text(s_pi_z),
        "pair_char_C_tau_x_S_pi_x": sign_text(c_tau_x_s_pi_x),
    }


@dataclass(frozen=True)
class PointData:
    params: GapEquationParams
    band: Any
    g_mu: np.ndarray
    s_mu: np.ndarray
    weight: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report Gamma1-16 components of projected fixed-nu pairing modes."
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
        default="symmetrized_lieb_gap_outputs/gamma_component_report",
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
    weight = harmonic_overlap_weight(params, band).reshape(-1)
    return PointData(params=params, band=band, g_mu=g_mu, s_mu=s_mu, weight=weight)


def weighted_norm(vec: np.ndarray, weight: np.ndarray) -> float:
    return float(np.sqrt(np.sum(weight * np.abs(vec) ** 2)))


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
        omega_overlaps = harmonic_overlaps(omega, data.band, data.params, top=1)
        omega_best = omega_overlaps[0]
        modes.append(
            {
                "channel": channel,
                "channel_rank": channel_rank,
                "lambda": values[idx],
                "omega": omega,
                "omega_basis": str(omega_best["name"]),
                "omega_phase": classify_basis(str(omega_best["name"])),
                "omega_basis_overlap": float(omega_best["overlap"]),
            }
        )
    return modes


def active_component(
    data: PointData,
    *,
    omega: np.ndarray,
    mu: str,
) -> tuple[np.ndarray, float]:
    mu_index = MU_LABELS.index(mu)
    component = data.g_mu[mu_index] * omega
    return component, weighted_norm(component, data.weight)


def component_report_rows(
    data: PointData,
    *,
    point: str,
    mode: dict[str, Any],
    overall_rank: int,
) -> list[dict[str, Any]]:
    channel = str(mode["channel"])
    active_norms: dict[int, float] = {}
    active_components: dict[int, np.ndarray] = {}

    for index, mu, nu in GAMMA16:
        if nu != channel:
            continue
        component, norm = active_component(data, omega=mode["omega"], mu=mu)
        active_norms[index] = norm
        active_components[index] = component

    norm_sq_total = sum(value * value for value in active_norms.values())
    rows: list[dict[str, Any]] = []
    for index, mu, nu in GAMMA16:
        active = nu == channel
        norm = active_norms.get(index, 0.0)
        basis = ""
        phase = ""
        basis_overlap = 0.0
        abs_basis_overlap = 0.0
        if active and norm > 1e-12:
            overlaps = component_harmonic_overlaps(
                active_components[index], data.band, data.params, top=1
            )
            basis = str(overlaps[0]["name"])
            phase = classify_basis(basis)
            basis_overlap = float(overlaps[0]["overlap"])
            abs_basis_overlap = float(overlaps[0]["abs_overlap"])

        attractive_sign = S_MU[mu] * S_NU[nu]
        row = {
            "point": point,
            "mu_F": data.params.mu_F,
            "r_over_J": data.params.r / data.params.J,
            "mode_overall_rank": overall_rank,
            "mode_channel": channel,
            "mode_channel_rank": mode["channel_rank"],
            "mode_lambda_real": float(np.real(mode["lambda"])),
            "mode_lambda_imag": float(np.imag(mode["lambda"])),
            "omega_phase": mode["omega_phase"],
            "omega_basis": mode["omega_basis"],
            "omega_basis_overlap": mode["omega_basis_overlap"],
            "gamma_index": index,
            "gamma": gamma_label(index, mu, nu),
            "tau_mu": mu,
            "sigma_nu": nu,
            "active_in_fixed_nu_mode": active,
            "weighted_norm": norm,
            "norm_fraction_within_active_sigma": (
                norm * norm / norm_sq_total if norm_sq_total > 1e-24 else 0.0
            ),
            "basis": basis,
            "phase": phase,
            "basis_overlap": basis_overlap,
            "abs_basis_overlap": abs_basis_overlap,
            "s_mu": S_MU[mu],
            "s_nu": S_NU[nu],
            "s_mu_s_nu": attractive_sign,
            "attractive_by_positive_product_rule": attractive_sign > 0,
            "eta_nu": ETA_NU_LABEL[nu],
        }
        row.update(pair_characters(mu, nu))
        rows.append(row)
    return rows


def summarize_modes(component_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in component_rows:
        grouped.setdefault((str(row["point"]), int(row["mode_overall_rank"])), []).append(row)

    summary_rows = []
    for (_, _), rows in grouped.items():
        active_rows = [row for row in rows if row["active_in_fixed_nu_mode"]]
        active_rows.sort(key=lambda row: float(row["weighted_norm"]), reverse=True)
        leading = active_rows[0] if active_rows else rows[0]
        row0 = rows[0]
        summary_rows.append(
            {
                "point": row0["point"],
                "mu_F": row0["mu_F"],
                "r_over_J": row0["r_over_J"],
                "mode_overall_rank": row0["mode_overall_rank"],
                "mode_channel": row0["mode_channel"],
                "mode_channel_rank": row0["mode_channel_rank"],
                "mode_lambda_real": row0["mode_lambda_real"],
                "mode_lambda_imag": row0["mode_lambda_imag"],
                "omega_phase": row0["omega_phase"],
                "omega_basis": row0["omega_basis"],
                "omega_basis_overlap": row0["omega_basis_overlap"],
                "leading_gamma": leading["gamma"],
                "leading_phase": leading["phase"],
                "leading_basis": leading["basis"],
                "leading_basis_overlap": leading["basis_overlap"],
                "leading_norm_fraction": leading["norm_fraction_within_active_sigma"],
                "active_gamma_count": len(active_rows),
            }
        )
    summary_rows.sort(key=lambda row: (row["point"], int(row["mode_overall_rank"])))
    return summary_rows


def point_label(mu: float, r_over_J: float) -> str:
    return f"mu={mu:.8g},r_over_J={r_over_J:.8g}"


def run_point(
    *,
    args: argparse.Namespace,
    mu: float,
    r_over_J: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    data = prepare_point(make_params(args=args, mu=mu, r_over_J=r_over_J))
    modes: list[dict[str, Any]] = []
    for channel in CHANNELS:
        modes.extend(top_channel_modes(data, channel, args.top_modes))
    modes.sort(key=lambda mode: float(np.real(mode["lambda"])), reverse=True)

    rows: list[dict[str, Any]] = []
    point = point_label(mu, r_over_J)
    for overall_rank, mode in enumerate(modes, start=1):
        rows.extend(
            component_report_rows(
                data,
                point=point,
                mode=mode,
                overall_rank=overall_rank,
            )
        )
    return rows, summarize_modes(rows)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: list[dict[str, Any]], max_modes_per_point: int = 8) -> None:
    print("\nGamma1-16 component report")
    print("--------------------------")
    by_point: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        by_point.setdefault(str(row["point"]), []).append(row)
    for point, rows in by_point.items():
        print(point)
        for row in rows[:max_modes_per_point]:
            print(
                "  #{rank:02d} nu={nu} lambda={lam:+.6e} "
                "{gamma} {phase} frac={frac:.3f} basis={basis} overlap={overlap:+.3f}".format(
                    rank=int(row["mode_overall_rank"]),
                    nu=row["mode_channel"],
                    lam=float(row["mode_lambda_real"]),
                    gamma=row["leading_gamma"],
                    phase=row["leading_phase"],
                    frac=float(row["leading_norm_fraction"]),
                    basis=row["leading_basis"],
                    overlap=float(row["leading_basis_overlap"]),
                )
            )


def gamma_definition_rows() -> list[dict[str, Any]]:
    rows = []
    for index, mu, nu in GAMMA16:
        row = {
            "gamma_index": index,
            "gamma": gamma_label(index, mu, nu),
            "tau_mu": mu,
            "sigma_nu": nu,
            "s_mu": S_MU[mu],
            "s_nu": S_NU[nu],
            "s_mu_s_nu": S_MU[mu] * S_NU[nu],
            "attractive_by_positive_product_rule": S_MU[mu] * S_NU[nu] > 0,
            "eta_nu": ETA_NU_LABEL[nu],
        }
        row.update(pair_characters(mu, nu))
        rows.append(row)
    return rows


def main() -> None:
    args = parse_args()
    points = parse_points(args.points)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    component_rows_all: list[dict[str, Any]] = []
    mode_summary_all: list[dict[str, Any]] = []
    for mu, ratio in points:
        component_rows, mode_summary = run_point(args=args, mu=mu, r_over_J=ratio)
        component_rows_all.extend(component_rows)
        mode_summary_all.extend(mode_summary)

    write_csv(out_dir / "gamma16_definitions.csv", gamma_definition_rows())
    write_csv(out_dir / "gamma_component_report.csv", component_rows_all)
    write_csv(out_dir / "mode_summary.csv", mode_summary_all)

    json_summary = {
        "settings": {
            "Nk": args.Nk,
            "T": args.T,
            "J": args.J,
            "top_modes_per_channel": args.top_modes,
            "mu_selection": args.mu_selection,
            "susceptibility_type": "projected_formula",
            "overlap_weight_mode": "pair_factor",
            "basis": "Gamma1-16 tau_mu sigma_nu",
            "fixed_nu_limitation": (
                "Each current single-band eigenmode has one fixed sigma_nu. "
                "Rows with a different sigma_nu are reported as inactive zero "
                "components; a full BdG/multicomponent solver can activate and "
                "mix more sectors."
            ),
        },
        "gamma_definitions": gamma_definition_rows(),
        "mode_summary": mode_summary_all,
    }
    (out_dir / "summary.json").write_text(
        json.dumps(json_summary, indent=2), encoding="utf-8"
    )

    print_summary(mode_summary_all)
    print(f"\nWrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
