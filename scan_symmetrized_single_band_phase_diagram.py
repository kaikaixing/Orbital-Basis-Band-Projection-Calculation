from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap

from symmetrized_lieb_gap_solver import (
    CHANNELS,
    ETA_NU,
    GapEquationParams,
    build_kernel_for_channel,
    compute_form_factors_on_grid,
    diagonalize_lieb_band,
    harmonic_overlaps,
    infer_s_mu_from_tau_z,
    pauli_basis_2,
    solve_channel_eigenproblem,
)


PHASE_TO_CODE = {
    "s_prime": 0,
    "p_prime": 1,
    "d_prime": 2,
    "p_wave": 3,
    "other": 4,
}

PHASE_LABELS = {
    "s_prime": "s'",
    "p_prime": "p'",
    "d_prime": "d'",
    "p_wave": "p",
    "other": "other",
}

MU_LABELS = ("0", "x", "y", "z")

PAPER_GAMMA_LABELS = {
    ("0", "0"): "Gamma1 tau0 sigma0",
    ("z", "z"): "Gamma2 tauz sigmaz",
    ("z", "0"): "Gamma3 tauz sigma0",
    ("0", "z"): "Gamma4 tau0 sigmaz",
    ("x", "y"): "Gamma5 taux sigmay",
    ("y", "x"): "Gamma6 tauy sigmax",
    ("x", "x"): "Gamma7 taux sigmax",
    ("y", "y"): "Gamma8 tauy sigmay",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan the fixed-nu single-band projected phase diagram from "
            "symmetrized_lieb_gap_solver.py."
        )
    )
    parser.add_argument("--Nk", type=int, default=13)
    parser.add_argument("--mu-min", type=float, default=0.0)
    parser.add_argument("--mu-max", type=float, default=3.0)
    parser.add_argument("--mu-points", type=int, default=25)
    parser.add_argument("--r-over-J-min", type=float, default=1.001)
    parser.add_argument("--r-over-J-max", type=float, default=2.0)
    parser.add_argument("--r-over-J-points", type=int, default=31)
    parser.add_argument(
        "--J",
        type=float,
        default=2.0 / 3.0,
        help="Base J.  r is set to (r/J)*J at every scan point.",
    )
    parser.add_argument("--T", type=float, default=0.05)
    parser.add_argument("--g", type=float, default=1.0)
    parser.add_argument(
        "--susceptibility-type",
        default="projected_formula",
        choices=(
            "projected_formula",
            "lattice",
            "paper56",
            "paper78",
            "paper1234",
        ),
    )
    parser.add_argument(
        "--wrap-susceptibility-momentum",
        action="store_true",
        help="Wrap chi momenta back into [-pi, pi).  Leave off for the PDF-final half-angle formula.",
    )
    parser.add_argument(
        "--paper-chi-prefactor",
        type=float,
        default=1.0,
        help="Use 1.0 for the PDF-final chi, or 0.25 for the older framework convention.",
    )
    parser.add_argument(
        "--mu-selection",
        default="all",
        choices=("all", "positive_product"),
    )
    parser.add_argument(
        "--overlap-weight-mode",
        default="pair_factor",
        choices=("uniform", "pair_factor", "fs_shell"),
    )
    parser.add_argument(
        "--classify-object",
        default="projected_component",
        choices=("omega", "projected_component"),
        help=(
            "Classify the solved Omega_nu(k), or classify the rank-one "
            "projected diagnostic component "
            "C_proj_{mu,nu}(k)=g_mu(k) Omega_nu(k). This diagnostic is not "
            "the full orbital HS field Delta_{mu,nu}(k)."
        ),
    )
    parser.add_argument(
        "--component-set",
        default="paper_gamma",
        choices=("paper_gamma", "all"),
        help="Diagnostic projected components considered when --classify-object=projected_component.",
    )
    parser.add_argument("--fs-cutoff", type=float, default=0.2)
    parser.add_argument(
        "--out-dir",
        default="symmetrized_lieb_gap_outputs/single_band_phase",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.r_over_J_min <= 1.0:
        raise ValueError("--r-over-J-min must be greater than 1.0 to avoid chi singularities")
    if args.r_over_J_max <= args.r_over_J_min:
        raise ValueError("--r-over-J-max must be greater than --r-over-J-min")
    if args.Nk < 5:
        raise ValueError("--Nk must be at least 5")


def classify_basis(name: str) -> str:
    if name.startswith("s prime"):
        return "s_prime"
    if name.startswith("d prime"):
        return "d_prime"
    if name.startswith("u_plus") or name.startswith("u_minus"):
        return "p_prime"
    if name.startswith("sin(kx/2") or name.startswith("sin(ky/2"):
        return "p_prime"
    if name.startswith("p wave f1") or name.startswith("p wave f2"):
        return "p_wave"
    return "other"


def align_phase(vec: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(vec)))
    if abs(vec[idx]) < 1e-14:
        return vec
    return vec * np.exp(-1.0j * np.angle(vec[idx]))


def component_harmonic_overlaps(
    component: np.ndarray,
    band,
    params: GapEquationParams,
    top: int = 1,
) -> list[dict[str, object]]:
    data = np.real(align_phase(component)).reshape(params.Nk, params.Nk)
    weight = np.maximum(band.W.reshape(params.Nk, params.Nk), 0.0)
    if params.overlap_weight_mode == "uniform":
        weight = np.ones_like(weight)
    elif params.overlap_weight_mode == "fs_shell":
        abs_xi = np.abs(band.xi.reshape(params.Nk, params.Nk))
        shell = abs_xi <= params.fs_cutoff
        if not np.any(shell):
            shell = abs_xi == np.nanmin(abs_xi)
        weight = shell.astype(float)

    rows = []
    for name, basis in [
        item for item in [
            ("s prime ~ cos(kx/2) cos(ky/2)", np.cos(0.5 * band.kx_grid) * np.cos(0.5 * band.ky_grid)),
            ("d prime ~ sin(kx/2) sin(ky/2)", np.sin(0.5 * band.kx_grid) * np.sin(0.5 * band.ky_grid)),
            ("p wave f1 ~ sin kx (1 - cos ky)", np.sin(band.kx_grid) * (1.0 - np.cos(band.ky_grid))),
            ("p wave f2 ~ sin ky (1 - cos kx)", np.sin(band.ky_grid) * (1.0 - np.cos(band.kx_grid))),
            ("u_plus ~ sin((kx+ky)/2)", np.sin(0.5 * (band.kx_grid + band.ky_grid))),
            ("u_minus ~ sin((kx-ky)/2)", np.sin(0.5 * (band.kx_grid - band.ky_grid))),
        ]
    ]:
        numerator = float(np.sum(weight * data * basis))
        data_norm = float(np.sum(weight * data * data))
        basis_norm = float(np.sum(weight * basis * basis))
        denom = float(np.sqrt(data_norm * basis_norm))
        overlap = numerator / denom if denom > 1e-14 else 0.0
        rows.append(
            {
                "name": name,
                "overlap": overlap,
                "abs_overlap": abs(overlap),
            }
        )
    rows.sort(key=lambda row: row["abs_overlap"], reverse=True)
    return rows[:top]


def projected_component_summary(
    omega: np.ndarray,
    channel: str,
    g_mu: np.ndarray,
    band,
    params: GapEquationParams,
    component_set: str,
) -> dict[str, object]:
    rows = []
    for mu_index, mu_label in enumerate(MU_LABELS):
        gamma_label = PAPER_GAMMA_LABELS.get(
            (mu_label, channel), f"tau{mu_label} sigma{channel}"
        )
        if component_set == "paper_gamma" and (mu_label, channel) not in PAPER_GAMMA_LABELS:
            continue
        component = g_mu[mu_index] * omega
        norm = float(np.linalg.norm(component))
        overlap = component_harmonic_overlaps(component, band, params, top=1)[0]
        rows.append(
            {
                "mu": mu_label,
                "gamma": gamma_label,
                "component_norm": norm,
                "basis": str(overlap["name"]),
                "basis_overlap": float(overlap["overlap"]),
                "phase": classify_basis(str(overlap["name"])),
            }
        )
    if not rows:
        return {
            "mu": "",
            "gamma": "",
            "component_norm": 0.0,
            "basis": "unclassified",
            "basis_overlap": 0.0,
            "phase": "other",
        }
    rows.sort(key=lambda row: row["component_norm"], reverse=True)
    return rows[0]


def solve_scan_point(mu: float, r_over_J: float, args: argparse.Namespace) -> dict[str, object]:
    params = GapEquationParams(
        Nk=args.Nk,
        mu_F=float(mu),
        T=args.T,
        g=args.g,
        r=float(r_over_J * args.J),
        J=args.J,
        susceptibility_type=args.susceptibility_type,
        wrap_susceptibility_momentum=bool(args.wrap_susceptibility_momentum),
        paper_chi_prefactor=args.paper_chi_prefactor,
        mu_selection=args.mu_selection,
        make_plots=False,
        dense_eig_max_dim=1_000_000,
        overlap_weight_mode=args.overlap_weight_mode,
        fs_cutoff=args.fs_cutoff,
        harmonic_basis_set="paper",
    )
    band = diagonalize_lieb_band(params)
    tau_list, _ = pauli_basis_2()
    s_mu = infer_s_mu_from_tau_z(tau_list, tau_list[3])
    g_mu = compute_form_factors_on_grid(band.eigenvectors, tau_list, params.Nk)

    channel_rows = []
    for channel in CHANNELS:
        kernel = build_kernel_for_channel(channel, params, band, g_mu, s_mu)
        eigenvalue, eigenvector = solve_channel_eigenproblem(
            kernel, params, ETA_NU[channel]
        )
        if args.classify_object == "omega":
            overlap = harmonic_overlaps(eigenvector, band, params, top=1)[0]
            component_summary = {
                "mu": "",
                "gamma": f"Omega_{channel}",
                "component_norm": float(np.linalg.norm(eigenvector)),
                "basis": str(overlap["name"]),
                "basis_overlap": float(overlap["overlap"]),
                "phase": classify_basis(str(overlap["name"])),
            }
        else:
            component_summary = projected_component_summary(
                eigenvector,
                channel,
                g_mu,
                band,
                params,
                args.component_set,
            )
        channel_rows.append(
            {
                "channel": channel,
                "eta": ETA_NU[channel],
                "lambda": float(np.real(eigenvalue)),
                "lambda_imag": float(np.imag(eigenvalue)),
                "component_mu": component_summary["mu"],
                "component_gamma": component_summary["gamma"],
                "component_norm": component_summary["component_norm"],
                "basis": component_summary["basis"],
                "basis_overlap": component_summary["basis_overlap"],
                "phase": component_summary["phase"],
            }
        )

    channel_rows.sort(key=lambda row: row["lambda"], reverse=True)
    winner = channel_rows[0]
    runner_up = channel_rows[1]
    out: dict[str, object] = {
        "mu": float(mu),
        "r_over_J": float(r_over_J),
        "r": float(r_over_J * args.J),
        "J": float(args.J),
        "winner_phase": winner["phase"],
        "winner_channel": winner["channel"],
        "winner_eta": winner["eta"],
        "winner_component_mu": winner["component_mu"],
        "winner_component_gamma": winner["component_gamma"],
        "winner_component_norm": winner["component_norm"],
        "winner_basis": winner["basis"],
        "winner_basis_overlap": winner["basis_overlap"],
        "lambda_winner": winner["lambda"],
        "lambda_runner_up": runner_up["lambda"],
        "lambda_margin": winner["lambda"] - runner_up["lambda"],
        "lambda_0": next(row["lambda"] for row in channel_rows if row["channel"] == "0"),
        "lambda_x": next(row["lambda"] for row in channel_rows if row["channel"] == "x"),
        "lambda_y": next(row["lambda"] for row in channel_rows if row["channel"] == "y"),
        "lambda_z": next(row["lambda"] for row in channel_rows if row["channel"] == "z"),
        "basis_0": next(row["basis"] for row in channel_rows if row["channel"] == "0"),
        "basis_x": next(row["basis"] for row in channel_rows if row["channel"] == "x"),
        "basis_y": next(row["basis"] for row in channel_rows if row["channel"] == "y"),
        "basis_z": next(row["basis"] for row in channel_rows if row["channel"] == "z"),
        "gamma_0": next(row["component_gamma"] for row in channel_rows if row["channel"] == "0"),
        "gamma_x": next(row["component_gamma"] for row in channel_rows if row["channel"] == "x"),
        "gamma_y": next(row["component_gamma"] for row in channel_rows if row["channel"] == "y"),
        "gamma_z": next(row["component_gamma"] for row in channel_rows if row["channel"] == "z"),
    }
    return out


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    fieldnames = [
        "mu",
        "r_over_J",
        "r",
        "J",
        "winner_phase",
        "winner_channel",
        "winner_eta",
        "winner_component_mu",
        "winner_component_gamma",
        "winner_component_norm",
        "winner_basis",
        "winner_basis_overlap",
        "lambda_winner",
        "lambda_runner_up",
        "lambda_margin",
        "lambda_0",
        "lambda_x",
        "lambda_y",
        "lambda_z",
        "basis_0",
        "basis_x",
        "basis_y",
        "basis_z",
        "gamma_0",
        "gamma_x",
        "gamma_y",
        "gamma_z",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_to_grids(
    rows: list[dict[str, object]], mus: np.ndarray, r_over_Js: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.empty((len(mus), len(r_over_Js)), dtype=float)
    leading_lambda = np.empty_like(phase)
    margin = np.empty_like(phase)
    by_key = {(row["mu"], row["r_over_J"]): row for row in rows}
    for iy, mu in enumerate(mus):
        for ix, ratio in enumerate(r_over_Js):
            row = by_key[(float(mu), float(ratio))]
            phase[iy, ix] = PHASE_TO_CODE[str(row["winner_phase"])]
            leading_lambda[iy, ix] = float(row["lambda_winner"])
            margin[iy, ix] = float(row["lambda_margin"])
    return phase, leading_lambda, margin


def plot_phase_map(
    r_over_Js: np.ndarray, mus: np.ndarray, phase: np.ndarray, path: Path
) -> None:
    cmap = ListedColormap(
        [
            "#f4a6b7",  # s'
            "#c9b8e8",  # p'
            "#9ec5e8",  # d'
            "#f2d16b",  # p
            "#bab0ac",  # other
        ]
    )
    norm = BoundaryNorm(np.arange(-0.5, len(PHASE_TO_CODE) + 0.5, 1.0), cmap.N)
    fig, ax = plt.subplots(figsize=(6.2, 5.3), constrained_layout=True)
    mesh = ax.pcolormesh(r_over_Js, mus, phase, cmap=cmap, norm=norm, shading="nearest")
    cbar = fig.colorbar(mesh, ax=ax, ticks=list(PHASE_TO_CODE.values()))
    cbar.ax.set_yticklabels([PHASE_LABELS[key] for key in PHASE_TO_CODE])
    ax.set_xlabel(r"$r/J$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title("Single-band projected pairing phase")
    for phase_name, code in PHASE_TO_CODE.items():
        mask = phase == code
        if np.any(mask) and np.any(~mask):
            ax.contour(
                r_over_Js,
                mus,
                mask.astype(float),
                levels=[0.5],
                colors="k",
                linewidths=0.75,
                alpha=0.55,
            )
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_scalar_map(
    r_over_Js: np.ndarray,
    mus: np.ndarray,
    values: np.ndarray,
    path: Path,
    label: str,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.3), constrained_layout=True)
    mesh = ax.pcolormesh(r_over_Js, mus, values, cmap="viridis", shading="nearest")
    fig.colorbar(mesh, ax=ax, label=label)
    ax.set_xlabel(r"$r/J$")
    ax.set_ylabel(r"$\mu$")
    ax.set_title(title)
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    counts: dict[str, int] = {}
    for row in rows:
        phase = str(row["winner_phase"])
        counts[phase] = counts.get(phase, 0) + 1
    sample_points = {}
    for mu in (0.5, 1.5, 2.5):
        candidates = sorted(rows, key=lambda row: abs(float(row["mu"]) - mu) + abs(float(row["r_over_J"]) - 1.5))
        sample_points[str(mu)] = candidates[0]
    return {"phase_counts": counts, "line_r_over_J_1p5_samples": sample_points}


def main() -> None:
    args = parse_args()
    validate_args(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mus = np.linspace(args.mu_min, args.mu_max, args.mu_points)
    r_over_Js = np.linspace(
        args.r_over_J_min, args.r_over_J_max, args.r_over_J_points
    )
    rows = []
    total = len(mus) * len(r_over_Js)
    count = 0
    print(
        f"Scanning single-band projected phase map: Nk={args.Nk}, "
        f"{len(mus)} mu points x {len(r_over_Js)} r/J points"
    )
    for mu in mus:
        for ratio in r_over_Js:
            count += 1
            if count == 1 or count % max(1, total // 20) == 0 or count == total:
                print(f"[{count:4d}/{total}] mu={mu:.4g}, r/J={ratio:.4g}")
            rows.append(solve_scan_point(float(mu), float(ratio), args))

    csv_path = out_dir / "single_band_phase_scan.csv"
    npz_path = out_dir / "single_band_phase_scan.npz"
    phase_path = out_dir / "single_band_phase_map.png"
    lambda_path = out_dir / "single_band_lambda_map.png"
    margin_path = out_dir / "single_band_margin_map.png"
    summary_path = out_dir / "run_summary.json"

    write_csv(rows, csv_path)
    phase, leading_lambda, margin = rows_to_grids(rows, mus, r_over_Js)
    np.savez(
        npz_path,
        mus=mus,
        r_over_Js=r_over_Js,
        phase=phase,
        leading_lambda=leading_lambda,
        lambda_margin=margin,
        phase_to_code=PHASE_TO_CODE,
    )
    plot_phase_map(r_over_Js, mus, phase, phase_path)
    plot_scalar_map(
        r_over_Js,
        mus,
        leading_lambda,
        lambda_path,
        r"$\lambda_{\max}$",
        "Leading eigenvalue",
    )
    plot_scalar_map(
        r_over_Js,
        mus,
        margin,
        margin_path,
        r"$\lambda_1-\lambda_2$",
        "Leading-channel margin",
    )

    summary = {
        "command_args": vars(args),
        "phase_to_code": PHASE_TO_CODE,
        "summary": summarize(rows),
        "outputs": {
            "csv": str(csv_path.resolve()),
            "npz": str(npz_path.resolve()),
            "phase_map": str(phase_path.resolve()),
            "lambda_map": str(lambda_path.resolve()),
            "margin_map": str(margin_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Outputs written to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
