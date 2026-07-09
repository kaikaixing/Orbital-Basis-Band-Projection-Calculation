from __future__ import annotations

"""Run compact full Gamma-basis benchmarks for the teacher-facing comparison.

This script is the recommended entry point for the orbital-basis calculation.
It prints the leading Gamma component and the dominant momentum harmonic at
the same benchmark points used by the band-projection scripts.
"""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np

from full_16_gamma_pairing_solver import (
    GAMMA16,
    ModelParams,
    allowed_pairs,
    block_groups,
    build_mchi_generalized,
    channel_weights,
    compute_bubbles,
    make_grid,
    selected_gammas,
    solve_eigenproblem,
)


DEFAULT_POINTS = (
    (0.5, 1.5),
    (1.5, 1.5),
    (2.5, 1.5),
    (0.75, 1.0753),
)


def harmonic_basis(qx: np.ndarray, qy: np.ndarray) -> list[tuple[str, np.ndarray]]:
    return [
        ("d' ~ sin(kx/2) sin(ky/2)", np.sin(qx / 2.0) * np.sin(qy / 2.0)),
        ("s' ~ cos(kx/2) cos(ky/2)", np.cos(qx / 2.0) * np.cos(qy / 2.0)),
        ("p+ ~ sin((kx+ky)/2)", np.sin((qx + qy) / 2.0)),
        ("p- ~ sin((kx-ky)/2)", np.sin((qx - qy) / 2.0)),
        ("f1 ~ sin kx (1-cos ky)", np.sin(qx) * (1.0 - np.cos(qy))),
        ("f2 ~ sin ky (1-cos kx)", np.sin(qy) * (1.0 - np.cos(qx))),
    ]


def align_mode(vec: np.ndarray) -> np.ndarray:
    idx = int(np.argmax(np.abs(vec)))
    if abs(vec[idx]) < 1e-14:
        return vec
    phase = np.angle(vec[idx])
    return vec * np.exp(-1j * phase)


def best_harmonic(
    vec: np.ndarray,
    *,
    dominant_index: int,
    n_gamma: int,
    qx: np.ndarray,
    qy: np.ndarray,
    Nk: int,
) -> tuple[str, float]:
    aligned = align_mode(vec)
    components = aligned.reshape(n_gamma, Nk, Nk)
    component = np.real(components[dominant_index])
    comp_norm2 = float(np.sum(component * component))
    best_name = ""
    best_overlap = 0.0
    for name, basis in harmonic_basis(qx, qy):
        basis_norm2 = float(np.sum(basis * basis))
        if comp_norm2 < 1e-14 or basis_norm2 < 1e-14:
            overlap = 0.0
        else:
            overlap = float(np.sum(component * basis) / np.sqrt(comp_norm2 * basis_norm2))
        if abs(overlap) > abs(best_overlap):
            best_name = name
            best_overlap = overlap
    return best_name, best_overlap


def solve_point(
    *,
    mu: float,
    r_over_J: float,
    params: ModelParams,
    basis_set: str,
    block_mode: str,
    sign_side: str,
) -> dict[str, object]:
    p = replace(params, mu=mu, m=r_over_J * params.J)
    gammas = selected_gammas(basis_set)
    groups = block_groups(gammas, block_mode)
    pairs = allowed_pairs(groups)
    _, _, qx, qy, vqx, vqy, Nk, _, pref = make_grid(p)
    bubble, bubblep = compute_bubbles(vqx, vqy, gammas, pairs, p)
    Mchi = build_mchi_generalized(
        vqx,
        vqy,
        gammas,
        pairs,
        bubble,
        bubblep,
        p,
        sign_side=sign_side,
    )
    evals, evecs = solve_eigenproblem(Mchi, pref, p.num_eigs)
    weights = channel_weights(evecs[:, 0], len(gammas), len(vqx))
    dominant = int(np.argmax(weights))
    gamma = gammas[dominant]
    harmonic, overlap = best_harmonic(
        evecs[:, 0],
        dominant_index=dominant,
        n_gamma=len(gammas),
        qx=qx,
        qy=qy,
        Nk=Nk,
    )
    lambda2 = evals[1] if len(evals) > 1 else np.nan
    return {
        "mu": mu,
        "r_over_J": r_over_J,
        "lambda1": evals[0],
        "lambda2_over_lambda1": float(np.real(lambda2) / np.real(evals[0]))
        if np.real(evals[0]) != 0
        else np.nan,
        "dominant_gamma": gamma.label,
        "dominant_basis": f"tau{gamma.mu} sigma{gamma.nu}",
        "dominant_fraction": float(weights[dominant]),
        "best_harmonic": harmonic,
        "harmonic_overlap": overlap,
        "eta": gamma.eta_gap,
        "rho": gamma.rho_gauge,
        "coupling": gamma.coupling_sign,
    }


def markdown_table(title: str, rows: list[dict[str, object]]) -> str:
    lines = [
        f"## {title}",
        "",
        "| mu | r/J | leading Gamma | basis | sign (eta,rho,c) | Re(lambda1) | lambda2/lambda1 | weight | harmonic | overlap |",
        "|---:|---:|---|---|---|---:|---:|---:|---|---:|",
    ]
    for row in rows:
        sign_tuple = (
            f"({row['eta']:+d},{row['rho']:+d},{row['coupling']:+d})"
        )
        lines.append(
            "| {mu:g} | {r_over_J:g} | {dominant_gamma} | {dominant_basis} | "
            "{sign_tuple} | {lambda1:.6e} | {ratio:.4f} | {weight:.3f} | "
            "{harmonic} | {overlap:+.3f} |".format(
                mu=row["mu"],
                r_over_J=row["r_over_J"],
                dominant_gamma=row["dominant_gamma"],
                dominant_basis=row["dominant_basis"],
                sign_tuple=sign_tuple,
                lambda1=float(np.real(row["lambda1"])),
                ratio=row["lambda2_over_lambda1"],
                weight=row["dominant_fraction"],
                harmonic=row["best_harmonic"],
                overlap=row["harmonic_overlap"],
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the four full-16 Gamma benchmarks.")
    parser.add_argument("--Nk-base", type=int, default=8)
    parser.add_argument("--Nw", type=int, default=8)
    parser.add_argument("--num-eigs", type=int, default=6)
    parser.add_argument("--J", type=float, default=2.0 / 3.0)
    parser.add_argument("--T", type=float, default=0.008108)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = replace(
        ModelParams(),
        Nk_base=args.Nk_base,
        Nw=args.Nw,
        num_eigs=args.num_eigs,
        J=args.J,
        T=args.T,
    )
    configs = [
        ("attractive8 / paper blocks", "attractive8", "paper8", "row"),
        ("all16 / SSG blocks", "all16", "ssg", "row"),
    ]
    sections = [
        "# Full Gamma Pairing Benchmark",
        "",
        f"Parameters: Nk_base={params.Nk_base}, Nk={params.Nk_base + 1}, "
        f"Nw={params.Nw}, J={params.J:.8g}, T={params.T:.6g}, "
        "m=(r/J)*J, sign_side=row.",
        "",
        "Sign tuple is (eta_gap, rho_gauge, coupling=s_mu*s_nu).",
        "",
    ]
    for title, basis_set, block_mode, sign_side in configs:
        rows = []
        for mu, r_over_J in DEFAULT_POINTS:
            rows.append(
                solve_point(
                    mu=mu,
                    r_over_J=r_over_J,
                    params=params,
                    basis_set=basis_set,
                    block_mode=block_mode,
                    sign_side=sign_side,
                )
            )
        sections.append(markdown_table(title, rows))
        sections.append("")

    output = "\n".join(sections)
    print(output)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
