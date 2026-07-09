from __future__ import annotations

"""Full spin-orbital Gamma-basis pairing calculation.

This is the orbital-basis route.  The unknown is a vector over both momentum
and Gamma_a=tau_mu sigma_nu components, so different orbital/spin pairing
channels can mix inside the allowed block.  This is intentionally separate
from the scalar band-projection solver in symmetrized_lieb_gap_solver.py,
where the unknown is only Omega_nu(k).

The main objects are:

* GammaDef: definition and symmetry characters of one Gamma_a matrix.
* compute_bubbles: Matsubara bubble in the spin-orbital basis.
* build_mchi_generalized: interaction-dressed Gamma-space kernel.
* solve_eigenproblem: leading full Gamma-basis pairing modes.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np
from numpy.linalg import inv
from scipy.sparse.linalg import eigs


TAU0 = np.array([[1, 0], [0, 1]], dtype=complex)
TAU1 = np.array([[0, 1], [1, 0]], dtype=complex)
TAU2 = np.array([[0, -1j], [1j, 0]], dtype=complex)
TAU3 = np.array([[1, 0], [0, -1]], dtype=complex)
TAU = {"0": TAU0, "x": TAU1, "y": TAU2, "z": TAU3}


@dataclass(frozen=True)
class ModelParams:
    Nk_base: int = 5
    m: float = 1.0
    J: float = 0.66
    mu: float = 0.5
    tp: float = 0.2
    tpp: float = -0.3
    lam: float = 0.3
    t: float = 1.0
    T: float = 0.008108
    Nw: int = 3
    delta: float = 1e-15
    num_eigs: int = 6


@dataclass(frozen=True)
class GammaDef:
    index: int
    mu: str
    nu: str

    @property
    def label(self) -> str:
        return f"Gamma{self.index}"

    @property
    def matrix(self) -> np.ndarray:
        return np.kron(TAU[self.mu], TAU[self.nu])

    @property
    def matrix_parity(self) -> int:
        """Sign xi_a in Gamma_a.T = xi_a Gamma_a."""

        return TRANSPOSE_SIGN[self.mu] * TRANSPOSE_SIGN[self.nu]

    @property
    def eta_gap(self) -> int:
        """Momentum parity Delta_a(-k) = eta_a Delta_a(k)."""

        return -self.matrix_parity

    @property
    def rho_gauge(self) -> int:
        """Reciprocal-gauge sign under tau_z pair transformation."""

        return S_MU[self.mu]

    @property
    def coupling_sign(self) -> int:
        """Attractive/repulsive Fierz sign, invisible in the old attractive 8."""

        return S_MU[self.mu] * S_NU[self.nu]

    @property
    def pair_ssg_character(self) -> tuple[int, int, int]:
        g_tau_z = S_MU[self.mu]
        s_pi_z = -S_NU[self.nu]
        tau_x = {"0": +1, "x": +1, "y": -1, "z": -1}[self.mu]
        spin_x = {"0": +1, "x": +1, "y": -1, "z": -1}[self.nu]
        c_tau_x_s_pi_x = -tau_x * spin_x
        return (g_tau_z, s_pi_z, c_tau_x_s_pi_x)


TRANSPOSE_SIGN = {"0": +1, "x": +1, "y": -1, "z": +1}
S_MU = {"0": +1, "x": -1, "y": -1, "z": +1}
S_NU = {"0": +1, "x": -1, "y": -1, "z": +1}

GAMMA16 = (
    GammaDef(1, "0", "0"),
    GammaDef(2, "z", "z"),
    GammaDef(3, "z", "0"),
    GammaDef(4, "0", "z"),
    GammaDef(5, "x", "y"),
    GammaDef(6, "y", "x"),
    GammaDef(7, "x", "x"),
    GammaDef(8, "y", "y"),
    GammaDef(9, "y", "0"),
    GammaDef(10, "y", "z"),
    GammaDef(11, "x", "0"),
    GammaDef(12, "x", "z"),
    GammaDef(13, "0", "x"),
    GammaDef(14, "0", "y"),
    GammaDef(15, "z", "x"),
    GammaDef(16, "z", "y"),
)


def sign_text(value: int) -> str:
    return "+" if value > 0 else "-"


def make_grid(p: ModelParams):
    x = -np.pi + 2.0 * np.pi * np.arange(p.Nk_base + 1) / p.Nk_base
    y = -np.pi + 2.0 * np.pi * np.arange(p.Nk_base + 1) / p.Nk_base
    qx, qy = np.meshgrid(x, y, indexing="xy")
    Nk = p.Nk_base + 1
    vqx = qx.reshape(Nk * Nk)
    vqy = qy.reshape(Nk * Nk)
    dxy = (x[1] - x[0]) * (y[1] - y[0])
    pref = dxy * p.T / (2.0 * np.pi) ** 2
    return x, y, qx, qy, vqx, vqy, Nk, dxy, pref


def H0(kx: float, ky: float, p: ModelParams) -> np.ndarray:
    return (
        (-(p.tp + p.tpp) * (np.cos(kx) + np.cos(ky)) - p.mu) * np.kron(TAU0, TAU0)
        + (-4.0 * p.t * np.cos(kx / 2.0) * np.cos(ky / 2.0)) * np.kron(TAU1, TAU0)
        + (p.lam * np.sin(kx / 2.0) * np.sin(ky / 2.0)) * np.kron(TAU2, TAU3)
        + ((p.tp - p.tpp) * (np.cos(kx) - np.cos(ky))) * np.kron(TAU3, TAU0)
    )


def H0p(kx: float, ky: float, p: ModelParams) -> np.ndarray:
    return (
        (-(p.tp + p.tpp) * (np.cos(kx) + np.cos(ky)) - p.mu) * np.kron(TAU0, TAU0)
        + (4.0 * p.t * np.cos(kx / 2.0) * np.cos(ky / 2.0)) * np.kron(TAU1, TAU0)
        + (-p.lam * np.sin(kx / 2.0) * np.sin(ky / 2.0)) * np.kron(TAU2, TAU3)
        + ((p.tp - p.tpp) * (np.cos(kx) - np.cos(ky))) * np.kron(TAU3, TAU0)
    )


def selected_gammas(basis_set: str) -> tuple[GammaDef, ...]:
    """Select the Gamma basis used in the full orbital calculation.

    ``attractive8`` reproduces the old paper-style attractive subset.
    ``all16`` keeps all tau_mu sigma_nu matrices and is the appropriate check
    when repulsive channels are included rather than projected out by hand.
    """

    if basis_set == "attractive8":
        return GAMMA16[:8]
    if basis_set == "all16":
        return GAMMA16
    raise ValueError(f"Unknown basis set: {basis_set}")


def block_groups(gammas: tuple[GammaDef, ...], block_mode: str) -> list[list[int]]:
    """Group Gamma components into blocks that are allowed to mix.

    The full kernel is built only inside these blocks.  This keeps the symmetry
    assumption explicit: if two Gamma components are in different blocks, their
    mixing is being set to zero by the chosen symmetry classification.
    """

    if block_mode == "paper8":
        if len(gammas) != 8 or [g.index for g in gammas] != list(range(1, 9)):
            raise ValueError("block_mode='paper8' requires basis_set='attractive8'")
        return [list(range(0, 4)), list(range(4, 6)), list(range(6, 8))]

    groups: dict[object, list[int]] = defaultdict(list)
    for local_index, gamma in enumerate(gammas):
        if block_mode == "ssg":
            key = gamma.pair_ssg_character
        elif block_mode == "eta-rho":
            key = (gamma.eta_gap, gamma.rho_gauge)
        elif block_mode == "full":
            key = "full"
        else:
            raise ValueError(f"Unknown block mode: {block_mode}")
        groups[key].append(local_index)
    return list(groups.values())


def allowed_pairs(groups: Iterable[Iterable[int]]) -> list[tuple[int, int]]:
    pairs = []
    for group in groups:
        group = list(group)
        pairs.extend((a, b) for a in group for b in group)
    return pairs


def build_hamiltonian_tables(vqx: np.ndarray, vqy: np.ndarray, p: ModelParams):
    H_table = np.zeros((len(vqx), 4, 4), dtype=complex)
    Hp_table = np.zeros((len(vqx), 4, 4), dtype=complex)
    for ik, (kx, ky) in enumerate(zip(vqx, vqy)):
        H_table[ik] = H0(float(kx), float(ky), p)
        Hp_table[ik] = H0p(float(kx), float(ky), p)
    return H_table, Hp_table


def compute_bubbles(
    vqx: np.ndarray,
    vqy: np.ndarray,
    gammas: tuple[GammaDef, ...],
    pairs: list[tuple[int, int]],
    p: ModelParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Gamma-space particle-particle bubbles on the momentum grid.

    ``bubble`` uses H0(k), while ``bubblep`` uses the reciprocal-gauge related
    H0p(k).  The final kernel combines both pieces so the full Gamma-basis
    calculation can be compared with the reciprocal-gauge symmetrized
    projected solver.
    """

    n_gamma = len(gammas)
    H_table, Hp_table = build_hamiltonian_tables(vqx, vqy, p)
    gamma_mats = [gamma.matrix for gamma in gammas]
    bubble = np.zeros((len(vqx), n_gamma, n_gamma), dtype=complex)
    bubblep = np.zeros_like(bubble)
    eye4 = np.eye(4, dtype=complex)

    for ik in range(len(vqx)):
        Hk = H_table[ik]
        Hkp = Hp_table[ik]
        for n in range(-p.Nw - 1, p.Nw + 1):
            freq = (2 * n + 1) * np.pi * p.T
            iwn = 1j * freq * eye4

            Gp = inv(iwn - Hk)
            Gh = inv(iwn + Hk.T)
            Gpp = inv(iwn - Hkp)
            Ghp = inv(iwn + Hkp.T)

            for a, b in pairs:
                G1 = gamma_mats[a]
                G2 = gamma_mats[b]
                bubble[ik, a, b] += np.trace(Gp @ G1 @ Gh @ G2)
                bubblep[ik, a, b] += np.trace(Gpp @ G1 @ Ghp @ G2)

    return bubble, bubblep


def chi_piece(qx: np.ndarray, qy: np.ndarray, p: ModelParams, sign_of_J: int) -> np.ndarray:
    q2 = qx * qx + qy * qy
    denom = p.delta * q2 + p.m + sign_of_J * p.J * np.cos(qx / 2.0) * np.cos(qy / 2.0)
    return 0.25 / denom


def interaction_matrices(vqx: np.ndarray, vqy: np.ndarray, p: ModelParams):
    px = vqx[:, None]
    py = vqy[:, None]
    kx = vqx[None, :]
    ky = vqy[None, :]
    minus_pm = chi_piece(px - kx, py - ky, p, sign_of_J=-1)
    minus_pp = chi_piece(px + kx, py + ky, p, sign_of_J=-1)
    plus_pm = chi_piece(px - kx, py - ky, p, sign_of_J=+1)
    plus_pp = chi_piece(px + kx, py + ky, p, sign_of_J=+1)
    return minus_pm, minus_pp, plus_pm, plus_pp


def coupling_for_pair(gammas: tuple[GammaDef, ...], a: int, b: int, sign_side: str) -> int:
    if sign_side == "row":
        return gammas[a].coupling_sign
    if sign_side == "column":
        return gammas[b].coupling_sign
    if sign_side == "none":
        return +1
    raise ValueError(f"Unknown sign side: {sign_side}")


def build_mchi_generalized(
    vqx: np.ndarray,
    vqy: np.ndarray,
    gammas: tuple[GammaDef, ...],
    pairs: list[tuple[int, int]],
    bubble: np.ndarray,
    bubblep: np.ndarray,
    p: ModelParams,
    *,
    sign_side: str,
) -> np.ndarray:
    n_tot = len(vqx)
    n_gamma = len(gammas)
    Mchi = np.zeros((n_gamma * n_tot, n_gamma * n_tot), dtype=complex)
    minus_pm, minus_pp, plus_pm, plus_pp = interaction_matrices(vqx, vqy, p)

    for a, b in pairs:
        row0 = a * n_tot
        col0 = b * n_tot
        eta_b = gammas[b].eta_gap
        rho_b = gammas[b].rho_gauge
        coupling = coupling_for_pair(gammas, a, b, sign_side)
        chi_minus = minus_pm + eta_b * minus_pp
        chi_plus = plus_pm + eta_b * plus_pp
        block = coupling * (
            bubble[:, a, b][None, :] * chi_minus
            + rho_b * bubblep[:, a, b][None, :] * chi_plus
        )
        Mchi[row0 : row0 + n_tot, col0 : col0 + n_tot] = block

    return Mchi


def build_mchi_legacy_paper8(
    vqx: np.ndarray,
    vqy: np.ndarray,
    bubble: np.ndarray,
    bubblep: np.ndarray,
    p: ModelParams,
) -> np.ndarray:
    n_tot = len(vqx)
    Mchi = np.zeros((8 * n_tot, 8 * n_tot), dtype=complex)
    minus_pm, minus_pp, plus_pm, plus_pp = interaction_matrices(vqx, vqy, p)

    for a in range(8):
        if a < 4:
            b_range = range(0, 4)
            eta = -1
            rho = +1
        elif a < 6:
            b_range = range(4, 6)
            eta = +1
            rho = -1
        else:
            b_range = range(6, 8)
            eta = -1
            rho = -1
        row0 = a * n_tot
        for b in b_range:
            col0 = b * n_tot
            block = bubble[:, a, b][None, :] * (minus_pm + eta * minus_pp)
            block += rho * bubblep[:, a, b][None, :] * (plus_pm + eta * plus_pp)
            Mchi[row0 : row0 + n_tot, col0 : col0 + n_tot] = block

    return Mchi


def solve_eigenproblem(Mchi: np.ndarray, pref: float, num_eigs: int):
    A = -pref * Mchi
    dim = A.shape[0]
    if dim <= max(256, num_eigs + 2):
        evals, evecs = np.linalg.eig(A)
    else:
        evals, evecs = eigs(A, k=num_eigs, which="LR")
    order = np.argsort(evals.real)[::-1]
    return evals[order][:num_eigs], evecs[:, order][:, :num_eigs]


def channel_weights(vec: np.ndarray, n_gamma: int, n_tot: int) -> np.ndarray:
    components = vec.reshape(n_gamma, n_tot)
    weights = np.sum(np.abs(components), axis=1)
    total = np.sum(weights)
    return weights / total if total > 0 else weights


def print_gamma_table(gammas: tuple[GammaDef, ...]) -> None:
    print("Gamma sign table")
    print("idx  mu nu  eta_gap rho_gauge coupling pair_ssg")
    for gamma in gammas:
        chars = "".join(sign_text(v) for v in gamma.pair_ssg_character)
        print(
            f"{gamma.index:>2}   {gamma.mu:>1}  {gamma.nu:>1}     "
            f"{sign_text(gamma.eta_gap):>1}        {sign_text(gamma.rho_gauge):>1}        "
            f"{sign_text(gamma.coupling_sign):>1}      {chars}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Experimental full Gamma1-16 multichannel pairing solver."
    )
    parser.add_argument("--basis-set", choices=["attractive8", "all16"], default="all16")
    parser.add_argument(
        "--block-mode",
        choices=["auto", "paper8", "ssg", "eta-rho", "full"],
        default="auto",
        help="auto uses paper8 for attractive8 and ssg for all16.",
    )
    parser.add_argument(
        "--sign-side",
        choices=["row", "column", "none"],
        default="row",
        help="Where to apply the s_mu*s_nu coupling sign.",
    )
    parser.add_argument("--Nk-base", type=int, default=5)
    parser.add_argument("--Nw", type=int, default=3)
    parser.add_argument("--num-eigs", type=int, default=6)
    parser.add_argument("--mu", type=float, default=0.5)
    parser.add_argument("--m", type=float, default=1.0)
    parser.add_argument("--J", type=float, default=0.66)
    parser.add_argument("--T", type=float, default=0.008108)
    parser.add_argument("--check-paper8", action="store_true")
    parser.add_argument("--print-table", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p = replace(
        ModelParams(),
        Nk_base=args.Nk_base,
        Nw=args.Nw,
        num_eigs=args.num_eigs,
        mu=args.mu,
        m=args.m,
        J=args.J,
        T=args.T,
    )
    gammas = selected_gammas(args.basis_set)
    block_mode = args.block_mode
    if block_mode == "auto":
        block_mode = "paper8" if args.basis_set == "attractive8" else "ssg"
    groups = block_groups(gammas, block_mode)
    pairs = allowed_pairs(groups)
    _, _, _, _, vqx, vqy, Nk, _, pref = make_grid(p)

    if args.print_table:
        print_gamma_table(gammas)
        print()

    print(
        "Running full Gamma solver: "
        f"basis_set={args.basis_set}, block_mode={block_mode}, sign_side={args.sign_side}, "
        f"Nk={Nk}, Ntot={len(vqx)}, Nw={p.Nw}"
    )
    print("Block groups:", [[gammas[i].label for i in group] for group in groups])

    bubble, bubblep = compute_bubbles(vqx, vqy, gammas, pairs, p)
    Mchi = build_mchi_generalized(
        vqx,
        vqy,
        gammas,
        pairs,
        bubble,
        bubblep,
        p,
        sign_side=args.sign_side,
    )

    if args.check_paper8:
        if args.basis_set != "attractive8" or block_mode != "paper8":
            raise ValueError("--check-paper8 requires --basis-set attractive8 --block-mode paper8")
        legacy = build_mchi_legacy_paper8(vqx, vqy, bubble, bubblep, p)
        max_abs = float(np.max(np.abs(Mchi - legacy)))
        print(f"paper8 regression max_abs(M_generalized - M_legacy) = {max_abs:.3e}")

    evals, evecs = solve_eigenproblem(Mchi, pref, p.num_eigs)
    print("Leading eigenvalues")
    for n, val in enumerate(evals, start=1):
        print(f"{n:>2}: Re={val.real:+.8e}  Im={val.imag:+.3e}")

    weights = channel_weights(evecs[:, 0], len(gammas), len(vqx))
    order = np.argsort(weights)[::-1]
    print("Leading-mode component weights")
    for idx in order[: min(12, len(order))]:
        gamma = gammas[int(idx)]
        print(
            f"{gamma.label:>7} tau{gamma.mu} sigma{gamma.nu}: "
            f"{weights[idx]:.4f}  eta={sign_text(gamma.eta_gap)} "
            f"rho={sign_text(gamma.rho_gauge)} coupling={sign_text(gamma.coupling_sign)}"
        )


if __name__ == "__main__":
    main()
