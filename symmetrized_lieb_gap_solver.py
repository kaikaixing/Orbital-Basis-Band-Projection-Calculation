from __future__ import annotations

"""Single-band projected Lieb-lattice gap-equation solver.

This is the band-projection route.  The unknown solved here is the scalar
single-band gap function Omega_nu(k) in one fixed spin channel nu.  Orbital
information enters only through projected form factors
g_mu(k)=u_n^T(-k) tau_mu u_n(k).  Gamma-like labels can be assigned later as
a bookkeeping diagnostic, but this file does not compute the full orbital HS
eigenvector Delta_{mu,nu}(k).

Use this file when checking the reduced projected equation

    Omega_nu(p) = sum_k K_nu^sym(p,k) W(k) Omega_nu(k).

For the full spin-orbital Gamma-basis route, see
benchmarks/full_16_gamma_pairing_solver.py.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg


Array = np.ndarray
Susceptibility = Callable[[Array, Array, "GapEquationParams"], Array]


@dataclass
class GapEquationParams:
    # Momentum grid: kx, ky in [-pi, pi), with Nk^2 total points.
    Nk: int = 25

    # Default model follows lieb_routei_framework/lied_gap_core.py: an effective
    # two-orbital projected Lieb Hamiltonian
    #
    #   H_eff(k) = h0(k) tau0 + h1(k) taux + h3(k) tauz
    #
    # with h0 containing the chemical potential.  Set orbital_model="three_band"
    # to use the simple 3-band demonstration Hamiltonian below instead.
    orbital_model: str = "projected_two_orbital"
    tp_a: float = -0.3
    tp_b: float = 0.2
    orbital_eta: float = 1e-12
    band_sign: int = 1  # +1 upper effective band, -1 lower effective band.

    # Optional simple real, inversion-symmetric 3-band Lieb Hamiltonian:
    # H_AB = -2 t cos(kx/2), H_AC = -2 t cos(ky/2), H_BC = 0.
    # Onsite offsets are optional knobs for testing; keep them zero for the
    # simplest nearest-neighbor Lieb lattice.
    t: float = 1.0
    eps_a: float = 0.0
    eps_b: float = 0.0
    eps_c: float = 0.0

    # For orbital_model="projected_two_orbital", mu_F is the same parameter
    # called p.mu in lieb_routei_framework; xi_n(k) is directly the selected
    # eigenvalue of H_eff(k).  For orbital_model="three_band", xi_n=E_n-mu_F.
    mu_F: float = 2.5
    T: float = 0.05
    g: float = 1.0
    band_index: int = 0  # Only used by orbital_model="three_band".

    # Susceptibility parameters.  Available built-ins:
    #   "lattice":    chi0 / [r - J cos(qx/2) cos(qy/2)]
    #   "continuum": chi0 / [r + qx^2 + qy^2]
    #   "paper56":   framework chi56 matrix replacing the whole
    #                [chi(p-k)+eta_nu chi(p+k)] bracket
    #   "paper78":   framework chi78 matrix
    #   "paper1234": framework chi1234 matrix
    #   "projected_formula": use the final reciprocal-lattice symmetrized,
    #                mu-resolved equation from the note
    #       g^2/32 * sum_mu s_mu s_nu g_mu^*(p)g_mu(k)
    #       * [chi_pk(-J)+s_mu chi_pk(J)
    #          +eta chi_p,-k(-J)+eta s_mu chi_p,-k(J)]
    # A custom callable can also be passed to solve_all_channels(...).
    chi0: float = 1.0
    r: float = 1.0
    J: float = 1.0 / 1.5
    susceptibility_type: str = "projected_formula"
    wrap_susceptibility_momentum: bool = False
    chi_denom_tol: float = 1e-12
    # The PDF's final K_nu formula uses chi(q)=chi0/[r +/- J cos(qx/2)cos(qy/2)].
    # Set this to 0.25 to reproduce the older framework chi-piece convention.
    paper_chi_prefactor: float = 1.0
    chi_mixing_sign: float = -1.0

    # Numerical controls.
    xi_tol: float = 1e-10
    # Overall sign convention for the linearized gap kernel.  With the
    # PDF/Route-I chi(-J)+s_mu chi(+J) ordering below, +1 reproduces the
    # paper-sector kernels: Gamma1-4 -> chi1234 and Gamma5/7 -> chi56/chi78.
    kernel_sign: float = 1.0
    dense_eig_max_dim: int = 900
    sparse_num_eigs: int = 1
    sparse_which: str = "LR"
    output_dir: str = "symmetrized_lieb_gap_outputs"
    make_plots: bool = True
    overlap_weight_mode: str = "pair_factor"
    harmonic_basis_set: str = "paper"
    fs_cutoff: float = 0.2
    # "all": keep the full signed sum sum_mu s_mu s_nu ...
    # "positive_product": keep only mu terms with s_mu * s_nu = +1.
    # The latter is the attractive-only comparison used in the framework-style
    # diagnostics.
    mu_selection: str = "all"
    # Optional component-resolved projection.  For example, Gamma5=tau_x sigma_y
    # and Gamma7=tau_x sigma_x should be compared with active_mu_indices=(1,),
    # i.e. only the g_x(k) form factor, rather than the spin-channel sum over
    # all orbital mu.
    active_mu_indices: tuple[int, ...] | None = None
    enforce_parity_subspace: bool = True

    # External pairing-channel signs are convention dependent.  Override these
    # for the Fierz convention used in the derivation.  The internal s_mu signs
    # are not set here; for the default Pauli basis they are derived from
    # tau_z tau_mu tau_z = s_mu tau_mu.
    s_nu: dict[str, float] = field(
        default_factory=lambda: {"0": 1.0, "x": -1.0, "y": -1.0, "z": 1.0}
    )


@dataclass
class BandData:
    kx: Array
    ky: Array
    kx_grid: Array
    ky_grid: Array
    dA: float
    energies: Array
    xi: Array
    eigenvectors: Array
    W: Array


@dataclass
class ChannelResult:
    channel: str
    eta: int
    eigenvalue: complex
    eigenvector: Array
    parity_error: float
    dominant_structure: str
    plot_path: Path | None


CHANNELS = ("0", "x", "y", "z")

# From fermionic antisymmetry of d^T_{-k} sigma_nu d_k.  In the convention used
# here sigma_y is antisymmetric in the internal indices, so its orbital/momentum
# gap is even.  sigma_0, sigma_x, and sigma_z are symmetric, so their
# momentum-space gaps must be odd.
ETA_NU = {"0": -1, "x": -1, "y": +1, "z": -1}


def wrap_bz(q: Array) -> Array:
    return (q + np.pi) % (2.0 * np.pi) - np.pi


def make_momentum_grid(params: GapEquationParams) -> tuple[Array, Array, Array, Array, float]:
    k = np.linspace(-np.pi, np.pi, params.Nk, endpoint=False)
    kx_grid, ky_grid = np.meshgrid(k, k, indexing="xy")
    kx = kx_grid.reshape(-1)
    ky = ky_grid.reshape(-1)
    dk = 2.0 * np.pi / params.Nk
    dA = dk * dk / (2.0 * np.pi) ** 2
    return kx, ky, kx_grid, ky_grid, dA


def lieb_hamiltonian(kx: float, ky: float, params: GapEquationParams) -> Array:
    """Real inversion-symmetric Lieb-lattice Bloch Hamiltonian.

    The half-bond cosine form makes H0(-k)=H0(k).  Since this matrix is real
    symmetric, numpy.linalg.eigh returns eigenvectors that can be chosen real.
    """

    ax = -2.0 * params.t * np.cos(0.5 * kx)
    ay = -2.0 * params.t * np.cos(0.5 * ky)
    return np.array(
        [
            [params.eps_a, ax, ay],
            [ax, params.eps_b, 0.0],
            [ay, 0.0, params.eps_c],
        ],
        dtype=float,
    )


def projected_orbital_components(kx: Array, ky: Array, params: GapEquationParams) -> tuple[Array, Array, Array]:
    """Two-orbital effective Lieb model used in lieb_routei_framework.

    This mirrors lieb_routei_framework.lieb_gap_core.orbital_components:

        h0 = -(tp_a + tp_b)(cos kx + cos ky) - mu
        h1 = -4 t cos(kx/2) cos(ky/2)
        h3 = -(tp_a - tp_b)(cos kx - cos ky)

    The framework names the chemical-potential parameter `mu`; here it is
    `mu_F` to match the gap-equation notation.
    """

    kx, ky = np.broadcast_arrays(kx, ky)
    h0 = -(params.tp_a + params.tp_b) * (np.cos(kx) + np.cos(ky)) - params.mu_F
    h1 = -4.0 * params.t * np.cos(0.5 * kx) * np.cos(0.5 * ky)
    h3 = -(params.tp_a - params.tp_b) * (np.cos(kx) - np.cos(ky))
    return h0, h1, h3


def projected_lieb_hamiltonian(kx: float, ky: float, params: GapEquationParams) -> Array:
    h0, h1, h3 = projected_orbital_components(np.asarray(kx), np.asarray(ky), params)
    return np.array(
        [
            [float(h0 + h3), float(h1)],
            [float(h1), float(h0 - h3)],
        ],
        dtype=float,
    )


def diagonalize_lieb_band(params: GapEquationParams) -> BandData:
    kx, ky, kx_grid, ky_grid, dA = make_momentum_grid(params)
    n_points = kx.size
    energies = np.empty(n_points, dtype=float)

    if params.orbital_model == "projected_two_orbital":
        eigenvectors = np.empty((n_points, 2), dtype=float)
        if params.band_sign not in (-1, +1):
            raise ValueError("band_sign must be +1 or -1 for projected_two_orbital")
        selected_index = 1 if params.band_sign > 0 else 0
        hamiltonian = projected_lieb_hamiltonian
    elif params.orbital_model == "three_band":
        eigenvectors = np.empty((n_points, 3), dtype=float)
        if not 0 <= params.band_index < 3:
            raise ValueError("band_index must be 0, 1, or 2 for the three_band model")
        selected_index = params.band_index
        hamiltonian = lieb_hamiltonian
    else:
        raise ValueError("orbital_model must be 'projected_two_orbital' or 'three_band'")

    for idx, (kx_i, ky_i) in enumerate(zip(kx, ky)):
        vals, vecs = np.linalg.eigh(hamiltonian(float(kx_i), float(ky_i), params))
        u = vecs[:, selected_index]

        # Local real gauge: fix the largest component to be positive.  This is
        # enough for g_mu(k)=u^T tau_mu u because the bilinear is invariant under
        # u -> -u.  The inversion-symmetric gauge assumption H(-k)=H(k) implies
        # u(-k)=u(k) can be imposed away from degeneracies.
        anchor = int(np.argmax(np.abs(u)))
        if u[anchor] < 0.0:
            u = -u
        energies[idx] = vals[selected_index]
        eigenvectors[idx] = u

    # The derivation uses an inversion-symmetric band gauge, u_n(-k)=u_n(k).
    # H(-k)=H(k) makes this gauge available, but eigh chooses the overall sign
    # of each real eigenvector independently at each grid point.  Enforce the
    # inversion-pair relation explicitly before constructing g_mu(k).
    inv = inversion_indices(params.Nk)
    seen = np.zeros(n_points, dtype=bool)
    for idx in range(n_points):
        if seen[idx]:
            continue
        partner = int(inv[idx])
        if partner != idx:
            eigenvectors[partner] = eigenvectors[idx]
        seen[idx] = True
        seen[partner] = True

    if params.orbital_model == "projected_two_orbital":
        # The framework's H_eff already includes -mu_F in h0, so the selected
        # eigenvalue is xi_n(k), not a bare band energy requiring another
        # subtraction.
        xi = energies.copy()
    else:
        xi = energies - params.mu_F
    W = pair_factor(xi, params)
    return BandData(kx, ky, kx_grid, ky_grid, dA, energies, xi, eigenvectors, W)


def pair_factor(xi: Array, params: GapEquationParams) -> Array:
    """W(k)=tanh[xi_n(k)/(2T)]/[2 xi_n(k)], with the xi -> 0 limit."""

    out = np.empty_like(xi, dtype=float)
    small = np.abs(xi) < params.xi_tol
    out[small] = 1.0 / (4.0 * params.T)
    out[~small] = np.tanh(xi[~small] / (2.0 * params.T)) / (2.0 * xi[~small])
    return out


def complete_hermitian_basis_3() -> tuple[list[Array], list[str]]:
    """A simple complete Hermitian basis for 3x3 orbital matrices."""

    mats: list[Array] = []
    labels: list[str] = []

    mats.append(np.eye(3, dtype=complex))
    labels.append("I")

    for i, j, label in [(0, 1, "AB"), (0, 2, "AC"), (1, 2, "BC")]:
        sym = np.zeros((3, 3), dtype=complex)
        sym[i, j] = sym[j, i] = 1.0
        mats.append(sym)
        labels.append(f"S_{label}")

        asym = np.zeros((3, 3), dtype=complex)
        asym[i, j] = -1.0j
        asym[j, i] = +1.0j
        mats.append(asym)
        labels.append(f"A_{label}")

    mats.append(np.diag([1.0, -1.0, 0.0]).astype(complex))
    labels.append("D_1")
    mats.append((np.diag([1.0, 1.0, -2.0]) / np.sqrt(3.0)).astype(complex))
    labels.append("D_2")
    return mats, labels


def pauli_basis_2() -> tuple[list[Array], list[str]]:
    return (
        [
            np.array([[1, 0], [0, 1]], dtype=complex),
            np.array([[0, 1], [1, 0]], dtype=complex),
            np.array([[0, -1j], [1j, 0]], dtype=complex),
            np.array([[1, 0], [0, -1]], dtype=complex),
        ],
        ["tau0", "taux", "tauy", "tauz"],
    )


def compute_form_factors_on_grid(
    eigenvectors: Array,
    tau_matrices: Iterable[Array],
    Nk: int,
) -> Array:
    """Return g[mu, k] = u_n^T(-k) tau_mu u_n(k).

    This uses transpose, not Hermitian conjugation, matching the projected
    pairing form factor.  In the real inversion-symmetric gauge H(-k)=H(k),
    one may impose u(-k)=u(k), reducing this to u^T(k) tau_mu u(k).  The code
    still indexes the -k vector explicitly so the mathematical definition is
    visible and easy to audit.
    """

    tau = [np.asarray(mat, dtype=complex) for mat in tau_matrices]
    if not tau:
        raise ValueError("At least one tau matrix is required")
    dim = eigenvectors.shape[1]
    for idx, mat in enumerate(tau):
        if mat.shape != (dim, dim):
            raise ValueError(
                f"tau_matrices[{idx}] has shape {mat.shape}; expected {(dim, dim)}"
            )

    inv = inversion_indices(Nk)
    u_minus_k = eigenvectors[inv]
    u_k = eigenvectors

    g = np.empty((len(tau), eigenvectors.shape[0]), dtype=complex)
    for mu, mat in enumerate(tau):
        g[mu] = np.einsum("ki,ij,kj->k", u_minus_k, mat, u_k, optimize=True)
    return g


def infer_s_mu_from_tau_z(
    tau_matrices: Iterable[Array],
    tau_z: Array,
    tol: float = 1e-10,
) -> Array:
    """Infer s_mu from tau_z tau_mu tau_z = s_mu tau_mu.

    For the Pauli basis (tau0, taux, tauy, tauz), this gives
    s_mu = (+1, -1, -1, +1).
    """

    tau_z = np.asarray(tau_z, dtype=complex)
    signs = []
    for idx, tau_mu in enumerate(tau_matrices):
        tau_mu = np.asarray(tau_mu, dtype=complex)
        transformed = tau_z @ tau_mu @ tau_z
        plus_error = np.linalg.norm(transformed - tau_mu)
        minus_error = np.linalg.norm(transformed + tau_mu)
        scale = max(float(np.linalg.norm(tau_mu)), 1.0)
        if plus_error <= tol * scale:
            signs.append(+1.0)
        elif minus_error <= tol * scale:
            signs.append(-1.0)
        else:
            raise ValueError(
                "Cannot infer s_mu for tau_matrices[{}]: "
                "tau_z tau_mu tau_z is neither +tau_mu nor -tau_mu".format(idx)
            )
    return np.asarray(signs, dtype=float)


def susceptibility(qx: Array, qy: Array, params: GapEquationParams) -> Array:
    if params.wrap_susceptibility_momentum:
        qx = wrap_bz(qx)
        qy = wrap_bz(qy)

    if params.susceptibility_type == "continuum":
        denom = params.r + qx * qx + qy * qy
    elif params.susceptibility_type == "lattice":
        denom = params.r - params.J * np.cos(0.5 * qx) * np.cos(0.5 * qy)
    else:
        raise ValueError("susceptibility_type must be 'lattice' or 'continuum'")

    if np.any(np.abs(denom) < params.chi_denom_tol):
        # Keep the formula unchanged; this warning points out a parameter choice
        # that makes the model susceptibility singular on the sampled grid.
        print(
            "Warning: susceptibility denominator is near zero on the grid. "
            "Consider increasing r or changing J."
        )
    return params.chi0 / denom


def paper_chi_piece(qx: Array, qy: Array, params: GapEquationParams, sign_of_J: int) -> Array:
    if params.wrap_susceptibility_momentum:
        qx = wrap_bz(qx)
        qy = wrap_bz(qy)
    denom = params.r + sign_of_J * params.J * np.cos(0.5 * qx) * np.cos(0.5 * qy)
    if np.any(np.abs(denom) < params.chi_denom_tol):
        print(
            "Warning: paper susceptibility denominator is near zero on the grid. "
            "Consider increasing r or changing J."
        )
    return params.paper_chi_prefactor * params.chi0 / denom


def paper_susceptibility_matrix(kind: str, band: BandData, params: GapEquationParams) -> Array:
    """Framework paper chi matrices as functions of the external pair (p,k).

    These replace the entire Cooper bracket [chi(p-k)+eta_nu chi(p+k)] rather
    than being passed through the simple eta symmetrizer again.
    """

    px = band.kx[:, None]
    py = band.ky[:, None]
    kx = band.kx[None, :]
    ky = band.ky[None, :]

    minus_pm = paper_chi_piece(px - kx, py - ky, params, sign_of_J=-1)
    minus_pp = paper_chi_piece(px + kx, py + ky, params, sign_of_J=-1)
    plus_pm = paper_chi_piece(px - kx, py - ky, params, sign_of_J=+1)
    plus_pp = paper_chi_piece(px + kx, py + ky, params, sign_of_J=+1)

    if kind == "paper56":
        return (minus_pm + minus_pp) - (plus_pm + plus_pp)
    if kind == "paper78":
        return (minus_pm - minus_pp) - (plus_pm - plus_pp)
    if kind == "paper1234":
        return (minus_pm - minus_pp) + (plus_pm - plus_pp)
    raise ValueError("paper susceptibility kind must be 'paper56', 'paper78', or 'paper1234'")


def projected_formula_chi_pieces(band: BandData, params: GapEquationParams) -> tuple[Array, Array, Array, Array]:
    """Return chi_{p,k}(J), chi_{p,k}(-J), chi_{p,-k}(J), chi_{p,-k}(-J)."""

    px = band.kx[:, None]
    py = band.ky[:, None]
    kx = band.kx[None, :]
    ky = band.ky[None, :]

    chi_pk_J = paper_chi_piece(px - kx, py - ky, params, sign_of_J=+1)
    chi_pk_mJ = paper_chi_piece(px - kx, py - ky, params, sign_of_J=-1)
    chi_pmk_J = paper_chi_piece(px + kx, py + ky, params, sign_of_J=+1)
    chi_pmk_mJ = paper_chi_piece(px + kx, py + ky, params, sign_of_J=-1)
    return chi_pk_J, chi_pk_mJ, chi_pmk_J, chi_pmk_mJ


def build_kernel_for_channel(
    channel: str,
    params: GapEquationParams,
    band: BandData,
    g_mu: Array,
    s_mu: Array,
    chi_func: Susceptibility | None = None,
) -> Array:
    """Construct K^nu_{p,k} for one fixed external channel nu.

    Very important: nu is fixed here and is not summed over.  The only internal
    sum is over orbital/fluctuation labels mu:

        sum_mu s_mu s_nu g_mu^*(p) g_mu(k).

    The bracket chi(p-k)+eta_nu chi(p+k) is the parity-projected Cooper kernel.
    It is the direct implementation of 1/2 [K(p,k)+eta_nu K(p,-k)].  If the
    unsymmetrized convention carries prefactor g^2/8, this projection produces
    the g^2/16 prefactor used below; we do not add any extra 1/2 beyond the
    boxed formula.
    """

    if channel not in ETA_NU:
        raise ValueError(f"Unknown channel {channel!r}; expected one of {CHANNELS}")
    if s_mu.shape != (g_mu.shape[0],):
        raise ValueError("s_mu must have one entry per tau_mu matrix")
    chi = susceptibility if chi_func is None else chi_func
    if params.active_mu_indices is None:
        mu_indices = tuple(range(g_mu.shape[0]))
    else:
        mu_indices = tuple(int(mu) for mu in params.active_mu_indices)
        bad = [mu for mu in mu_indices if mu < 0 or mu >= g_mu.shape[0]]
        if bad:
            raise ValueError(f"active_mu_indices contains invalid entries: {bad}")

    eta = ETA_NU[channel]
    s_channel = params.s_nu.get(channel)
    if s_channel is None:
        raise ValueError(f"Missing s_nu sign for channel {channel!r}")

    if params.susceptibility_type == "projected_formula":
        if params.orbital_model != "projected_two_orbital" or g_mu.shape[0] < 4:
            raise ValueError("projected_formula requires the projected two-orbital Pauli basis")
        chi_pk_J, chi_pk_mJ, chi_pmk_J, chi_pmk_mJ = projected_formula_chi_pieces(
            band, params
        )
        kernel_sum = np.zeros((band.kx.size, band.kx.size), dtype=complex)
        for mu in mu_indices:
            if params.mu_selection == "positive_product" and s_mu[mu] * s_channel <= 0:
                continue
            if params.mu_selection != "all" and params.mu_selection != "positive_product":
                raise ValueError("mu_selection must be 'all' or 'positive_product'")
            smu = s_mu[mu]
            form_mu = s_channel * smu * np.outer(
                np.conjugate(g_mu[mu]), g_mu[mu]
            )
            chi_mu = (
                chi_pk_mJ
                + smu * chi_pk_J
                + eta * chi_pmk_mJ
                + eta * smu * chi_pmk_J
            )
            kernel_sum += form_mu * chi_mu
        prefactor = params.g * params.g / 32.0
        return params.kernel_sign * prefactor * kernel_sum * band.W[None, :] * band.dA

    px = band.kx[:, None]
    py = band.ky[:, None]
    kx = band.kx[None, :]
    ky = band.ky[None, :]

    if chi_func is None and params.susceptibility_type in ("paper56", "paper78", "paper1234"):
        chi_sym = paper_susceptibility_matrix(params.susceptibility_type, band, params)
    else:
        chi_pm = chi(px - kx, py - ky, params)
        chi_pp = chi(px + kx, py + ky, params)
        chi_sym = chi_pm + eta * chi_pp

    vertex_sum = np.zeros((band.kx.size, band.kx.size), dtype=complex)
    for mu in mu_indices:
        if params.mu_selection == "positive_product" and s_mu[mu] * s_channel <= 0:
            continue
        if params.mu_selection != "all" and params.mu_selection != "positive_product":
            raise ValueError("mu_selection must be 'all' or 'positive_product'")
        # This is the only internal summation in the fixed-nu gap equation:
        #   sum_mu s_mu g_mu^*(p) g_mu(k).
        # nu is external and is never summed here.
        vertex_sum += s_mu[mu] * np.outer(np.conjugate(g_mu[mu]), g_mu[mu])

    prefactor = params.g * params.g / 16.0
    return params.kernel_sign * prefactor * s_channel * vertex_sum * band.W[None, :] * chi_sym * band.dA


def parity_basis_matrix(Nk: int, eta: int) -> Array:
    inv = inversion_indices(Nk)
    n = Nk * Nk
    seen = np.zeros(n, dtype=bool)
    columns = []
    for idx in range(n):
        if seen[idx]:
            continue
        partner = int(inv[idx])
        if partner == idx:
            seen[idx] = True
            if eta == +1:
                col = np.zeros(n, dtype=complex)
                col[idx] = 1.0
                columns.append(col)
            continue
        col = np.zeros(n, dtype=complex)
        col[idx] = 1.0 / np.sqrt(2.0)
        col[partner] = eta / np.sqrt(2.0)
        columns.append(col)
        seen[idx] = True
        seen[partner] = True
    return np.stack(columns, axis=1)


def solve_eigenproblem(K: Array, params: GapEquationParams) -> tuple[complex, Array]:
    dim = K.shape[0]
    if dim <= params.dense_eig_max_dim:
        vals, vecs = scipy.linalg.eig(K)
    else:
        k = min(params.sparse_num_eigs, dim - 2)
        vals, vecs = scipy.sparse.linalg.eigs(K, k=k, which=params.sparse_which)

    order = np.argsort(np.real(vals))[::-1]
    lead = int(order[0])
    return vals[lead], vecs[:, lead]


def solve_channel_eigenproblem(
    K: Array,
    params: GapEquationParams,
    eta: int,
) -> tuple[complex, Array]:
    if not params.enforce_parity_subspace:
        return solve_eigenproblem(K, params)

    Q = parity_basis_matrix(params.Nk, eta)
    K_reduced = Q.conj().T @ K @ Q
    val, vec_reduced = solve_eigenproblem(K_reduced, params)
    return val, Q @ vec_reduced


def inversion_indices(Nk: int) -> Array:
    iy, ix = np.indices((Nk, Nk))
    return (((-iy) % Nk) * Nk + ((-ix) % Nk)).reshape(-1)


def parity_error(vec: Array, Nk: int, eta: int) -> float:
    inv = inversion_indices(Nk)
    denom = float(np.linalg.norm(vec))
    if denom < 1e-30:
        return 0.0
    return float(np.linalg.norm(vec[inv] - eta * vec) / denom)


def align_phase(vec: Array) -> Array:
    idx = int(np.argmax(np.abs(vec)))
    if abs(vec[idx]) < 1e-14:
        return vec
    return vec * np.exp(-1.0j * np.angle(vec[idx]))


def gap_grid(vec: Array, Nk: int) -> Array:
    return np.real(align_phase(vec)).reshape(Nk, Nk)


def harmonic_basis(params: GapEquationParams, band: BandData) -> list[tuple[str, Array]]:
    kx = band.kx_grid
    ky = band.ky_grid
    bases = [
        ("1", np.ones_like(kx)),
        ("cos kx + cos ky", np.cos(kx) + np.cos(ky)),
        ("cos kx - cos ky", np.cos(kx) - np.cos(ky)),
        ("sin kx sin ky", np.sin(kx) * np.sin(ky)),
        ("s prime ~ cos(kx/2) cos(ky/2)", np.cos(0.5 * kx) * np.cos(0.5 * ky)),
        ("d prime ~ sin(kx/2) sin(ky/2)", np.sin(0.5 * kx) * np.sin(0.5 * ky)),
        ("sin(kx/2) cos(ky/2)", np.sin(0.5 * kx) * np.cos(0.5 * ky)),
        ("sin(ky/2) cos(kx/2)", np.sin(0.5 * ky) * np.cos(0.5 * kx)),
        ("sin kx + sin ky", np.sin(kx) + np.sin(ky)),
        ("sin kx - sin ky", np.sin(kx) - np.sin(ky)),
        ("p wave f1 ~ sin kx (1 - cos ky)", np.sin(kx) * (1.0 - np.cos(ky))),
        ("p wave f2 ~ sin ky (1 - cos kx)", np.sin(ky) * (1.0 - np.cos(kx))),
        ("u_plus ~ sin((kx+ky)/2)", np.sin(0.5 * (kx + ky))),
        ("u_minus ~ sin((kx-ky)/2)", np.sin(0.5 * (kx - ky))),
    ]
    if params.orbital_model == "projected_two_orbital":
        _, h1, h3 = projected_orbital_components(kx, ky, params)
        bases.append(("h_1(k)", h1))
        bases.append(("h_3(k)", h3))
    return bases


def filter_harmonic_basis(
    bases: list[tuple[str, Array]],
    basis_set: str,
) -> list[tuple[str, Array]]:
    if basis_set in (None, "full"):
        return bases
    if basis_set == "paper":
        keep_prefixes = ("s prime", "d prime", "p wave", "u_plus", "u_minus")
        return [(name, basis) for name, basis in bases if name.startswith(keep_prefixes)]
    raise ValueError("harmonic_basis_set must be 'full' or 'paper'")


def harmonic_overlap_weight(params: GapEquationParams, band: BandData) -> Array:
    mode = params.overlap_weight_mode
    if mode in (None, "uniform"):
        return np.ones_like(band.kx_grid, dtype=float)
    if mode in ("pair_factor", "W", "w"):
        return np.maximum(band.W.reshape(params.Nk, params.Nk), 0.0)
    if mode == "fs_shell":
        abs_xi = np.abs(band.xi.reshape(params.Nk, params.Nk))
        shell = abs_xi <= params.fs_cutoff
        if not np.any(shell):
            shell = abs_xi == np.nanmin(abs_xi)
        return shell.astype(float)
    raise ValueError("overlap_weight_mode must be 'uniform', 'pair_factor', or 'fs_shell'")


def harmonic_overlaps(
    vec: Array,
    band: BandData,
    params: GapEquationParams,
    top: int = 4,
) -> list[dict[str, object]]:
    data = gap_grid(vec, params.Nk)
    weight = harmonic_overlap_weight(params, band)
    bases = filter_harmonic_basis(harmonic_basis(params, band), params.harmonic_basis_set)
    rows = []
    for name, basis in bases:
        finite = np.isfinite(data) & np.isfinite(basis) & np.isfinite(weight)
        data_f = np.where(finite, data, 0.0)
        basis_f = np.where(finite, basis, 0.0)
        weight_f = np.where(finite, weight, 0.0)

        numerator = float(np.sum(weight_f * data_f * basis_f))
        data_norm = float(np.sum(weight_f * data_f * data_f))
        basis_norm = float(np.sum(weight_f * basis_f * basis_f))
        denom = float(np.sqrt(data_norm * basis_norm))
        coeff = numerator / basis_norm if basis_norm > 1e-14 else 0.0
        overlap = numerator / denom if denom > 1e-14 else 0.0
        rows.append(
            {
                "name": name,
                "coeff": coeff,
                "overlap": overlap,
                "abs_overlap": abs(overlap),
                "weight_mode": params.overlap_weight_mode,
                "basis_set": params.harmonic_basis_set,
            }
        )
    rows.sort(key=lambda item: item["abs_overlap"], reverse=True)
    return rows[:top]


def dominant_gap_structure(vec: Array, band: BandData, params: GapEquationParams) -> str:
    rows = harmonic_overlaps(vec, band, params, top=1)
    if not rows:
        return "unclassified"
    best = rows[0]
    return (
        f"{best['name']} (overlap {best['overlap']:+.3f}, "
        f"{best['weight_mode']}, {best['basis_set']} basis)"
    )


def plot_gap_heatmap(
    result: ChannelResult,
    band: BandData,
    params: GapEquationParams,
    output_dir: Path,
) -> Path:
    data = gap_grid(result.eigenvector, params.Nk)
    vmax = max(float(np.max(np.abs(data))), 1e-12)
    fig, ax = plt.subplots(figsize=(4.8, 4.2), constrained_layout=True)
    mesh = ax.pcolormesh(
        band.kx_grid,
        band.ky_grid,
        data,
        shading="auto",
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, label=r"Re aligned $\Omega_\nu(k)$")
    ax.set_title(
        rf"$\nu={result.channel}$, $\lambda={result.eigenvalue.real:.5g}$, "
        rf"$\eta={result.eta}$"
    )
    ax.set_xlabel(r"$k_x$")
    ax.set_ylabel(r"$k_y$")
    ax.set_aspect("equal")
    ax.set_xticks([-np.pi, 0.0, np.pi])
    ax.set_xticklabels([r"$-\pi$", "0", r"$\pi$"])
    ax.set_yticks([-np.pi, 0.0, np.pi])
    ax.set_yticklabels([r"$-\pi$", "0", r"$\pi$"])

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"gap_channel_{result.channel}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def solve_all_channels(
    params: GapEquationParams,
    tau_matrices: Iterable[Array] | None = None,
    tau_labels: Iterable[str] | None = None,
    s_mu: Iterable[float] | None = None,
    chi_func: Susceptibility | None = None,
) -> tuple[list[ChannelResult], BandData]:
    """Solve the fixed-nu symmetrized gap equation for all four channels."""

    if tau_matrices is None:
        if params.orbital_model == "projected_two_orbital":
            tau_list, labels = pauli_basis_2()
        else:
            tau_list, labels = complete_hermitian_basis_3()
    else:
        tau_list = [np.asarray(mat, dtype=complex) for mat in tau_matrices]
        labels = list(tau_labels) if tau_labels is not None else [f"tau{m}" for m in range(len(tau_list))]

    if s_mu is None:
        if params.orbital_model == "projected_two_orbital" and len(tau_list) == 4:
            # Derive the same orbital sign vector used in lieb_routei_framework:
            # tau_z tau_mu tau_z = s_mu tau_mu gives
            # s_mu = (+1, -1, -1, +1) for (tau0, taux, tauy, tauz).
            # The external s_nu signs remain convention dependent and are kept
            # in params.s_nu.
            tau_z = tau_list[3]
            s_mu_arr = infer_s_mu_from_tau_z(tau_list, tau_z)
        else:
            raise ValueError(
                "s_mu must be provided for custom or non-Pauli orbital bases. "
                "For the default projected two-orbital Pauli basis it is "
                "derived from tau_z tau_mu tau_z = s_mu tau_mu."
            )
    else:
        s_mu_arr = np.asarray(list(s_mu), dtype=float)
    if s_mu_arr.shape != (len(tau_list),):
        raise ValueError("s_mu must have one sign per tau_mu matrix")

    print(f"Using tau_mu basis: {', '.join(labels)}")
    print(f"Using s_mu signs: {s_mu_arr.tolist()}")
    print(f"Using s_nu signs: {params.s_nu}")
    print("Reminder: each K_nu is built separately; the solver never sums over nu.")

    band = diagonalize_lieb_band(params)
    g_mu = compute_form_factors_on_grid(band.eigenvectors, tau_list, params.Nk)

    results: list[ChannelResult] = []
    output_dir = Path(params.output_dir)
    for channel in CHANNELS:
        K = build_kernel_for_channel(channel, params, band, g_mu, s_mu_arr, chi_func=chi_func)
        eta = ETA_NU[channel]
        eigenvalue, eigenvector = solve_channel_eigenproblem(K, params, eta)
        err = parity_error(eigenvector, params.Nk, eta)
        structure = dominant_gap_structure(eigenvector, band, params)
        result = ChannelResult(
            channel=channel,
            eta=eta,
            eigenvalue=eigenvalue,
            eigenvector=eigenvector,
            parity_error=err,
            dominant_structure=structure,
            plot_path=None,
        )
        if params.make_plots:
            result.plot_path = plot_gap_heatmap(result, band, params, output_dir)
        results.append(result)

    results.sort(key=lambda item: np.real(item.eigenvalue), reverse=True)
    return results, band


def result_table(results: list[ChannelResult]) -> list[dict[str, object]]:
    return [
        {
            "channel": item.channel,
            "eta_nu": item.eta,
            "lambda_real": float(np.real(item.eigenvalue)),
            "lambda_imag": float(np.imag(item.eigenvalue)),
            "parity_error": item.parity_error,
            "dominant_structure": item.dominant_structure,
            "plot_path": str(item.plot_path) if item.plot_path is not None else "",
        }
        for item in results
    ]


def print_result_table(results: list[ChannelResult]) -> None:
    rows = result_table(results)
    headers = ["nu", "eta", "Re(lambda)", "Im(lambda)", "parity error", "structure"]
    print()
    print("Leading fixed-nu channels")
    print("-------------------------")
    print(
        f"{headers[0]:>3}  {headers[1]:>4}  {headers[2]:>14}  "
        f"{headers[3]:>10}  {headers[4]:>12}  {headers[5]}"
    )
    for row in rows:
        print(
            f"{row['channel']:>3}  {row['eta_nu']:>4}  "
            f"{row['lambda_real']:>+14.7e}  {row['lambda_imag']:>+10.2e}  "
            f"{row['parity_error']:>12.3e}  {row['dominant_structure']}"
        )


def main() -> None:
    params = GapEquationParams()
    results, _ = solve_all_channels(params)
    print_result_table(results)
    if params.make_plots:
        print(f"\nSaved heatmaps to: {Path(params.output_dir).resolve()}")


if __name__ == "__main__":
    main()
