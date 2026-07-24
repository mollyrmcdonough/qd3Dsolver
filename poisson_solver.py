"""3D Poisson solver on the same grid/discretization as the rest of qd3Dsolver.

-div(eps_r(r) grad phi) = rho(r)/eps0 is mathematically identical in structure to the
Ben Daniel-Duke kinetic operator already built and validated for the k.p diagonal (k_i^2) terms
in kp_confined.py -- same "div(coefficient * grad)" form, just with the relative dielectric
constant eps_r(r) as the coefficient instead of a Luttinger parameter, and no hbar^2/2m0
prefactor. Reuses that exact (tested) discretization rather than re-deriving it.

Units: lengths in nm, charge density in elementary charges per nm^3, potential in Volts (so that,
for a singly-charged carrier, q*phi in eV numerically equals phi in Volts -- the standard
semiconductor-device convention).

References: the macroscopic Poisson equation in a linear dielectric, -div(eps grad phi) = rho,
is standard electrostatics -- see, e.g., J. D. Jackson, "Classical Electrodynamics"
(electrostatics of macroscopic media). The unit constants below were validated numerically
against Coulomb's law (point charge in uniform eps_r; agreement to 0.5% at the largest tested
box) rather than by transcription, per the citation policy in qdsolver_core.py.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from kp_confined import diagonal_k2_operator

E_CHARGE = 1.602176634e-19  # C
EPS0 = 8.8541878128e-12     # F/m

# -div(eps_r grad phi) = rho[e/nm^3] * POISSON_CONST  (derived by converting the SI Poisson
# equation's length scale from meters to nm; see the notebook for the derivation)
POISSON_CONST = E_CHARGE * 1e9 / EPS0  # V*nm

# Coulomb's law constant in the same units, phi(r) = Q[e] * COULOMB_CONST / (eps_r * r[nm])
COULOMB_CONST = E_CHARGE / (4 * np.pi * EPS0 * 1e-9)  # V*nm


def build_poisson_operator(eps_r_field, h, periodic=False):
    """-div(eps_r(r) grad) as a sparse real matrix (1/nm^2, before the POISSON_CONST factor)."""
    op = diagonal_k2_operator(eps_r_field, h, periodic)
    return (op(0) + op(1) + op(2)).real.tocsr()


def solve_poisson(eps_r_field, rho_field, h, periodic=False, L=None, x0=None, rtol=1e-10):
    """Solve -div(eps_r grad phi) = rho(r)[e/nm^3] * POISSON_CONST for phi (Volts).

    Uses CG (the operator is real symmetric positive-definite) rather than a direct sparse
    solve -- direct factorization of this operator was measured to scale very badly for large
    3D grids (a 121^3 grid didn't finish in several minutes), while CG solved the same system
    in about 20s and, more importantly, scales far better as the grid grows. Pass a precomputed
    `L` (from build_poisson_operator) and the previous solution as `x0` to speed up repeated
    solves inside a self-consistency loop, where eps_r_field doesn't change between iterations.
    """
    if L is None:
        L = build_poisson_operator(eps_r_field, h, periodic)
    rhs = rho_field.ravel() * POISSON_CONST
    phi, info = spla.cg(L, rhs, rtol=rtol, x0=x0, maxiter=5000)
    if info != 0:
        raise RuntimeError(f"Poisson CG solve did not converge (info={info})")
    return phi.reshape(eps_r_field.shape)
