"""Self-consistent Schrodinger-Poisson loop for a single confined electron-hole pair.

Adds back the Poisson self-consistency aestimo itself does in 1D, extended to the 3D
single-band QD solver: lets the electron and hole attract each other electrostatically
(exciton binding), which the earlier "naive transition energy" (just E_e - E_h from two
independent solves) explicitly ignored.

Self-interaction: sourcing phi from the *combined* density p(r)-n(r) would have each particle
feel a contribution from its own charge, which is spurious for a single electron-hole pair (no
other electrons/holes for it to be shielded by/interact with). Instead, two separate Poisson
solves are used each iteration: phi_from_n (sourced by the electron's density alone, felt by the
hole) and phi_from_p (sourced by the hole's density alone, felt by the electron).

Double counting: the naive E_e - E_h (using the self-consistent eigenvalues directly) is NOT
the exciton transition energy -- each SCF eigenvalue already includes the *full* mutual
interaction energy once (from that particle's own point of view), so summing them counts the
electron-hole attraction twice, the standard Hartree double-counting problem. First attempt at
this gave E_e-E_h *increasing* very slightly under Coulomb attraction (wrong sign entirely for
an attractive interaction) purely from this bug. The correct transition energy subtracts the
interaction energy once: E_int = integral of p(r)*phi_from_n(r) dV (equivalently
-integral of n(r)*phi_from_p(r) dV; the two differ in sign because phi_from_n and phi_from_p
are themselves oppositely signed, not because of an error -- checked they agree in magnitude).

References: this is the self-consistent-field (Hartree) approximation for a single
electron-hole pair; the total-energy double-counting correction is the standard one for
Hartree-type mean-field theories -- see, e.g., R. M. Martin, "Electronic Structure: Basic
Theory and Practical Methods" (Cambridge, 2004), the Hartree/total-energy discussion. What
this level of theory leaves out for excitons: exchange and correlation beyond mean field
(configuration interaction), which full QD treatments such as Stier/Grundmann/Bimberg,
Phys. Rev. B 59, 5688 (1999) go beyond. Per the citation policy in qdsolver_core.py, equation
numbers are not quoted.
"""
import numpy as np

import qdsolver_core as qd
from poisson_solver import solve_poisson, build_poisson_operator


def self_consistent_exciton(m_e_field, m_h_field, V_e_field, V_h_field, eps_field, h,
                             damping=0.3, max_iter=30, tol=1e-5, verbose=True):
    """Returns a dict with the converged E_e, psi_e, E_h, psi_h, phi_from_n, phi_from_p,
    and the naive (non-interacting) and Coulomb-corrected transition energies, plus the
    per-iteration history.
    """
    shape = V_e_field.shape
    phi_from_n = np.zeros(shape)
    phi_from_p = np.zeros(shape)
    L_poisson = build_poisson_operator(eps_field, h)

    history = []
    E_e_naive = E_h_naive = None
    for it in range(max_iter):
        V_e_total = V_e_field - phi_from_p  # electron feels the hole's attractive potential
        V_h_total = V_h_field + phi_from_n  # hole feels the electron's attractive potential

        E_e, psi_e = qd.solve_states(m_e_field, V_e_total, h, n_states=1, hole=False)
        E_h, psi_h = qd.solve_states(m_h_field, V_h_total, h, n_states=1, hole=True)
        if it == 0:
            E_e_naive, E_h_naive = E_e[0], E_h[0]

        n_density = (psi_e[:, 0] ** 2).reshape(shape)
        p_density = (psi_h[:, 0] ** 2).reshape(shape)

        phi_from_n_new = solve_poisson(eps_field, -n_density, h, L=L_poisson, x0=phi_from_n.ravel())
        phi_from_p_new = solve_poisson(eps_field, p_density, h, L=L_poisson, x0=phi_from_p.ravel())

        change = max(np.max(np.abs(phi_from_n_new - phi_from_n)),
                     np.max(np.abs(phi_from_p_new - phi_from_p)))
        phi_from_n = damping * phi_from_n_new + (1 - damping) * phi_from_n
        phi_from_p = damping * phi_from_p_new + (1 - damping) * phi_from_p

        E_int = np.sum(p_density * phi_from_n_new) * (h ** 3)  # eV; see module docstring
        transition = E_e[0] - E_h[0] + E_int  # double-counting-corrected transition energy

        history.append(dict(iter=it, E_e=E_e[0], E_h=E_h[0], E_int=E_int, transition=transition, change=change))
        if verbose:
            print(f"iter {it:2d}: E_e={E_e[0]*1e3:9.3f} meV  E_h={E_h[0]*1e3:9.3f} meV  "
                  f"E_int={E_int*1e3:7.3f} meV  transition={transition*1e3:9.3f} meV  "
                  f"max|dphi|={change*1e3:.5f} mV")
        if change < tol:
            break

    naive_transition = E_e_naive - E_h_naive
    return dict(
        E_e=E_e[0], psi_e=psi_e[:, 0], E_h=E_h[0], psi_h=psi_h[:, 0],
        phi_from_n=phi_from_n, phi_from_p=phi_from_p, E_int=E_int,
        naive_transition=naive_transition,
        coulomb_transition=transition,
        binding_energy=naive_transition - transition,
        history=history,
    )
