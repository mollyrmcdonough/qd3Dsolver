"""Reproduce Pryor's Fig. 6: ground-state exciton electron and hole wave functions.

Source: C. Pryor, Phys. Rev. B 57, 7190 (1998), preprint arXiv:cond-mat/9710304. Fig. 6's
caption, verbatim:

    FIG. 6. Electron and hole wave functions for the ground state exciton in the Hartree
    approximation, with b = 14 nm. Surfaces are sum_{i=1}^{8} |psi_i(r)|^2 equal to 0.1 of the
    peak value.

So: eight-band Hamiltonian (the sum runs over all eight spinor components), b = 14 nm, Hartree
electron-hole self-consistency, and an isosurface at exactly 0.1 of the peak density.

Sec. V describes the self-consistency: "psi^e and psi^h were found by self-consistent iteration,
with convergence to within 0.1 meV usually taking only two iterations", and states what the
figure is meant to show -- "In spite of the complex band structure seen in Fig. 2, the electron
and hole wave functions appear ordinary. The wave functions are spread out over most of the
island, with no signs of localization around smaller regions."

Strain comes from elasticity_fd (Pryor's own method, position-dependent elastic constants).

Run: python pryor_fig6.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import qdsolver_core as qd
import elasticity_fd as ef
import piezoelectric as pz
import kp_confined as kpc
import kp_pryor as kp
import pryor1998 as pr
import eigensolvers as eig
from poisson_solver import solve_poisson, build_poisson_operator

BASE = 14.0
dot, matrix = pr.PRYOR_TABLE_I['InAs'], pr.PRYOR_TABLE_I['GaAs']
eps_star = qd.eigenstrain(dot['a0'], matrix['a0'])
C_dot = (dot['C11'], dot['C12'], dot['C44'])
C_mat = (matrix['C11'], matrix['C12'], matrix['C44'])
EXACT_VOL = BASE ** 2 * (BASE / 2) / 3


def clean_grid(h, pad):
    """Mirror-symmetric grid with the island base exactly on a grid plane."""
    nx = int(round(2 * (BASE / 2 + pad) / h)) // 2 * 2 + 1
    kz0 = int(round(pad / h))
    nz = kz0 + int(round((BASE / 2 + pad) / h)) + 1
    cx = qd.centered_axis(nx, h)
    cz = (np.arange(nz) - kz0) * h
    X, Y, Z = np.meshgrid(cx, cx, cz, indexing='ij')
    pyr = qd.pyramid_mask(X, Y, Z, BASE)
    assert qd.mirror_asymmetry(pyr, 0) == 0 and qd.mirror_asymmetry(pyr, 1) == 0
    return cx, cz, X, Y, Z, pyr


def density(vec, shape):
    """sum_{i=1}^{8} |psi_i(r)|^2, normalized to unit integral over the grid (per unit volume
    is applied by the caller through h^3). This is exactly the quantity Pryor isosurfaces."""
    n = int(np.prod(shape))
    comps = np.asarray(vec).reshape(8, n)
    rho = (np.abs(comps) ** 2).sum(axis=0)
    return (rho / rho.sum()).reshape(shape)


def build(h=1.0, pad=8.0, use_piezo=True, verbose=True):
    cx, cz, X, Y, Z, pyr = clean_grid(h, pad)
    t0 = time.time()
    strain = ef.solve_strain_fd(pyr, eps_star, C_dot, C_mat, h, tol=1e-10)
    g = strain.at(pyr)
    if verbose:
        print(f"grid {X.shape} = {X.size:,} pts, h={h}, "
              f"vol err {qd.mask_volume_error(pyr,h,EXACT_VOL):+.3f}")
        print(f"  FD elasticity: CG {strain.cg_iterations} iters, rel {strain.cg_residual:.1e}, "
              f"{time.time()-t0:.0f}s;  <Tr eps> = {g['exx']+g['eyy']+g['ezz']:+.5f}")

    phi = np.zeros(X.shape)
    if use_piezo:
        e14 = np.where(pyr, dot['e14'], matrix['e14'])
        phi = pz.potential(strain, e14, matrix['eps_R'], h)
        if verbose:
            print(f"  piezo potential {phi.min()*1e3:+.1f} .. {phi.max()*1e3:+.1f} meV, "
                  f"C4 antisymmetry {pz.c4_antisymmetry_error(phi):.1e}")
    return dict(cx=cx, cz=cz, X=X, Y=Y, Z=Z, pyr=pyr, strain=strain, phi=phi, h=h)


def eight_band_states(env, U_extra=0.0, k=2, tol=1e-7, maxiter=4000, verbose=True):
    """Electron and hole ground states of the eight-band Hamiltonian.

    `U_extra` is an ELECTRON potential energy (eV) added to every band -- both E_c and E_v
    shift together under an electrostatic potential, which is the whole point of doing this in
    the eight-band electron convention rather than patching E_c and E_v in opposite directions.
    """
    pyr, h, strain, phi = env['pyr'], env['h'], env['strain'], env['phi']
    V_e, V_h, _, _ = pr.band_edge_fields(pyr, strain.trace)

    # Electrostatic potential energy of an ELECTRON is -e*phi = -phi (eV per Volt).
    U = -phi + U_extra
    Ec, Ev = V_e + U, V_h + U

    ops = kpc.GridOperators(pyr.shape, h, periodic=False)
    fields = kp.material_fields(pyr, dot, matrix, Ev, Ec, n_bands=8)
    H = kp.confined_hamiltonian(ops, fields, n_bands=8, strain=strain)

    sig_e = float(Ec[pyr].min())
    sig_h = kp.hole_sigma(Ev, strain, np.where(pyr, dot['b'], matrix['b']),
                          np.where(pyr, dot['d'], matrix['d']),
                          np.where(pyr, dot['delta_so'], matrix['delta_so']), inside_mask=pyr)
    out = {}
    for label, sig in (('electron', sig_e), ('hole', sig_h)):
        if verbose:
            print(f"    {label}: sigma = {sig:.4f} eV")
        E, V, info = eig.solve_interior(H, k=k, sigma=sig, tol=tol, maxiter=maxiter,
                                        verbose=verbose)
        idx = int(np.argmin(np.abs(E - sig))) if label == 'hole' else int(np.argmin(np.abs(E - sig)))
        out[label] = dict(E=E, V=V, pick=idx, info=info)
    return out, H


def hartree(env, n_iter=3, k=2, verbose=True):
    """Self-consistent Hartree electron-hole pair, as in Pryor's Sec. V.

    Two separate Poisson solves per iteration, so neither particle feels its own charge: the
    electron moves in the potential of the hole's density and vice versa. This is the same
    no-self-interaction construction as schrodinger_poisson.self_consistent_exciton, lifted to
    the eight-band density sum_i |psi_i|^2.
    """
    shape, h = env['pyr'].shape, env['h']
    eps_field = np.full(shape, matrix['eps_R'])       # Pryor's Table I footnote: InAs value used
    L = build_poisson_operator(eps_field, h)          # throughout the structure
    phi_n = np.zeros(shape)
    phi_p = np.zeros(shape)
    hist = []

    for it in range(n_iter):
        if verbose:
            print(f"  Hartree iteration {it}")
        # Electron sees the hole's potential; hole sees the electron's.
        st_e, _ = eight_band_states(env, U_extra=-phi_p, k=k, verbose=verbose)
        e = st_e['electron']
        E_e = e['E'][e['pick']]
        n_dens = density(e['V'][:, e['pick']], shape)

        st_h, _ = eight_band_states(env, U_extra=-phi_n, k=k, verbose=False)
        hh = st_h['hole']
        E_h = hh['E'][hh['pick']]
        p_dens = density(hh['V'][:, hh['pick']], shape)

        # Densities are normalized to one particle, so rho = +/- density / h^3 in e/nm^3.
        phi_n = solve_poisson(eps_field, -n_dens / h ** 3, h, L=L, x0=phi_n.ravel())
        phi_p = solve_poisson(eps_field, p_dens / h ** 3, h, L=L, x0=phi_p.ravel())

        E_int = float(np.sum(p_dens * phi_n))         # eV, counted once
        hist.append(dict(it=it, E_e=E_e, E_h=E_h, E_int=E_int, gap=E_e - E_h + E_int))
        if verbose:
            print(f"    E_e = {E_e:.5f}  E_h = {E_h:.5f}  E_int = {E_int*1e3:+.2f} meV  "
                  f"exciton = {(E_e-E_h+E_int)*1e3:.1f} meV")

    return dict(n=n_dens, p=p_dens, E_e=E_e, E_h=E_h, phi_n=phi_n, phi_p=phi_p, history=hist)


def pyramid_edges(base):
    """The 8 edges of the square-based {101} pyramid, as (x, y, z) segment endpoints."""
    a = base / 2.0
    corners = [(-a, -a, 0.0), (a, -a, 0.0), (a, a, 0.0), (-a, a, 0.0)]
    apex = (0.0, 0.0, a)
    segs = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    segs += [(c, apex) for c in corners]
    return segs


def plot_wavefunctions(env, res, level=0.1, elev=18, azim=-58, figsize=(11, 5)):
    """Pryor's Fig. 6: isosurfaces of sum_i |psi_i|^2 at `level` times the peak value, inside
    a wireframe of the island.

    The isosurface level is Pryor's, not a tuning knob: his caption specifies 0.1 of the peak.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
    from skimage import measure

    cx, cz, h, pyr = env['cx'], env['cz'], env['h'], env['pyr']
    origin = (cx[0], cx[0], cz[0])
    a = BASE / 2.0

    fig = plt.figure(figsize=figsize)
    for n, (label, dens, colour) in enumerate((('Electron', res['n'], '#4a7fb5'),
                                               ('Hole', res['p'], '#5a5a5a'))):
        ax = fig.add_subplot(1, 2, n + 1, projection='3d')

        verts, faces, _, _ = measure.marching_cubes(dens, level * dens.max(),
                                                    spacing=(h, h, h))
        verts = verts + np.array(origin)
        mesh = Poly3DCollection(verts[faces], alpha=0.85, facecolor=colour,
                                edgecolor='none')
        ax.add_collection3d(mesh)

        ax.add_collection3d(Line3DCollection(pyramid_edges(BASE), colors='k', linewidths=1.2))

        # The {101} facets fix height = base/2, so the box aspect must be 1:1:0.5 or the
        # island is drawn twice as tall as it is.
        ax.set_xlim(-a, a); ax.set_ylim(-a, a); ax.set_zlim(0, a)
        ax.set_box_aspect((1, 1, 0.5), zoom=1.25)
        ax.view_init(elev=elev, azim=azim)
        ax.set_axis_off()
        ax.set_title(label, fontsize=11)
        frac = dens[pyr].sum()
        ax.text2D(0.5, 0.03, f"{frac*100:.0f}% of the density inside the island",
                  transform=ax.transAxes, ha='center', fontsize=8, color='0.35')

    fig.suptitle(r'Ground-state exciton wave functions, $b = 14$ nm '
                 r'(surfaces: $\sum_{i=1}^{8}|\psi_i|^2 = 0.1\times$ peak) — cf. Pryor Fig. 6',
                 fontsize=10)
    fig.tight_layout()
    return fig


if __name__ == '__main__':
    env = build(h=1.0, pad=8.0)
    t0 = time.time()
    res = hartree(env, n_iter=2)
    print(f"\ntotal {time.time()-t0:.0f}s")
    print(f"electron peak density {res['n'].max():.3e}, hole peak {res['p'].max():.3e}")
    inside = env['pyr']
    print(f"fraction of electron density inside the island: {res['n'][inside].sum():.3f}")
    print(f"fraction of hole density inside the island:     {res['p'][inside].sum():.3f}")
