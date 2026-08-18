"""Yeap et al. PRB 79, 075305 (2009): the band-edge half of their calculation, at 80 K.

This is everything in their Figs. 2, 5 and 6(a) EXCEPT the hole level itself -- i.e. strain,
deformation potentials, and the k = 0 band edges. No confined-state eigensolve is involved, so it
runs in seconds, is unaffected by the interface artifact documented in `kp_confined`, and is
independent of the resolution floor that limits how much of their Fig. 6(b) can be reproduced.

Why this carries weight. Their Fig. 6(a) decomposes the transition into four separately plotted
curves: the electron level, the hole level, the strained dot valence edge, and the matrix valence
edge at the interface. Only the hole level needs the eigensolver. If the other three match, then
any disagreement in the transition energy is attributable to the hole level alone rather than to
the strain or the band alignment -- which is the isolation this whole staging exists to buy.

Energy zero
-----------
Both papers put zero at the UNSTRAINED InSb valence edge; this package's default puts it at the
unstrained matrix valence edge. The two differ by exactly VBO(matrix), which is -0.59 eV for
InAs, so every number here is converted before comparison. Skipping that conversion would shift
everything by 590 meV while leaving all the trends intact -- which is precisely the kind of error
that looks like physics.

Run:  python scripts/yeap_band_profile.py
"""
import os
import sys

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import heterostructure as hs
import kp_pryor as kp
import materials as mt

BASE = 20.0          # nm. Band edges are size-independent, so this is chosen for resolution.
CELLS = 12           # across the height
T = 80.0

#: Read off their Fig. 6(a) by eye, on the unstrained-InSb-valence zero.
#:
#: TREAT THE DOT-EDGE COLUMN AS THE WEAKER NUMBER. These are estimates from a small printed
#: figure with no tabulated values, and the dot edge at AR = 1 can be checked independently:
#: a SPHERICAL inclusion has purely hydrostatic strain, so there is no shear and no valence
#: splitting, and the edge is just -a_v * Tr(eps) = -(-0.36)(-0.0908) = -0.0327 eV. This code
#: gives -0.0313, agreeing with that closed form to 1.4 meV, while the figure read says +0.02.
#: The tolerance below is therefore set to 60 meV for the dot edge -- wide enough to accommodate
#: the read, not the calculation. The matrix column is a much sharper comparison because those
#: values sit on well-separated curves.
YEAP_FIG6A = {
    #  AR:  (dot hh edge, matrix hh edge at the interface, electron level E1)
    1.0: (+0.02, -0.79, -0.195),
    2.0: (+0.09, -0.75, -0.195),
    4.0: (+0.14, -0.72, -0.195),
}
DOT_EDGE_TOL = 0.060      # figure-read limited; see above
IFACE_TOL = 0.040


def edges(env):
    """k = 0 valence branches and conduction edge over the whole grid."""
    m = env['mask']
    pick = lambda k: np.where(m, env['dot'][k], env['matrix'][k])
    return kp.local_band_edges(env['strain'], env['Ev'], env['Ec'], pick('delta_so'),
                               pick('a_c'), pick('a_v'), pick('b'), pick('d'),
                               hydrostatic_applied=True)


def heavy_hole(env, e):
    """Energy of the branch with the most HEAVY-HOLE character at each point.

    Not v1. `local_band_edges` returns v1 >= v2 >= v3 *numbered, not labelled*, and which branch
    is heavy-hole depends on the sign of the local shear. Inside the compressed dot the heavy hole
    is on top, so v1 is right there. In the TENSILE matrix around the dot the ordering inverts:
    the heavy hole bends DOWN and the light hole bends UP, so v1 is the light hole and taking it
    reports the wrong branch by ~500 meV.

    Rybchenko/Yeap describe exactly this -- the heavy-hole edge in the matrix near the interface
    is "lower in energy than in the matrix away from the dot... which increases the heavy-hole
    confinement even further", while the light-hole edge there is "higher". Their Fig. 6(a)
    plots the heavy hole, so a comparison against v1 is a comparison against the wrong band.
    """
    m = env['mask']
    pick = lambda k: np.where(m, env['dot'][k], env['matrix'][k])
    ch = kp.local_band_character(env['strain'], env['Ev'], pick('delta_so'), pick('a_v'),
                                 pick('b'), pick('d'), hydrostatic_applied=True)
    stack = np.stack([e['v1'], e['v2'], e['v3']], axis=-1)      # (..., level)
    idx = np.argmax(ch[..., :, 0], axis=-1)                     # HH weight is character 0
    return np.take_along_axis(stack, idx[..., None], axis=-1)[..., 0]


def profile(dot, matrix, AR, cells=CELLS, base=BASE):
    """Dot interior and matrix-at-interface band edges, on the papers' zero."""
    height = base / AR
    h = max(height / cells, 0.15)
    env = hs.build(hs.ellipsoid(base, height), dot, matrix=matrix, h=h,
                   pad=base, z_pad=base, use_piezo=False, vol_tol=0.35, verbose=False)
    m = env['mask']
    inner = binary_erosion(m, iterations=1)
    if not inner.any():
        inner = m
    shell = binary_dilation(m, iterations=1) & ~m      # one voxel of matrix, touching the dot

    e = edges(env)
    hh = heavy_hole(env, e)
    shift = env['matrix']['VBO']            # repo zero -> papers' zero (unstrained InSb VB = 0)
    # MEAN over the eroded interior, not max. Eshelby makes the interior homogeneous, so the mean
    # is the estimator of a constant while the max is biased upward by whatever scatter the
    # staircased mask leaves -- measured at 5-12 meV, which is exactly the size of the spurious
    # h-dependence a max produces.
    return dict(
        AR=AR, h=h, n_in=int(inner.sum()), n_shell=int(shell.sum()),
        v1_dot=float(hh[inner].mean()) + shift,
        v1_dot_std=float(hh[inner].std()),
        v1_dot_max=float(hh[inner].max()) + shift,
        v2_dot=float(e['v2'][inner].mean()) + shift,
        v3_dot=float(e['v3'][inner].mean()) + shift,
        cb_dot=float(e['cb'][inner].mean()) + shift,
        # Matrix: the heavy hole bends DOWN at the interface, so the extremum of interest is the
        # minimum of the HH branch in the shell, not the maximum of v1.
        v1_iface=float(hh[shell].min()) + shift,
        v1_far=float(hh[~m].min()) + shift,
        cb_far=float(env['Ec_far']) + shift,
        vol_err=env['vol_err'],
    )


if __name__ == '__main__':
    mt.set_temperature(T)
    InSb, InAs = mt.material('InSb'), mt.material('InAs')
    fail = []

    def check(name, got, want, tol, unit='meV', scale=1e3):
        ok = abs(got - want) <= tol
        if not ok:
            fail.append(name)
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {got*scale:+8.1f} vs {want*scale:+8.1f} "
              f"{unit} (tol {tol*scale:.0f})")

    print(f"Yeap et al. band-edge targets, T = {T:g} K, zero = unstrained InSb valence edge\n")

    print("1. UNSTRAINED ALIGNMENT (their Fig. 2a)")
    off = InSb['VBO'] - InAs['VBO']
    print(f"       InSb Ev {InSb['VBO'] - InSb['VBO']:+.4f}   InAs Ev "
          f"{InAs['VBO'] - InSb['VBO']:+.4f}   InAs Ec "
          f"{InAs['VBO'] - InSb['VBO'] + InAs['Eg']:+.4f} eV")
    check("valence offset InSb-InAs (their 'approx 600 meV')", off, 0.590, 0.020)
    overlap = -(InAs['VBO'] - InSb['VBO'] + InAs['Eg'])
    check("broken-gap overlap, unstrained (their Fig 2a ~180 meV)", overlap, 0.180, 0.020)

    print("\n2. STRAINED SPHERICAL DOT (their Fig. 2b)")
    p = profile('InSb', 'InAs', 1.0)
    hh_conf = p['v1_dot'] - p['v1_iface']
    lh_conf = p['v2_dot'] - p['v1_iface']
    print(f"       dot v1 {p['v1_dot']:+.4f}  v2 {p['v2_dot']:+.4f}  v3 {p['v3_dot']:+.4f}  "
          f"cb {p['cb_dot']:+.4f} eV")
    print(f"       matrix at interface v1 {p['v1_iface']:+.4f}, far {p['v1_far']:+.4f} eV")
    check("heavy-hole confinement potential (their 'approx 800 meV')", hh_conf, 0.800, 0.040)
    print(f"       light-hole confinement {lh_conf*1e3:+.1f} meV "
          f"(their 'does not exceed 600 meV'): "
          f"{'OK' if lh_conf <= 0.640 else 'EXCEEDS'}")

    print("\n3. BAND EDGES vs ASPECT RATIO (their Fig. 6a)")
    print(f"{'AR':>5} {'h':>6} {'dot hh':>9} {'target':>8} | {'matrix hh':>10} {'target':>8} | "
          f"{'interior std':>13} {'vol_err':>8}")
    print("-" * 78)
    rows = {}
    for AR in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0):
        p = profile('InSb', 'InAs', AR)
        rows[AR] = p
        tgt = YEAP_FIG6A.get(AR)
        t_dot = f"{tgt[0]:+8.2f}" if tgt else " " * 8
        t_ifc = f"{tgt[1]:+8.2f}" if tgt else " " * 8
        print(f"{AR:5.1f} {p['h']:6.3f} {p['v1_dot']:+9.4f} {t_dot} | {p['v1_iface']:+10.4f} "
              f"{t_ifc} | {p['v1_dot_std']*1e3:12.2f}m {p['vol_err']:+8.3f}", flush=True)
    for AR, tgt in YEAP_FIG6A.items():
        if AR in rows:
            check(f"dot hh edge at AR={AR:g}", rows[AR]['v1_dot'], tgt[0], DOT_EDGE_TOL)
            check(f"matrix hh edge at interface, AR={AR:g}", rows[AR]['v1_iface'], tgt[1],
                  IFACE_TOL)

    # The sphere is the one geometry with a closed form, and it is a far sharper test than any
    # figure read: purely hydrostatic strain means no shear, no valence splitting, and the edge
    # is exactly -a_v * Tr(eps).
    p1 = rows[1.0]
    import elasticity_fd as ef
    K_i = ef.voigt_moduli(*mt.elastic(InSb))[0]
    mu_m = ef.voigt_moduli(*mt.elastic(InAs))[1]
    tr = ef.inhomogeneous_sphere_trace(mt.misfit(InSb, InAs), K_i, mu_m)
    analytic = -InSb['a_v'] * tr
    print(f"\n   ANALYTIC SPHERE CHECK (no shear, edge = -a_v*Tr): Tr = {tr:+.6f}, "
          f"edge = {analytic:+.4f} eV")
    check("dot hh edge at AR=1 vs the closed form", p1['v1_dot'], analytic, 0.005)

    print("\n4. h-INDEPENDENCE of the dot edge (homogeneous interior => must not move)")
    print(f"{'cells':>6} {'h':>6} {'dot hh':>9} {'interior std':>13}")
    print("-" * 40)
    vals = []
    for cells in (8, 12, 20):
        p = profile('InSb', 'InAs', 2.0, cells=cells)
        vals.append(p['v1_dot'])
        print(f"{cells:6d} {p['h']:6.3f} {p['v1_dot']:+9.4f} {p['v1_dot_std']*1e3:12.2f}m",
              flush=True)
    check("dot hh edge spread over 8..20 cells", max(vals) - min(vals), 0.0, 0.005)

    print("\n5. COMPOSITION TREND (their Fig. 5: hh and lh move DOWN with As, split-off UP)")
    print(f"{'x_As':>6} {'v1 (hh)':>9} {'v2':>9} {'v3 (so)':>9}")
    print("-" * 36)
    prev = None
    trend_ok = True
    for x_as in (0.0, 0.2, 0.4):
        p = profile(('InAsSb', 1.0 - x_as), 'InAs', 2.0)
        print(f"{x_as:6.1f} {p['v1_dot']:+9.4f} {p['v2_dot']:+9.4f} {p['v3_dot']:+9.4f}",
              flush=True)
        if prev is not None:
            if not (p['v1_dot'] < prev['v1_dot'] and p['v3_dot'] > prev['v3_dot']):
                trend_ok = False
        prev = p
    if not trend_ok:
        fail.append("composition trend signs")
    print(f"  [{'PASS' if trend_ok else 'FAIL'}] hh moves down and split-off moves up with As")

    print("\n" + "=" * 70)
    if fail:
        print(f"{len(fail)} CHECK(S) FAILED:")
        for f in fail:
            print(f"  - {f}")
        sys.exit(1)
    print("All Yeap band-edge targets met.")
