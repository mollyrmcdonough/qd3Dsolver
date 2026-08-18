"""Is the matrix electron actually bound, and by how much? The box series that decides it.

    PYTHONIOENCODING=utf-8 python -u scripts/electron_binding.py

Why this exists
---------------
Every transition energy in this project is E_trans = E_c(reference) - E_hole, and the reference
has been E_c(InAs, far) -- the electron sitting exactly at the matrix conduction band edge, with
no binding. That is an UPPER BOUND on the transition, not the transition.

Yeap et al. do solve for the electron: their Fig. 6(a) E1 sits flat at -0.20 eV on their zero,
which is 16.8 meV below our E_c(InAs, far) = -0.1832 eV on the same zero. So the two calculations
differ by that 17 meV and by nothing else in the electron channel. Folding it in moves every
point of the Fig. 6 comparison uniformly: the AR 2 agreement is 0 meV with it and +17 meV
without, and the curve crossing moves from AR 2.0 to about 2.2.

17 meV read off a panel whose ticks are 0.10 eV apart is under a fifth of a tick, and an earlier
reading of the same figure gave 12 meV. That is not good enough to carry into a production sweep,
so this computes it instead.

THE THING THAT MAKES THIS HARD, AND THE THING THAT MAKES IT INTERESTING
-----------------------------------------------------------------------
The electron in a broken-gap system is not a dot state. The island expels it -- the InSb dot's
conduction edge sits ~810 meV ABOVE the InAs matrix -- and what binds it, if anything, is the
shallow tensile shell the island's compression creates around itself (Yeap's Fig. 2(b) marks it
with dotted ellipses).

A shallow, extended state needs a large box. Hard walls add 3*G0*pi^2/(m L^2) of pure
quantisation: 645 meV at L = 8.1 nm and m = 0.026, which is why an earlier attempt here reported
the state as unbound -- that number was the box and nothing else. It falls only as 1/L^2, so
reaching a few meV needs L ~ 65 nm. `heterostructure.electron_states_bigbox` gets there by
embedding the strained region in far-field material at the same grid spacing (exact, since strain
decays as 1/r^3) and solving matrix-free with a DST preconditioner that is exact in the far field.

**And a shallow 3D well need not bind at all.** Unlike 1D and 2D, three dimensions have a finite
threshold: below it there is no bound state whatsoever. So "unbound" is a physically available
answer here, not a numerical failure -- and it is distinguishable from a box artifact by the
SHAPE of the box series, which is the whole design of section 3:

    genuinely bound   E(L) approaches its limit EXPONENTIALLY, ~exp(-2 kappa L), and settles
    not bound         E(L) - Ec_far falls as 1/L^2 and keeps going, never settling

WHAT IT FOUND: NOT BOUND, AT ANY SIZE IN THE PRODUCTION RANGE
--------------------------------------------------------------
At h = 0.5 nm, AR 1, three island sizes. `excess` is (E - Ec_far) minus the empty-box quantum --
the island's barrier minus the shell's well, with the box arithmetic taken out:

    base 2.5 nm    L    20     30     40     50     60
                   E-Ec_far  88.1   48.3   27.2   17.5   12.1 meV
                   quantum   89.6   48.2   27.1   17.4   12.1 meV
                   excess    -1.5   +0.1   +0.1   +0.1    0.0 meV

    base 10 nm     L    30     40     50     60     80
                   excess   +21.5  +15.1  +10.1   +6.5   +2.9 meV

    base 20 nm     L    40     50     60     80    100    120
                   excess   +34.2  +33.9  +27.4  +10.7   +4.8   +2.5 meV

Every series is pure 1/L^2 with the excess decaying to zero FROM ABOVE. Nothing settles
exponentially; nothing goes negative. On the 2.5 nm island the level is the empty box to 0.1 meV
-- barrier and shell cancel to within a tenth of a meV -- and on the larger islands the barrier
wins outright, so growing the island makes binding LESS likely, not more.

That is the expected sign once stated plainly: the island is a **810 meV barrier** for the
electron and the tensile shell is a thin skin around it. The 3D shallow-well threshold decides it
quantitatively -- 150 meV deep but only ~1 nm thick gives V0*R^2 ~ 0.15 eV.nm^2 against the
~3.6 eV.nm^2 a bound state needs at m = 0.026, short by more than an order of magnitude, which is
why no amount of box refinement rescues it. (`ingasb_dot.pocket_metrics` reached the same verdict
from the criterion alone; this is the direct solve that confirms it.)

Truncating the strain tail cannot be the cause: base 20 nm at L = 80, strain pad 10 -> 16 nm moved
the level +1.2 meV, i.e. slightly LESS bound, despite the shell deepening 145.7 -> 160.9 meV.

**Consequence.** E_c(InAs, far) is the correct electron reference for the whole production size
range, so E_trans = E_c(InAs, far) - E_hole with nothing subtracted -- the solid curve in
`plot_yeap_fig6.py`. Yeap's 17 meV is then a property of their finite FEM domain, the same reading
already reached for their flat-aspect-ratio hole levels, and consistent with it: both are states
that are not actually bound.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import heterostructure as hs
import materials as mt
import qdsolver_core as qd

BASE, T_K = 2.5, 80.0
OUT = '_electron_binding.json'


def section1_empty_box(h=0.5, L=20.0, m=0.026):
    """The solver against the analytic infinite cubic well, with no dot anywhere.

    `build_hamiltonian` uses hard walls, so an empty box must return 3*G0*pi^2/(m L^2) in the
    continuum limit and the exact discrete Dirichlet eigenvalue at finite h. Checking both
    separates a bug in the new LOBPCG/DST path from ordinary discretization error.
    """
    print("=" * 78)
    print("1. EMPTY BOX -- the new solver against the analytic infinite well")
    n = int(round(L / h))
    me = np.full((n, n, n), m)
    V = np.zeros((n, n, n))
    H = qd.build_hamiltonian(me, V, h)

    t = qd.HBAR2_OVER_2M0 / (m * h ** 2)
    exact_discrete = 3.0 * 2.0 * t * (1.0 - np.cos(np.pi / (n + 1)))
    continuum = 3.0 * qd.HBAR2_OVER_2M0 * np.pi ** 2 / (m * ((n + 1) * h) ** 2)

    from scipy.fft import dstn, idstn
    import scipy.sparse.linalg as spla
    lam = np.zeros((n, n, n))
    j = np.arange(1, n + 1)
    w = 2.0 * t * (1.0 - np.cos(np.pi * j / (n + 1)))
    for ax in range(3):
        lam = lam + w.reshape([-1 if a == ax else 1 for a in range(3)])
    sigma = -1e-3

    def mm(x):
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            v = x[:, c].reshape(n, n, n)
            out[:, c] = idstn(dstn(v, type=1, norm='ortho') / (lam - sigma),
                              type=1, norm='ortho').ravel()
        return out

    M = spla.LinearOperator((n ** 3, n ** 3), matvec=lambda v: mm(v.reshape(-1, 1)).ravel(),
                            matmat=mm, dtype=float)
    X = np.random.default_rng(0).standard_normal((n ** 3, 2))
    t0 = time.time()
    E, _ = spla.lobpcg(H, X, M=M, largest=False, tol=1e-10, maxiter=400)
    got = float(np.min(E))
    print(f"   L = {L:g} nm, h = {h:g} nm, m = {m:g}, {n**3:,} points, {time.time()-t0:.1f}s")
    print(f"   LOBPCG + DST      {got*1e3:10.4f} meV")
    print(f"   exact discrete    {exact_discrete*1e3:10.4f} meV   "
          f"(difference {abs(got-exact_discrete)*1e6:.2f} ueV)")
    print(f"   continuum limit   {continuum*1e3:10.4f} meV   "
          f"(discretization {abs(exact_discrete-continuum)*1e3:.3f} meV)")
    ok = abs(got - exact_discrete) < 1e-6
    print(f"   -> {'PASS' if ok else 'FAIL'}: the solver reproduces the discrete Dirichlet "
          f"eigenvalue")
    return ok


def section2_well_integral(AR=1.0, BASE=BASE):
    """Is h = 0.5 fine enough for the electron potential?

    The electron box must be coarse to be affordable, but the tensile shell that might bind it is
    only a nanometre or two thick. A weakly bound 3D state responds to the well mainly through its
    integrated strength, so compare that integral -- not a pointwise field -- between the coarse
    grid the electron solve will use and a fine one.
    """
    print("\n" + "=" * 78)
    print("2. IS h = 0.5 ENOUGH? integrated well strength vs grid")
    print(f"   {'h':>6} {'cells/dot':>10} {'vol err':>9} {'min Ec-Ec_far':>14} "
          f"{'shell vol':>11} {'integral':>13}")
    out = {}
    for h in (0.25, 0.35, 0.50):
        mt.set_temperature(T_K)
        env = hs.build(hs.ellipsoid(BASE, BASE / AR), 'InSb', matrix='InAs', h=h,
                       pad=10.0, vol_tol=0.40, use_piezo=False, verbose=False)
        d = env['Ec_far'] - env['Ec']              # depth below the far field; >0 is a well
        shell = (~env['mask']) & (d > 0)
        integral = float(d[shell].sum() * h ** 3)   # eV.nm^3 -- what a shallow state feels
        out[h] = integral
        print(f"   {h:6.2f} {BASE/h:10.1f} {env['vol_err']:+9.3f} "
              f"{-d[shell].max()*1e3:13.1f}m {shell.sum()*h**3:10.1f}  {integral:12.4f}")
    ref = out[0.25]
    print(f"   -> h = 0.50 carries {out[0.50]/ref*100:.1f}% of the h = 0.25 well integral "
          f"({(out[0.50]/ref-1)*100:+.1f}%)")
    return out


def section3_box_series(AR=1.0, h=0.5, boxes=(20.0, 30.0, 40.0, 50.0, 60.0), store=None,
                        BASE=BASE, pad=10.0):
    """The measurement. Binding vs box size, with the empty-box quantum alongside it.

    Two things move the level and only one of them is physics, which is why the box quantum is
    printed alongside. The island is a BARRIER for the electron -- the InSb conduction edge sits
    ~810 meV above the InAs matrix -- so it pushes the level UP, while the tensile shell pulls it
    down. The excess of E - Ec_far over the empty-box quantum is the net of the two, and its sign
    is the answer: positive means the barrier wins and there is nothing to bind.

    `pad` sets how far the strain solve extends before `embed_electron_fields` continues with far
    field. Truncating the strain tail throws away attractive shell, so it biases toward "unbound"
    -- worth a sensitivity check at the largest island rather than an assumption.
    """
    print("\n" + "=" * 78)
    print(f"3. BOX SERIES at base {BASE:g} nm, AR {AR:g}, h = {h:g} nm, strain pad {pad:g} nm")
    mt.set_temperature(T_K)
    env = hs.build(hs.ellipsoid(BASE, BASE / AR), 'InSb', matrix='InAs', h=h,
                   pad=pad, vol_tol=0.40, use_piezo=False, verbose=False)
    print(f"   Ec(InAs, far) = {env['Ec_far']:.4f} eV;  deepest point of the tensile shell "
          f"{(env['Ec'].min()-env['Ec_far'])*1e3:+.1f} meV below it")
    print(f"   {'L':>6} {'points':>11} {'E':>10} {'binding':>10} {'box quantum':>12} "
          f"{'excess':>9} {'in shell':>9} {'verdict':>9}")
    rows = []
    for L in boxes:
        t0 = time.time()
        r = hs.electron_states_bigbox(env, L, k=2, verbose=False)
        rows.append(dict(base=BASE, AR=AR, h=h, L=L, pad=pad, n_pts=r['n_pts'],
                         E=float(r['E'][0]), binding=r['binding'], box_quantum=r['box_quantum'],
                         # `binding` comes back in meV, `box_quantum` in eV. Keep excess in eV.
                         excess=-r['binding'] * 1e-3 - r['box_quantum'],
                         in_shell=float(r['in_shell'][0]), inside=float(r['inside'][0]),
                         Ec_far=r['Ec_far'], seconds=time.time() - t0))
        b = rows[-1]
        print(f"   {L:6.0f} {b['n_pts']:11,} {b['E']:10.4f} {b['binding']:+9.1f}m "
              f"{b['box_quantum']*1e3:11.1f}m {b['excess']*1e3:+8.1f}m "
              f"{b['in_shell']*100:8.1f}% "
              f"{'BOUND' if b['binding'] > 0 else 'unbound':>9}  {b['seconds']:.0f}s", flush=True)
        if store is not None:
            store[f"b{BASE:g}_AR{AR:.2f}_h{h:.2f}_p{pad:g}_L{L:.0f}"] = b
            json.dump(store, open(OUT, 'w'), indent=1)

    print("\n   How to read the trend:")
    print("     a real bound state settles EXPONENTIALLY once L exceeds a few decay lengths;")
    print("     an unbound one tracks the box quantum down as 1/L^2 and never settles.")
    for a, b in zip(rows, rows[1:]):
        dq = (a['box_quantum'] - b['box_quantum']) * 1e3
        print(f"     L {a['L']:.0f} -> {b['L']:.0f}: E moved {(b['E']-a['E'])*1e3:+7.2f} meV, "
              f"box quantum moved {-dq:+7.2f} meV, excess {b['excess']*1e3:+7.2f} meV")
    print("     excess = (E - Ec_far) - box quantum: the island's barrier minus the shell's well.")
    print("     It must reach 0 from above if unbound, or go negative if a bound state exists.")
    return rows


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ar', type=float, default=1.0)
    ap.add_argument('--base', type=float, default=BASE,
                    help="island base diameter in nm; the tensile shell grows with it, so the "
                         "verdict at Yeap's 2.5 nm does NOT carry to the production sizes")
    ap.add_argument('--h', type=float, default=0.5)
    ap.add_argument('--pad', type=float, default=10.0,
                    help="strain-solve pad in nm; beyond it the electron box is filled with far "
                         "field, which discards attractive shell and biases toward 'unbound'")
    ap.add_argument('--boxes', type=float, nargs='*',
                    default=[20.0, 30.0, 40.0, 50.0, 60.0])
    ap.add_argument('--skip', type=int, default=0, help="skip the first N sections")
    a = ap.parse_args()

    store = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if a.skip < 1:
        section1_empty_box()
    if a.skip < 2:
        section2_well_integral(AR=a.ar, BASE=a.base)
    section3_box_series(AR=a.ar, h=a.h, boxes=tuple(a.boxes), store=store, BASE=a.base,
                        pad=a.pad)
