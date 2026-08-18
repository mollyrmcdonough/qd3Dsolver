"""Which operator produces the interface artifact? Hold parameters uniform and see what survives.

WHAT THIS SCRIPT FOUND IS STILL RIGHT; ITS PREMISE WAS NOT. It localised the interface modes to
the gamma3 discontinuity in the cross terms, and that stands. It assumed the cause was a
DISCRETIZATION mismatch, and that turned out to be wrong -- see the correction at the end. Read
`kp_planewave` for the mechanism before acting on anything here.

The premise (as written at the time, and mistaken)
--------------------------------------------------
`kp_confined` discretizes the two kinds of second-derivative term differently. Diagonal terms
`k_i alpha k_i` use a compact three-point stencil with alpha averaged at cell FACES; cross terms
`k_i alpha k_j` use central differences with alpha at NODES. Where alpha jumps, the two disagree
about where the material changes, and the standing hypothesis was that this seeds the grid-scale
interface modes that make small dots unreachable (measured on a 2.5 nm island: confinement drifts
430 -> 174 meV over h = 0.50 -> 0.25 with no convergence, localisation stuck at 43-57% against the
80-97% real states show).

`ops.cross` is called with **gamma3 only** (`kp_pryor.py`, the R and S blocks); gamma2 reaches the
Hamiltonian through the well-behaved diagonal operator. So the cross terms can be switched off as
a source of interface variation by holding gamma3 uniform, without touching anything else.

The four cases, and the verdict they returned
---------------------------------------------
    A  full                every field stepped. The known-bad baseline.
    B  uniform g3          cross terms see no interface. Diagonal terms still do.
    C  uniform g1,g2,g3    onto InSb's own set -- no kinetic coefficient step anywhere, and a
                           LARGE gamma3 (16.5) everywhere.
    D  uniform g1,g2,g3    onto InAs's own set -- same, with a small gamma3 (9.2).

Holding a parameter uniform changes the physics, so the energies are NOT comparable between
cases. Only the h-trend is. Confinement in meV over h = 0.50 / 0.40 / 0.32:

    A   429.4 -> 311.3 -> 225.8    (-118, -85)   never settles
    B   601.0 -> 562.8 -> 564.0    ( -38,  +1)   converged
    C   616.4 -> 565.5 -> 558.5    ( -51,  -7)   converged
    D   557.2 -> 509.6 -> 505.5    ( -48,  -4)   converged

**The cross terms are confirmed as the cause.** Every case without a step in gamma3 converges;
the only case with one diverges. C is the control that closes it: it carries a LARGER uniform
gamma3 than the stepped case has anywhere and is still clean, so it is the discontinuity and not
the magnitude.

THE CORRECTION: it is not the STENCIL, it is the OPERATOR
---------------------------------------------------------
The rows above say a gamma3 discontinuity in the cross terms is what diverges. They do not say
that the cross-term DISCRETIZATION is what is wrong, and it is not. Two rewrites of that stencil
changed nothing (`cross_k2_operator_staggered`, `cross_k2_operator_fe`), and a plane-wave basis
with no stencil at all reproduces the divergence exactly.

The mechanism is that the six-band kinetic tensor is Legendre-Hadamard elliptic but not STRONGLY
elliptic -- as an 18x18 matrix over (direction, band) it has eigenvalues up to +1.74 for InSb --
so at a discontinuity the symmetrized operator has no maximum. Scaling gamma2 and gamma3 until
that eigenvalue goes negative makes the divergence vanish with nothing else changed. See
`kp_planewave.strong_ellipticity` and `scripts/planewave_validation.py --only 3`.

And it explains why the divergence tracks gamma3 specifically, which the stencil hypothesis never
did. lambda_max of the kinetic tensor, dropping one coefficient at a time:

                     full      gamma3 = 0    gamma2 = 0    both = 0
    InSb           +1.7412       -0.1448       +0.5601      -1.3259
    InAs           +0.9373       -0.1143       +0.2896      -0.7620

**Setting gamma3 to zero restores strong ellipticity; setting gamma2 to zero does not.** gamma3 is
also the only coefficient `ops.cross` is ever called with. So the parameter this script isolated
by experiment is exactly the one that carries the unboundedness -- the same answer, for a reason
no stencil is involved in. This script measured the right thing and named the wrong culprit.

A second result, and a correction to how these are read: **localisation does not discriminate at
this dot size.** The three converged solves sit at 40.7%, 46.8% and 54.7% -- indistinguishable
from A's 42-54%. The 75% threshold used in the production sweep is a large-dot criterion; on a
2.5 nm island only the h-trend is evidence. (The random-vector baseline here is 0.9%, so even
40% is forty times baseline.)

Uniform values are taken from whichever material keeps the result elliptic; see CASES.

Run:  python -u scripts/gamma3_diagnostic.py [--base 2.5] [--AR 1] [--pad 4]
      python -u scripts/gamma3_diagnostic.py --only A --scheme fe        # test a candidate fix

`--scheme` selects the cross-term discretization (see `kp_confined.GridOperators`). Case A under
a candidate scheme is the acceptance test for that scheme: it is the ONLY case that fails under
'central', so a fix is one that makes case A converge without disturbing B/C/D.

**Use `-u`, and note every print here passes flush=True.** A first attempt at this run was
launched without either, redirected to a log, and spent an hour producing an empty file with no
way to tell progress from a hang -- the exact failure CLAUDE.md warns about. Results are also
appended to `_gamma3_diag.json` after every solve, so an interrupted run keeps what it has.
"""
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import heterostructure as hs
import kp_confined as kpc
import kp_pryor as kp
import materials as mt

T = 80.0
#: Coarse to fine. 0.25 is deliberately omitted -- the signature is the TREND, which is already
#: unambiguous over 0.50 -> 0.32 (the original measurement drifted 430 -> 299 meV over exactly
#: that span), and the finest point costs more than the other three together.
SPACINGS = (0.50, 0.40, 0.32)
ARTIFACT = 0.046                      # eV.nm^2, the fitted 1/h^2 interface-mode energy
TOL, NSTATES = 1e-6, 3                # a diagnostic reads trends; it does not need 1e-8
OUT = '_gamma3_diag.json'

#: (label, fields to flatten, where to take the uniform value from).
#:
#: **The source matters, and getting it wrong invalidates the control.** The first version of
#: this script took every uniform value from the dot. For gamma3 that puts InSb's 16.5 into the
#: InAs matrix, where gamma1 = 20.0, so gamma1 - 2*gamma3 = -13: the six-band Luttinger operator
#: stops being elliptic and the solve returned sixteen states at +2.33 eV with 0% of their
#: density in the island -- a completely different pathology, and useless as a control for the
#: one under investigation. Taking gamma3 from the MATRIX (9.2) keeps both materials elliptic
#: (InSb 34.8 - 18.4 = +16.4, InAs 20.0 - 18.4 = +1.6). `check_elliptic` now enforces this.
#:
#: Cases C and D flatten all three onto one material's own set, which is self-consistent by
#: construction and needs no such care.
CASES = (
    ('A full', (), None),
    ('B uniform g3', ('gamma3',), 'matrix'),
    ('C uniform g1,g2,g3 (InSb)', ('gamma1', 'gamma2', 'gamma3'), 'dot'),
    ('D uniform g1,g2,g3 (InAs)', ('gamma1', 'gamma2', 'gamma3'), 'matrix'),
)


def check_elliptic(fields, label):
    """Both Luttinger inequalities, at every grid point, before any solve.

    The six-band Hamiltonian needs gamma1 - 2*gamma2 > 0 and gamma1 - 2*gamma3 > 0 for the
    HH and LH branches to curve the same way. Violate either and the operator admits modes whose
    energy is limited only by the grid -- they look like converged eigenpairs and they are not
    states at all.

    Worth knowing even when it passes: these materials sit CLOSE to the boundary. gamma1 - 2*gamma3
    is +1.8 for InSb and +1.6 for InAs, i.e. 5.2% and 8.0% of gamma1, against 16% for GaAs. Linear
    averaging across the interface preserves it (+1.65 to +1.75 at every mixture), so plain
    averaging is not what breaks it -- but there is not much room.
    """
    g1, g2, g3 = fields['gamma1'], fields['gamma2'], fields['gamma3']
    for name, q in (('gamma1-2*gamma2', g1 - 2 * g2), ('gamma1-2*gamma3', g1 - 2 * g3)):
        if q.min() <= 0:
            raise ValueError(
                f"{label}: {name} = {q.min():.2f} <= 0 at {int((q <= 0).sum())} point(s). "
                "This parameter combination is not elliptic and the solve would return "
                "grid-limited modes, not states. Choose the uniform value from the other "
                "material.")


def solve(env, uniform, source='dot', label='', k=NSTATES, scheme='central'):
    """Six-band hole solve with `uniform` fields flattened to one material's value."""
    m, h = env['mask'], env['h']
    n = m.size
    ops = kpc.GridOperators(m.shape, h, periodic=False, scheme=scheme)
    fields = kp.material_fields(m, mt.kp_params(env['dot']), mt.kp_params(env['matrix']),
                                env['Ev'], env['Ec'], n_bands=6)
    for name in uniform:
        # Read the value back out of the FIELD rather than from kp_params. material_fields
        # renames (gamma1L -> gamma1) and may transform, so taking it from the field is the only
        # way to be sure the uniform value is exactly what the stepped field holds there.
        where = m if source == 'dot' else ~m
        vals = fields[name][where]
        assert vals.ptp() < 1e-12, f"{name} is not constant in the {source}"
        fields[name] = np.full(m.shape, float(vals.flat[0]))
    check_elliptic(fields, label or str(uniform))

    H = kp.confined_hamiltonian(ops, fields, n_bands=6, strain=env['strain'])
    nb = H.shape[0] // n
    t0 = time.time()
    E, V = spla.eigsh(H, k=min(2 * k, H.shape[0] - 2), which='LA', tol=TOL, maxiter=20000)
    order = np.argsort(-E)
    E, V = E[order], V[:, order]
    loc = np.array([float((np.abs(V[:, j].reshape(nb, n)) ** 2).sum(axis=0).reshape(m.shape)[m]
                          .sum() / (np.abs(V[:, j]) ** 2).sum()) for j in range(V.shape[1])])

    top = hs.valence_edge_top(env)
    keep = E <= top + 1e-6            # a bound hole lies below the edge that binds it
    if not keep.any():
        return dict(E=None, loc=None, top=top, seconds=time.time() - t0, n_spurious=len(E))
    return dict(E=float(E[keep][0]), loc=float(loc[keep][0]), top=float(top),
                seconds=time.time() - t0, n_spurious=int((~keep).sum()))


if __name__ == '__main__':
    base = float(sys.argv[sys.argv.index('--base') + 1]) if '--base' in sys.argv else 2.5
    AR = float(sys.argv[sys.argv.index('--AR') + 1]) if '--AR' in sys.argv else 1.0
    pad = float(sys.argv[sys.argv.index('--pad') + 1]) if '--pad' in sys.argv else 4.0
    scheme = sys.argv[sys.argv.index('--scheme') + 1] if '--scheme' in sys.argv else 'central'
    only = sys.argv[sys.argv.index('--only') + 1] if '--only' in sys.argv else None
    height = base / AR

    say = lambda s='': print(s, flush=True)

    mt.set_temperature(T)
    say(f"InSb in InAs, base {base:g} nm, AR = d/h = {AR:g} (height {height:g} nm), "
        f"pad {pad:g} nm, T = {T:g} K")
    say("A tight box is defensible: this hole is bound by hundreds of meV and decays in ~1 nm.")
    say("Comparable ACROSS cases: the h-trend and the localisation. NOT the energies.")

    done = json.load(open(OUT)) if os.path.exists(OUT) else {}
    if done:
        say(f"resuming: {len(done)} solve(s) already in {OUT}")

    for label, uniform, source in CASES:
        if only and not label.startswith(only):
            continue
        say(f"\n=== {label}   [cross: {scheme}] " + '=' * max(4, 42 - len(label)))
        say(f"{'h':>6} {'cells':>6} {'unknowns':>10} {'artifact':>9} {'ratio':>6} "
            f"{'v_top':>8} {'E_hole':>9} {'conf':>9} {'loc':>7} {'spur':>5} {'sec':>6}")
        prev = None
        for h in SPACINGS:
            key = f"{label}_{scheme}_b{base:g}_AR{AR:g}_h{h:.3f}_pad{pad:g}"
            if key in done:
                r = done[key]
            else:
                env = hs.build(hs.ellipsoid(base, height), 'InSb', matrix='InAs', h=h,
                               pad=pad, z_pad=pad, use_piezo=False, vol_tol=0.35, verbose=False)
                r = solve(env, uniform, source=source, label=label, scheme=scheme)
                r['unknowns'] = int(env['mask'].size * 6)
                r['Ev_far'] = float(env['Ev_far'])
                r['baseline'] = float(env['mask'].sum() / env['mask'].size)
                done[key] = r
                json.dump(done, open(OUT, 'w'), indent=1)   # persist EVERY solve
            V0 = r['top'] - r['Ev_far']
            art = ARTIFACT / h ** 2
            if r['E'] is None:
                say(f"{h:6.3f} {height/h:6.1f} {r['unknowns']:10,d} {art:9.3f} "
                    f"{art/V0:6.2f} {r['top']:8.4f}   no state below the edge "
                    f"({r['n_spurious']} above)")
                continue
            conf = (r['top'] - r['E']) * 1e3
            d = f"{conf-prev:+7.1f}" if prev is not None else '      -'
            prev = conf
            say(f"{h:6.3f} {height/h:6.1f} {r['unknowns']:10,d} {art:9.3f} "
                f"{art/V0:6.2f} {r['top']:8.4f} {r['E']:+9.4f} {conf:8.1f}m "
                f"{r['loc']*100:6.1f}% {r['n_spurious']:5d} {r['seconds']:6.0f}"
                f"   d(conf) {d}")
        # Random-vector baseline: a state below this is actively avoiding the island.
        say(f"       random-vector localisation baseline {r['baseline']*100:.1f}%")

    say("\nRead the TREND, not the energies. A converging sequence at high localisation is a")
    say("real state; a monotone drift at 40-70% is the artifact. Case C must be clean.")
