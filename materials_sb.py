"""Material parameters for the InAs / In(x)Ga(1-x)Sb antimonide system.

This is a *different* material system from the Pryor InAs/GaAs benchmark, and almost everything
that made that benchmark easy is absent here. There is no reference calculation to check against,
the band alignment is broken-gap rather than type-I, and the parameter set has to be assembled
from a database that is internally inconsistent in two places. So this module is written to make
provenance visible: every number carries a source tag, `audit()` prints them, and the two values
that are known to be suspect are switchable rather than silently chosen.

Where the numbers come from
---------------------------
Almost all of them are read directly out of `database.py` in the sibling `aestimo` checkout
(`C:/Users/molly/code/aestimo`), which is this project's own materials database and therefore a
verifiable provenance chain: the values below were transcribed from it, not recalled. That file
attributes its III-V entries to Vurgaftman, Meyer and Ram-Mohan, "Band parameters for III-V
compound semiconductors and their alloys", J. Appl. Phys. 89, 5815 (2001) -- the standard
compilation for this family. Per the citation policy in `qdsolver_core.py`, table and equation
numbers from that review are not quoted, because the review itself was not fetched and checked
while writing this.

Four parameters the eight-band machinery needs are NOT in that database at all: `C44` for GaSb,
the [111] shear deformation potential `d`, and the piezoelectric constant `e14` for GaSb and
InSb. They are tagged `UNVERIFIED` below and are exposed as ordinary dictionary entries so they
can be overridden. Treat them as knobs, not as data.

Two problems in the source data, neither of them fixed silently
--------------------------------------------------------------
**1. The elastic constants are in two different units.** In `database.py` the InAs entry has
`C11 = 8.329` while GaSb has `C11 = 88.42` and InSb `C11 = 68.47`. aestimo multiplies whatever it
finds by 1e10 to get pascals, so it reads InAs as 83.29 GPa (right) and GaSb as 884 GPa (wrong by
10x). This module stores GPa throughout and applies the factor of 10 to the InAs/GaAs-family
entries only. `audit()` re-derives each value so the correction is visible.

**2. GaSb's valence deformation potential has the wrong sign relative to InAs and InSb.**
In the convention aestimo actually uses (`E_v -= a_v * Tr(eps)`, which is also Pryor's), the
database gives a_v = +1.00 for InAs and +0.31 for InSb but **-1.32** for GaSb. A negative a_v
there means the GaSb valence band moves *down* under the compression that moves InAs's *up*,
which is not a real physical difference between two closely related III-V antimonides -- it is a
sign error somewhere upstream. It matters a lot here, because the hole is confined in the
InGaSb and a_v is what sets the depth of its well.

`GASB_AV_CONVENTION` selects what to do about it, and there is no default that is simply
"correct":

  'database'  -- use -1.32 as stored. Reproduces what aestimo would compute. Hole confinement
                 in the dot comes out qualitatively different, so this is worth running once.
  'signfixed' -- use +1.32, i.e. the same sign as InAs and InSb, keeping the magnitude.
  'vurgaftman'-- use +0.80, the magnitude commonly quoted for GaSb in this convention.
                 UNVERIFIED: not present in database.py and not checked against the review.

The module default is 'signfixed': it is the minimal change that removes the internal
inconsistency without importing an unverified magnitude. Every function that consumes a_v takes
the material dicts as arguments, so switching is a one-line change in a notebook.

Band alignment
--------------
Unlike the Pryor benchmark, which pins its zero to the unstrained GaAs valence edge and quotes a
single valence-band offset, this system is handled through *absolute* valence-band energies
(`VBO`, the `AVb_E` field in database.py: InAs -6.67 eV, GaSb -6.25 eV, InSb -6.09 eV, all on the
review's common scale). Offsets are then differences, which is the only way to get a three-way
InAs/GaSb/InSb alloy alignment right.

That alignment is the whole point of the system. Taking those numbers at face value, unstrained:

    InAs   E_v = -6.67   E_c = -6.27
    GaSb   E_v = -6.25   E_c = -5.52
    InSb   E_v = -6.09   E_c = -5.92

so the In(x)Ga(1-x)Sb *valence* edge lies above the InAs *conduction* edge for every composition
-- by ~20 meV at x = 0 and ~180 meV at x = 1. This is the broken-gap (type-III) alignment InAs/
GaSb is known for, and it means electrons sit in the InAs matrix while holes sit in the dot,
spatially separated. Strain then moves both edges by a lot, because the misfit runs from +0.6%
at x = 0 to +6.9% at x = 1. `alignment_table()` prints where a given composition actually lands.

The alloy
---------
In(x)Ga(1-x)Sb, x = InSb fraction, matching database.py's `InGaSb` entry (Material1 = InSb,
Material2 = GaSb). Everything is interpolated linearly except the gap, which takes that entry's
bowing parameter, 0.415 eV. Its `delta_bowing_param` is 0.0, so the split-off is linear too.

That entry also carries `AVb_E = -2.1`, which this module deliberately does **not** use. The
parent materials' `AVb_E` values are absolute energies near -6 eV, so -2.1 cannot be an absolute
energy for the alloy and must be intended as a bowing parameter -- but a 2.1 eV valence-band
bowing is implausibly large, and nothing in the file says which it is. Linear VBO interpolation
is used instead, and `INGASB_VBO_BOWING` exposes the choice.
"""
import numpy as np

#: See the module docstring. One of 'database', 'signfixed', 'vurgaftman'.
GASB_AV_CONVENTION = 'signfixed'

#: Valence-band-offset bowing for In(x)Ga(1-x)Sb, eV. Zero = linear interpolation, which is what
#: this module uses; see the docstring for why database.py's -2.1 is not adopted.
INGASB_VBO_BOWING = 0.0

_GASB_AV = {'database': -1.32, 'signfixed': 1.32, 'vurgaftman': 0.80}

#: Where each value came from. 'db' = transcribed from aestimo's database.py; 'db/x10' = same,
#: with the documented factor-of-10 unit correction; 'derived' = computed here from db values;
#: 'UNVERIFIED' = not in the database and not checked against a source -- a knob, not a datum.
PROVENANCE = {
    'gamma1L': 'db (GA1)', 'gamma2L': 'db (GA2)', 'gamma3L': 'db (GA3)',
    'Eg': 'db', 'delta_so': 'db (delta)', 'Ep': 'db',
    'a_c': 'db (Ac)', 'a_v': 'db (Av), sign convention -- see module docstring',
    'b': 'db (B)', 'd': 'UNVERIFIED -- not in db',
    'e14': 'InAs: Pryor 1998 Table I. GaSb/InSb: UNVERIFIED -- not in db',
    'eps_R': 'db (epsilonStatic)',
    'C11': 'db, GPa', 'C12': 'db, GPa', 'C44': 'InAs: Pryor Table I. GaSb: UNVERIFIED',
    'a0': 'db', 'VBO': 'db (AVb_E), absolute valence-band energy',
    'F': 'db, Kane remote-band parameter -- used to DERIVE m_e, see electron_mass()',
}

#: The three binaries. Energies eV, elastic constants GPa, lattice constants Angstrom.
#: `VBO` is the ABSOLUTE valence-band energy; band offsets are differences of it.
SB_MATERIALS = {
    'InAs': dict(
        gamma1L=20.4, gamma2L=8.3, gamma3L=9.1,
        Eg=0.400, delta_so=0.38, Ep=21.5,
        a_c=-5.08, a_v=1.00, b=-1.8,
        d=-3.6,                       # UNVERIFIED for this parameter set; Pryor's Table I value
        e14=0.045, eps_R=15.15, F=-2.9,
        C11=83.29, C12=45.26, C44=39.59,      # db values x10 -- see docstring
        a0=6.0583, VBO=-6.67,
    ),
    'GaSb': dict(
        gamma1L=13.4, gamma2L=4.7, gamma3L=6.0,
        Eg=0.726, delta_so=0.76, Ep=27.0,
        a_c=-9.33, a_v=None, b=-2.0,          # a_v filled in below from GASB_AV_CONVENTION
        d=-4.6,                               # UNVERIFIED
        e14=-0.126, eps_R=15.69, F=-1.63,     # e14 UNVERIFIED (sign conventions vary)
        C11=88.42, C12=40.26, C44=43.2,       # C44 UNVERIFIED -- not in db
        a0=6.0959, VBO=-6.25,
    ),
    'InSb': dict(
        gamma1L=34.8, gamma2L=15.5, gamma3L=16.5,
        Eg=0.174, delta_so=0.81, Ep=23.3,
        a_c=-6.04, a_v=0.31, b=-2.0,
        d=-5.0,                               # UNVERIFIED
        e14=-0.071, eps_R=17.5, F=-0.23,      # e14 UNVERIFIED
        C11=68.47, C12=37.35, C44=31.11,
        a0=6.4794, VBO=-6.09,
    ),
}

SB_MATERIALS['GaSb']['a_v'] = _GASB_AV[GASB_AV_CONVENTION]

#: Gap bowing for In(x)Ga(1-x)Sb, from database.py's InGaSb entry.
INGASB_EG_BOWING = 0.415

#: Keys interpolated linearly in x. Everything the eight-band Hamiltonian and the elasticity
#: solver need, except Eg (bowed) and VBO (bowing exposed separately).
_LINEAR_KEYS = ('gamma1L', 'gamma2L', 'gamma3L', 'delta_so', 'Ep', 'a_c', 'a_v', 'b', 'd',
                'e14', 'eps_R', 'C11', 'C12', 'C44', 'a0', 'F')


def ingasb(x, eg_bowing=INGASB_EG_BOWING, vbo_bowing=INGASB_VBO_BOWING):
    """In(x)Ga(1-x)Sb parameters. `x` is the InSb fraction, so x = 0 is GaSb and x = 1 is InSb.

    Linear in everything but the gap. Vegard's law for the lattice constant is what makes the
    composition a strain knob here: a0 runs 6.0959 -> 6.4794 Angstrom, i.e. the misfit against
    InAs runs +0.62% -> +6.95%.
    """
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"composition x must be in [0, 1], got {x}")
    a, b_ = SB_MATERIALS['InSb'], SB_MATERIALS['GaSb']
    out = {k: x * a[k] + (1 - x) * b_[k] for k in _LINEAR_KEYS}
    out['Eg'] = x * a['Eg'] + (1 - x) * b_['Eg'] - eg_bowing * x * (1 - x)
    out['VBO'] = x * a['VBO'] + (1 - x) * b_['VBO'] - vbo_bowing * x * (1 - x)
    out['x'] = x
    out['name'] = f"In{x:.2f}Ga{1-x:.2f}Sb"
    return out


def electron_mass(mat):
    """Conduction-band effective mass in units of m0, DERIVED from the k.p parameters:

        m0/m_e = 1 + 2F + Ep * (Eg + 2*delta/3) / (Eg * (Eg + delta))

    the standard Kane expression relating the eight-band parameter set to the band-edge mass,
    with F absorbing the remote bands.

    Derived rather than tabulated on purpose. `database.py` lists `m_e = 0.4` for InAs, which is
    an order of magnitude off (InAs is ~0.026) and is evidently a typo; GaSb 0.039 and InSb
    0.0135 there are fine. Rather than use two good values and one bad one, all three come from
    the same formula, which also guarantees the single-band mass is consistent with the
    eight-band Hamiltonian built from the same Ep, Eg and delta.

    Checked against the accepted values: InAs 0.025 vs 0.026, GaSb 0.035 vs 0.039, InSb 0.010 vs
    0.0135. The InSb discrepancy is the expected one -- at a 0.17 eV gap the expression is very
    sensitive to Ep and F.
    """
    Eg, D, Ep, F = mat['Eg'], mat['delta_so'], mat['Ep'], mat['F']
    return 1.0 / (1.0 + 2.0 * F + Ep * (Eg + 2.0 * D / 3.0) / (Eg * (Eg + D)))


def elastic(mat):
    """(C11, C12, C44) in GPa, in the order elasticity_fd and strain_fourier expect."""
    return (mat['C11'], mat['C12'], mat['C44'])


def misfit(dot, matrix):
    """Lattice-mismatch eigenstrain of `dot` embedded in `matrix`.

    Same definition as qdsolver_core.eigenstrain: (a_matrix - a_dot) / a_dot, negative when the
    dot must be compressed to fit. Every InGaSb composition is larger than InAs, so this is
    negative throughout and the dot is under compression.
    """
    return (matrix['a0'] - dot['a0']) / dot['a0']


def band_edge_fields(inside_mask, trace_strain, dot, matrix, zero='matrix_vb'):
    """Strained conduction and valence band-edge fields (eV) for a dot in a matrix.

    Convention, matching `pryor1998.band_edge_fields` and aestimo's own `Strain_and_Masses`:

        E_c = E_c(unstrained) + a_c * Tr(eps)
        E_v = E_v(unstrained) - a_v * Tr(eps)

    with a_v positive for a material whose valence band rises under compression. This is the sign
    convention `kp_pryor` expects, so the result can be handed straight to `material_fields`.
    Note it is the OPPOSITE sign on a_v from the Van de Walle convention used by
    `qdsolver_core.MATERIALS`; mixing them moves the valence band the wrong way.

    Only the hydrostatic part is applied here. The shear part (the Bir-Pikus q, r, s terms) is
    applied inside the Hamiltonian builder from the full strain tensor, so passing these fields
    on with `include_hydrostatic=True` would count the hydrostatic shift twice.

    `zero` sets the energy origin: 'matrix_vb' puts zero at the unstrained matrix valence edge
    (the analogue of Pryor's choice, and what the plots here use), 'absolute' keeps the
    database's absolute scale.
    """
    ref = matrix['VBO'] if zero == 'matrix_vb' else 0.0

    Ev0 = np.where(inside_mask, dot['VBO'] - ref, matrix['VBO'] - ref)
    Ec0 = Ev0 + np.where(inside_mask, dot['Eg'], matrix['Eg'])

    a_c = np.where(inside_mask, dot['a_c'], matrix['a_c'])
    a_v = np.where(inside_mask, dot['a_v'], matrix['a_v'])

    return Ec0 + a_c * trace_strain, Ev0 - a_v * trace_strain


def kp_params(mat):
    """A copy of `mat` carrying the keys `kp_pryor.material_fields` reads.

    That function was written against `pryor1998.PRYOR_TABLE_I`, which uses the same key names
    used here, so this is mostly an assertion that nothing is missing -- it fails loudly rather
    than letting a KeyError surface from inside the Hamiltonian assembly.
    """
    needed = ('gamma1L', 'gamma2L', 'gamma3L', 'Eg', 'delta_so', 'Ep',
              'a_c', 'a_v', 'b', 'd')
    missing = [k for k in needed if mat.get(k) is None]
    if missing:
        raise KeyError(f"{mat.get('name', '?')} is missing {missing}")
    return dict(mat)


def alignment_table(compositions=(0.0, 0.2, 0.35, 0.5, 0.75, 1.0), matrix='InAs',
                    trace_strain=None):
    """Print where each composition puts the band edges, unstrained and (optionally) strained.

    `trace_strain` is a single representative Tr(eps) inside the dot -- pass the mean from a real
    solve to see where the edges actually land, or leave it None for the unstrained alignment.
    The last column is the quantity that decides the whole character of the system: the
    InGaSb valence edge minus the InAs conduction edge. Positive means broken gap.
    """
    m = SB_MATERIALS[matrix]
    ref = m['VBO']
    Ec_m, Ev_m = m['VBO'] - ref + m['Eg'], m['VBO'] - ref

    print(f"energy zero = unstrained {matrix} valence edge; "
          f"{matrix}: E_v = {Ev_m:.3f}, E_c = {Ec_m:.3f} eV")
    if trace_strain is not None:
        print(f"dot strained with Tr(eps) = {trace_strain:+.4f} (matrix taken unstrained)")
    head = f"{'x':>5} {'name':>14} {'misfit':>8} {'Eg':>7} {'E_v':>8} {'E_c':>8} {'E_v(dot)-E_c(InAs)':>20}"
    print(head)
    for x in compositions:
        d = ingasb(x)
        tr = 0.0 if trace_strain is None else trace_strain
        Ev = d['VBO'] - ref - d['a_v'] * tr
        Ec = (d['VBO'] - ref + d['Eg']) + d['a_c'] * tr
        print(f"{x:>5.2f} {d['name']:>14} {misfit(d, m)*100:>7.2f}% {d['Eg']:>7.3f} "
              f"{Ev:>8.3f} {Ec:>8.3f} {Ev - Ec_m:>+20.3f}")
    print("\npositive last column = broken gap: the dot's valence edge lies above the "
          f"{matrix} conduction edge,\nso electrons stay in the matrix and holes in the dot.")


def audit():
    """Print every parameter with its provenance, and re-derive the two known data problems."""
    print(f"GASB_AV_CONVENTION = {GASB_AV_CONVENTION!r}  ->  GaSb a_v = "
          f"{SB_MATERIALS['GaSb']['a_v']:+.2f} eV")
    print(f"INGASB_VBO_BOWING  = {INGASB_VBO_BOWING} eV   (0 = linear VBO interpolation)")
    print(f"INGASB_EG_BOWING   = {INGASB_EG_BOWING} eV\n")

    keys = [k for k in PROVENANCE if k != 'x']
    print(f"{'parameter':>10} {'InAs':>9} {'GaSb':>9} {'InSb':>9}   provenance")
    for k in keys:
        row = ' '.join(f"{SB_MATERIALS[m][k]:>9.3f}" for m in ('InAs', 'GaSb', 'InSb'))
        print(f"{k:>10} {row}   {PROVENANCE[k]}")

    print("\nknown problems in the source data:")
    print("  1. units: database.py stores InAs C11 = 8.329 (1e11 dyne/cm^2) but GaSb C11 = 88.42")
    print("     and InSb C11 = 68.47 (GPa). aestimo multiplies all three by 1e10, so it reads")
    print("     GaSb and InSb 10x too stiff. This module stores GPa and applies x10 to InAs only.")
    print(f"  2. sign: a_v is {SB_MATERIALS['InAs']['a_v']:+.2f} (InAs) and "
          f"{SB_MATERIALS['InSb']['a_v']:+.2f} (InSb) but -1.32 for GaSb in the database.")
    print("     A negative a_v moves the valence band the opposite way under compression, which")
    print("     is not a real difference between these materials. Currently using "
          f"{SB_MATERIALS['GaSb']['a_v']:+.2f}.")
    print("\nanything tagged UNVERIFIED is a knob, not a datum -- vary it before trusting a")
    print("result that depends on it.")
