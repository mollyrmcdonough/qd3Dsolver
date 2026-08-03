"""Band parameters for GaAs, InAs, GaSb, InSb and the In(x)Ga(1-x)Sb / InAs(1-x)Sb(x) alloys.

This is the single source of material data for the project. It replaces three overlapping and
mutually inconsistent tables that grew up here:

  `materials_sb.SB_MATERIALS`  -- three antimonide binaries, correct convention, hand-audited.
                                  Now a thin wrapper over this module; see that file.
  `qdsolver_core.MATERIALS`    -- GaAs/InAs/InSb in the OPPOSITE a_v sign convention (Van de
                                  Walle: `Av` positive). Still imported by the Pryor InAs/GaAs
                                  reproductions, which are validated against it, so it is left
                                  alone rather than migrated. Do not mix the two: see below.
  `aestimo_database.py`        -- unaudited, and its C11/C12 are in two different units. Nothing
                                  imports it and nothing should.

Where the numbers come from
---------------------------
Every value below is transcribed from `bandparameters_vurgaftman2001.xlsx` and
`bowingparameters_vurgaftman_2001.xlsx` in the repository root, which hold the band-parameter
and bowing tables of I. Vurgaftman, J. R. Meyer and L. R. Ram-Mohan, "Band parameters for III-V
compound semiconductors and their alloys", J. Appl. Phys. 89, 5815 (2001). Per the citation
policy in `qdsolver_core.py`, table numbers from that review are not quoted, because the review
itself was not fetched; the workbooks are the verifiable provenance chain.

Two rows of the band-parameter workbook -- `e_14` and `eps_r` -- are additions to the sheet
rather than part of the review's own band-parameter tables, which do not cover piezoelectric or
dielectric constants. They are tagged accordingly in `PROVENANCE` and there is a sign caveat on
e14 immediately below.

THE e14 SIGN. This module carries e14 NEGATIVE for all four binaries, as supplied. Pryor's own
Table I (transcribed in `pryor1998.PRYOR_TABLE_I`, read from the arXiv preprint) carries the
same magnitudes with the OPPOSITE sign: GaAs +0.159, InAs +0.045. Only one independent constant
exists in zincblende, so a global sign flip flips the piezoelectric potential everywhere and
swaps which side of the dot each carrier is pushed toward. It does not change |phi|, the C4
antisymmetry, or anything computed with `use_piezo=False`. `pryor1998.PRYOR_TABLE_I` is a
benchmark FIXTURE -- it reproduces a published figure and must keep its own values -- so the two
deliberately disagree. If you compare a piezo result here against a Pryor figure, that sign is
the first thing to check.

The sign convention for the deformation potentials
--------------------------------------------------
Vurgaftman's, which is not the most common one in the literature. The review states it and gives
the reason: the gap must OPEN under hydrostatic compression, i.e. dEg = a * Tr(eps) must be
positive when Tr(eps) < 0, which forces

    a_gap = a_c + a_v  <  0

and therefore a NEGATIVE a_v for all four binaries here (GaAs -1.16, InAs -1.00, GaSb -0.80,
InSb -0.36). The review notes that "our sign convention for a_v is different from many other
works found in the literature", where a_v is usually quoted positive and the gap potential
formed as a_c - a_v.

`band_edge_fields` applies

    E_c = E_c(unstrained) + a_c * Tr(eps)
    E_v = E_v(unstrained) - a_v * Tr(eps)

so with these signs, compression pushes E_c up and E_v down and the gap opens by |a_gap * Tr|.
`a_gap()` returns the convention-independent combination and is the thing to check against any
external source. Converting to or from the Van de Walle form used by `qdsolver_core.MATERIALS`
is exactly `a_v -> -a_v`; mixing them unnoticed moves the valence band the wrong way, which is a
~100 meV error that looks entirely plausible.

The energy scale
----------------
`VBO` is the unstrained valence-band edge on the review's own common scale, which puts InSb at
zero: GaAs -0.80, InAs -0.59, GaSb -0.03, InSb 0.00 eV. Only DIFFERENCES of it are physical, and
every consumer here takes a difference, so the choice of zero is free; the review's own scale is
used rather than an anchored absolute one so the numbers can be checked against the source by
eye. (`materials_sb` previously carried the same offsets anchored near -6 eV, inherited from
aestimo's `AVb_E`; the shift is uniform and changes nothing.)

Those offsets are what make this family interesting. Unstrained at 0 K, relative to InAs:

    InAs   E_v =  0.00   E_c = 0.417        <- the usual matrix
    GaSb   E_v = +0.56   E_c = 1.372        <- E_v(GaSb) is 0.14 eV ABOVE E_c(InAs)
    InSb   E_v = +0.59   E_c = 0.825        <- E_v(InSb) is 0.17 eV above E_c(InAs)
    GaAs   E_v = -0.21   E_c = 1.309

so InAs/GaSb and InAs/InSb are broken gap (type III) before strain does anything, while
InAs/GaAs is ordinary type I. `alignment_table()` prints this for any pair or alloy.

Temperature
-----------
Two quantities are temperature dependent and both are handled:

    a0(T)  = a0(300 K) + da/dT * (T - 300)          linear, from the workbook
    Eg(T)  = Eg(0) - alpha T^2 / (T + beta)         Varshni, from the workbook

`set_temperature(T)` updates `MATERIALS` in place and is the ONLY supported way to change T --
`MATERIALS` is populated at import, so assigning to the module constant afterwards is a silent
no-op. The default is 0 K, which is what the rest of the parameter set is: Ep, F and the
deformation potentials are all 0 K quantities, and the Kane expression in `electron_mass`
reproduces the review's own tabulated masses to the digit at 0 K and not at 300 K.

The lattice constant matters more than it looks. It is the ONLY thing that sets the misfit, and
the misfit drives every strain result. Between 0 K and 300 K the InSb-in-InAs misfit moves from
-6.475% to -6.499% -- 0.4% of itself, worth a fraction of a meV on a dot gap, but it is the
difference between a self-consistent parameter set and one that mixes temperatures. That mixing
is a mistake this project already made once, in the other direction: the GaSb offset was derived
from a 0 K table while the gaps were 300 K.

`eps_R` is a 300 K value and is NOT varied with temperature; it enters only the piezoelectric
potential, whose other inputs are far less certain than its temperature dependence.

The alloys
----------
Two, both parameterised by x = the InSb fraction so the endpoints are unambiguous:

    'InGaSb'   In(x)Ga(1-x)Sb    x = 0 -> GaSb,  x = 1 -> InSb
    'InAsSb'   InAs(1-x)Sb(x)    x = 0 -> InAs,  x = 1 -> InSb

Interpolation is P(x) = x*P_A + (1-x)*P_B - C*x*(1-x) with A the x = 1 parent, and C the bowing
parameter from the bowing workbook. C is zero -- i.e. the interpolation is linear -- for every
key the workbook does not list, which includes the Luttinger parameters, Ep (for InAsSb), the
deformation potentials, the elastic constants, VBO and the lattice constant. Linear a0 is
Vegard's law, and it is what makes composition a strain knob: In(x)Ga(1-x)Sb on InAs runs from
+0.62% misfit at x = 0 to +6.95% at x = 1.

Two of the listed bowings are large and are not typos. InAs(1-x)Sb(x) has C(Eg) = 0.67 eV, which
drives the gap to a minimum of 0.159 eV at x = 0.5 -- BELOW either parent, and the reason that
alloy is used for long-wavelength infrared. It also has C(delta_so) = 1.2 eV, which pulls the
split-off from a 0.6 eV linear value down to 0.30 eV at midpoint.

What is NOT here
----------------
- Anything but the four binaries and those two alloys. The other two ternaries this family
  admits, In(x)Ga(1-x)As and GaAs(1-x)Sb(x), are absent because their bowing parameters are not
  in the workbook. Adding them is a table entry, not code.
- Quaternaries.
- X and L valley DEFORMATION potentials. The valley gaps themselves are carried (`Eg_X`,
  `Eg_L`, and `lowest_valley()`), but they are not strain shifted, so the direct/indirect
  crossover under strain is not modelled. All four binaries are direct at Gamma unstrained by a
  wide margin, so this only matters if you go to alloys well outside this set.
- Temperature dependence of anything except a0 and the gaps.
"""
import numpy as np

#: The two workbooks every number below was transcribed from, relative to the repository root.
WORKBOOKS = ('bandparameters_vurgaftman2001.xlsx', 'bowingparameters_vurgaftman_2001.xlsx')

#: The binaries, in the workbook's column order.
BINARIES = ('GaAs', 'InAs', 'GaSb', 'InSb')

#: Lattice constant as (a at 300 K, da/dT) in Angstrom and Angstrom/K. The workbook writes these
#: as expressions, e.g. "5.65325+3.88E-5(T-300)"; they are split here so a0 can be evaluated.
_LATTICE = {
    'GaAs': (5.65325, 3.88e-5),
    'InAs': (6.05830, 2.74e-5),
    'GaSb': (6.09590, 4.72e-5),
    'InSb': (6.47940, 3.48e-5),
}

#: Varshni parameters per valley: (Eg at 0 K in eV, alpha in eV/K, beta in K). The workbook
#: quotes alpha in meV/K; it is converted here. `None` where the workbook has no entry -- the
#: InSb X and L valleys -- in which case `varshni` returns the 0 K gap at every temperature.
#: Only the Gamma row feeds any calculation; X and L are carried for `lowest_valley`.
_VARSHNI = {
    'GaAs': {'G': (1.519, 0.5405e-3, 204.0),
             'X': (1.981, 0.4600e-3, 204.0),
             'L': (1.815, 0.6050e-3, 204.0)},
    'InAs': {'G': (0.417, 0.2760e-3, 93.0),
             'X': (1.433, 0.2760e-3, 93.0),
             'L': (1.133, 0.2760e-3, 93.0)},
    'GaSb': {'G': (0.812, 0.4170e-3, 140.0),
             'X': (1.141, 0.4750e-3, 94.0),
             'L': (0.875, 0.5970e-3, 140.0)},
    'InSb': {'G': (0.235, 0.3200e-3, 170.0),
             'X': (0.630, None, None),
             'L': (0.930, None, None)},
}

#: Everything that is neither temperature dependent nor a Varshni input. Energies eV, elastic
#: constants GPa, masses in units of m0, e14 in C/m^2. `Eg`, `Eg_X`, `Eg_L` and `a0` are absent
#: on purpose: they are filled in by `set_temperature`, which runs at import.
_FIXED = {
    'GaAs': dict(delta_so=0.341, m_e=0.0670, m_so=0.172, Ep=28.8, F=-1.94,
                 gamma1L=6.98, gamma2L=2.06, gamma3L=2.93,
                 VBO=-0.80, a_c=-7.17, a_v=-1.16, b=-2.0, d=-4.8,
                 C11=122.10, C12=56.60, C44=60.00, e14=-0.160, eps_R=12.9),
    'InAs': dict(delta_so=0.390, m_e=0.0260, m_so=0.140, Ep=21.5, F=-2.90,
                 gamma1L=20.0, gamma2L=8.50, gamma3L=9.20,
                 VBO=-0.59, a_c=-5.08, a_v=-1.00, b=-1.8, d=-3.6,
                 C11=83.29, C12=45.26, C44=39.59, e14=-0.045, eps_R=15.15),
    'GaSb': dict(delta_so=0.760, m_e=0.0390, m_so=0.120, Ep=27.0, F=-1.63,
                 gamma1L=13.4, gamma2L=4.70, gamma3L=6.00,
                 VBO=-0.03, a_c=-7.50, a_v=-0.80, b=-2.0, d=-4.7,
                 C11=88.42, C12=40.26, C44=43.22, e14=-0.130, eps_R=15.7),
    'InSb': dict(delta_so=0.810, m_e=0.0135, m_so=0.110, Ep=23.3, F=-0.23,
                 gamma1L=34.8, gamma2L=15.5, gamma3L=16.5,
                 VBO=0.00, a_c=-6.94, a_v=-0.36, b=-2.0, d=-4.7,
                 C11=68.47, C12=37.35, C44=31.11, e14=-0.070, eps_R=16.8),
}

#: Keys interpolated across an alloy composition. Any key here that is missing from a given
#: alloy's `bowing` dict is interpolated linearly.
_INTERPOLATED = ('Eg', 'Eg_X', 'Eg_L', 'delta_so', 'm_e', 'm_so', 'Ep', 'F',
                 'gamma1L', 'gamma2L', 'gamma3L', 'VBO', 'a_c', 'a_v', 'b', 'd',
                 'C11', 'C12', 'C44', 'a0', 'e14', 'eps_R')

#: The alloys. `parents` is (A, B) where the composition x is the fraction of A, so x = 1 gives A
#: and x = 0 gives B. `bowing` holds C in P(x) = x*P_A + (1-x)*P_B - C*x*(1-x); keys absent from
#: it interpolate linearly.
#:
#: The bowing workbook also lists an m*_lh bowing of 0.011 for In(x)Ga(1-x)Sb. It is recorded in
#: `unused` and is deliberately not applied: the hole masses here come from the Luttinger
#: parameters, which the workbook gives no bowing for, so adopting it would make the light hole
#: inconsistent with the heavy hole and with the eight-band Hamiltonian built from the same
#: gammas.
_BOWING = {
    'InGaSb': dict(
        parents=('InSb', 'GaSb'),
        formula=lambda x: f"In{x:.2f}Ga{1 - x:.2f}Sb",
        bowing=dict(Eg=0.415, Eg_X=0.33, Eg_L=0.40, delta_so=0.10, m_e=0.0092, F=-6.84),
        unused=dict(m_lh=0.011),
    ),
    'InAsSb': dict(
        parents=('InSb', 'InAs'),
        formula=lambda x: f"InAs{1 - x:.2f}Sb{x:.2f}",
        bowing=dict(Eg=0.67, Eg_X=0.60, Eg_L=0.60, delta_so=1.20, m_e=0.035),
        unused={},
    ),
}

#: The alloy names, for iteration and for error messages.
ALLOYS = tuple(_BOWING)

#: Where each value came from. Read by `audit()` and by `scripts/dump_sb_parameters.py`.
PROVENANCE = {
    'Eg': 'workbook, Varshni from the 0 K gap -- see set_temperature',
    'Eg_X': 'workbook (carried, not strain shifted)',
    'Eg_L': 'workbook (carried, not strain shifted)',
    'delta_so': 'workbook',
    'm_e': 'workbook, tabulated. electron_mass() DERIVES it instead -- see that docstring',
    'm_so': 'workbook (carried, unused: the split-off mass follows from the 8-band parameters)',
    'Ep': 'workbook',
    'F': 'workbook',
    'gamma1L': 'workbook',
    'gamma2L': 'workbook',
    'gamma3L': 'workbook',
    'VBO': "workbook, on the review's own scale (InSb = 0). Only differences are physical",
    'a_c': 'workbook',
    'a_v': "workbook. NEGATIVE for all four -- the review's convention, see module docstring",
    'b': 'workbook',
    'd': 'workbook',
    'C11': 'workbook, GPa',
    'C12': 'workbook, GPa',
    'C44': 'workbook, GPa',
    'a0': 'workbook, a(300 K) + da/dT (T - 300); temperature dependent',
    'e14': 'workbook -- ADDED to the sheet, not from the review. Sign differs from Pryor 1998',
    'eps_R': 'workbook -- ADDED to the sheet, not from the review. 300 K value, not varied',
}

#: Temperature the table is currently evaluated at, K. Set through `set_temperature`.
TEMPERATURE = None

#: The binaries. Populated by `set_temperature`, which runs at import; every consumer reads this
#: dict, so mutating it in place is what makes a temperature switch take effect.
MATERIALS = {name: dict(vals, name=name) for name, vals in _FIXED.items()}


def varshni(name, valley='G', T=0.0):
    """Eg(T) for one binary and one valley, in eV.

        Eg(T) = Eg(0) - alpha T^2 / (T + beta)

    Returns Eg(0) exactly at T = 0. Where the workbook gives no alpha/beta -- the InSb X and L
    valleys -- the 0 K gap is returned at every temperature rather than raising, since those
    entries exist only to answer "is this material still direct" and a few tens of meV of
    Varshni shift cannot change that answer for InSb, whose Gamma valley is 0.4 eV lower.
    """
    Eg0, alpha, beta = _VARSHNI[name][valley]
    if alpha is None:
        return Eg0
    return Eg0 - alpha * T ** 2 / (T + beta)


def set_temperature(T=0.0):
    """Evaluate the table at temperature `T` (K); returns {name: (Eg, a0)} now in force.

    A function rather than a constant for the usual reason in this project: `MATERIALS` is built
    at import, so reassigning a module-level `TEMPERATURE` afterwards would be a silent no-op --
    the kind that makes a temperature study look like it found nothing. This updates the dicts
    every consumer actually reads, and `alloy()` interpolates from them on each call, so alloys
    follow automatically.

    Changing T is not local. It moves the gaps, which feed the conduction edge, the derived Kane
    mass and the eight-band Hamiltonian; and it moves the lattice constants, which set the
    misfit and therefore the entire strain field. Nothing is cached, so a solve done after this
    call uses the new values and one done before does not.
    """
    T = float(T)
    if T < 0.0:
        raise ValueError(f"temperature must be >= 0 K, got {T}")
    for name, mat in MATERIALS.items():
        a300, dadT = _LATTICE[name]
        mat['a0'] = a300 + dadT * (T - 300.0)
        mat['Eg'] = varshni(name, 'G', T)
        mat['Eg_X'] = varshni(name, 'X', T)
        mat['Eg_L'] = varshni(name, 'L', T)
    global TEMPERATURE
    TEMPERATURE = T
    return {name: (mat['Eg'], mat['a0']) for name, mat in MATERIALS.items()}


set_temperature(0.0)


def alloy(system, x):
    """Parameters for one alloy composition. `x` is the InSb fraction in both alloys.

        alloy('InGaSb', x)   In(x)Ga(1-x)Sb   x = 0 -> GaSb, x = 1 -> InSb
        alloy('InAsSb', x)   InAs(1-x)Sb(x)   x = 0 -> InAs, x = 1 -> InSb

    Returns a dict with the same keys as a binary, plus `system` and `x`, so it can be handed to
    anything that takes a material. At the endpoints it is named for the binary rather than
    "In1.00Ga0.00Sb": every interpolated key collapses to the parent there and the bowing term
    x(1-x) vanishes, so the alloy IS the binary and calling it anything else makes a notebook
    about an InSb dot read as though it were about an alloy.
    """
    if system not in _BOWING:
        raise ValueError(f"unknown alloy {system!r}; choose from {ALLOYS}")
    if not 0.0 <= x <= 1.0:
        raise ValueError(f"composition x must be in [0, 1], got {x}")
    spec = _BOWING[system]
    A, B = (MATERIALS[p] for p in spec['parents'])
    bow = spec['bowing']

    out = {k: x * A[k] + (1 - x) * B[k] - bow.get(k, 0.0) * x * (1 - x) for k in _INTERPOLATED}
    out['system'], out['x'] = system, x
    out['name'] = ({1.0: spec['parents'][0], 0.0: spec['parents'][1]}.get(x)
                   or spec['formula'](x))
    return out


def material(spec, x=None):
    """One entry point for both binaries and alloys.

        material('InAs')            -> the binary
        material('InGaSb', 0.35)    -> the alloy at x = 0.35

    Returns a copy for a binary so a caller cannot mutate `MATERIALS` by accident; `alloy`
    already builds a fresh dict.
    """
    if x is None:
        if spec not in MATERIALS:
            raise ValueError(f"unknown material {spec!r}; binaries are {BINARIES}, "
                             f"alloys are {ALLOYS} and need an x")
        return dict(MATERIALS[spec])
    return alloy(spec, x)


def electron_mass(mat):
    """Conduction-band effective mass in units of m0, DERIVED from the k.p parameters:

        m0/m_e = 1 + 2F + Ep * (Eg + 2*delta/3) / (Eg * (Eg + delta))

    the standard Kane expression relating the eight-band parameter set to the band-edge mass,
    with F absorbing the remote bands.

    Derived rather than read from `mat['m_e']` on purpose, so that the single-band mass is
    guaranteed consistent with the eight-band Hamiltonian assembled from the same Eg, delta and
    Ep. It costs nothing: at 0 K the expression reproduces the workbook's own tabulated masses
    for all four binaries to the digit --

        GaAs 0.0670 vs 0.067,  InAs 0.0260 vs 0.026,
        GaSb 0.0390 vs 0.039,  InSb 0.0135 vs 0.0135

    -- which is a joint check on Eg, delta_so, Ep and F, and is the sharpest evidence that the
    parameter set is internally consistent and is a 0 K set. At 300 K gaps the same expression
    is 24% out on InSb.

    ALLOYS ARE DIFFERENT, and `audit()` prints the discrepancy. Across both alloys and the whole
    composition range this expression comes out 9-16% BELOW the workbook's separately bowed
    m_e -- worst at midcomposition, e.g. In0.50Ga0.50Sb 0.0202 here against 0.0239 tabulated,
    and InAs0.50Sb0.50 0.0092 against 0.0110.

    The cause is that bowing m_e directly and bowing the ingredients of the Kane expression are
    two different prescriptions, and the review's recommended values are not consistent between
    them: both alloys bow Eg strongly downward (C = 0.415 and 0.67 eV), which pulls the derived
    mass down faster than the tabulated m_e bowing (C = 0.0092 and 0.035) does. It is NOT an
    artifact of In(x)Ga(1-x)Sb also carrying an F bowing -- InAs(1-x)Sb(x) has no F bowing and
    shows the same 13% gap at midcomposition.

    This function stays authoritative because the eight-band Hamiltonian is assembled from Ep,
    F, Eg and delta and has no other option; using the tabulated mass for a single-band solve
    while the multiband solve used this one would make the two disagree for no stated reason.
    Treat the spread as the honest uncertainty on an alloy electron mass, and note that it lands
    squarely on the confinement threshold, which goes as 1/m.
    """
    Eg, D, Ep, F = mat['Eg'], mat['delta_so'], mat['Ep'], mat['F']
    return 1.0 / (1.0 + 2.0 * F + Ep * (Eg + 2.0 * D / 3.0) / (Eg * (Eg + D)))


def a_gap(mat):
    """Hydrostatic deformation potential of the GAP, a_c + a_v, in eV.

    This is the convention-independent combination and therefore the one to check against any
    external source. It must be NEGATIVE: compression opens the gap in III-Vs, and the review
    states exactly that as the constraint fixing its a_v sign convention. If this comes out
    positive, an a_v has been imported from a source using the other convention.
    """
    return mat['a_c'] + mat['a_v']


def elastic(mat):
    """(C11, C12, C44) in GPa, in the order `elasticity_fd` and `strain_fourier` expect."""
    return (mat['C11'], mat['C12'], mat['C44'])


def misfit(dot, matrix):
    """Lattice-mismatch eigenstrain of `dot` embedded in `matrix`.

    Same definition as `qdsolver_core.eigenstrain`: (a_matrix - a_dot) / a_dot, negative when
    the dot is larger than the matrix and must be compressed to fit. Temperature enters the
    strain side here and only here, through both lattice constants.
    """
    return (matrix['a0'] - dot['a0']) / dot['a0']


def lowest_valley(mat):
    """(valley, Eg) for the lowest conduction valley of an UNSTRAINED material.

    All four binaries are direct at Gamma by a wide margin, and both alloys stay direct across
    the whole composition range, so this is a guard rather than a physics model: it exists so
    that a future alloy or a future table entry cannot quietly be treated as direct when it is
    not. It does NOT account for strain, which shifts the valleys by different amounts -- the X
    and L deformation potentials are not carried here.
    """
    gaps = {'G': mat['Eg'], 'X': mat['Eg_X'], 'L': mat['Eg_L']}
    valley = min(gaps, key=gaps.get)
    return valley, gaps[valley]


def kp_params(mat):
    """A copy of `mat` carrying the keys `kp_pryor.material_fields` reads.

    That function was written against `pryor1998.PRYOR_TABLE_I`, which uses the same key names,
    so this is mostly an assertion that nothing is missing -- it fails loudly here rather than
    letting a KeyError surface from inside the Hamiltonian assembly.
    """
    needed = ('gamma1L', 'gamma2L', 'gamma3L', 'Eg', 'delta_so', 'Ep', 'a_c', 'a_v', 'b', 'd')
    missing = [k for k in needed if mat.get(k) is None]
    if missing:
        raise KeyError(f"{mat.get('name', '?')} is missing {missing}")
    return dict(mat)


def band_edge_fields(inside_mask, trace_strain, dot, matrix, zero='matrix_vb'):
    """Strained conduction and valence band-edge fields (eV) for a dot in a matrix.

    Convention, matching `pryor1998.band_edge_fields` and aestimo's own `Strain_and_Masses`:

        E_c = E_c(unstrained) + a_c * Tr(eps)
        E_v = E_v(unstrained) - a_v * Tr(eps)

    read together with the NEGATIVE a_v of the module docstring: under compression (Tr < 0) this
    lifts E_c and lowers E_v, opening the gap by |a_gap * Tr|. It is the sign convention
    `kp_pryor` expects, so the result can be handed straight to `material_fields`. Note it is
    the OPPOSITE sign on a_v from `qdsolver_core.MATERIALS`; mixing them moves the valence band
    the wrong way.

    Only the HYDROSTATIC part is applied here. The shear part (the Bir-Pikus q, r, s terms) is
    applied inside the Hamiltonian builder from the full strain tensor, so passing these fields
    on with `include_hydrostatic=True` would count the hydrostatic shift twice.

    `zero` sets the energy origin: 'matrix_vb' puts zero at the unstrained matrix valence edge
    (the analogue of Pryor's choice, and what the plots here use), 'absolute' keeps the review's
    own scale, on which InSb's unstrained valence edge is zero.
    """
    ref = matrix['VBO'] if zero == 'matrix_vb' else 0.0

    Ev0 = np.where(inside_mask, dot['VBO'] - ref, matrix['VBO'] - ref)
    Ec0 = Ev0 + np.where(inside_mask, dot['Eg'], matrix['Eg'])

    a_c = np.where(inside_mask, dot['a_c'], matrix['a_c'])
    a_v = np.where(inside_mask, dot['a_v'], matrix['a_v'])

    return Ec0 + a_c * trace_strain, Ev0 - a_v * trace_strain


def alignment(dot, matrix, trace_strain=0.0):
    """Where `dot` sits relative to `matrix`, on a zero at the unstrained matrix valence edge.

    Returns the four band edges, the misfit, the two well depths, and

        overlap = E_v(dot) - E_c(matrix)

    which is the number that decides the character of the system: POSITIVE means BROKEN GAP, the
    dot's valence edge above the matrix's conduction edge, so electrons stay in the matrix and
    holes in the dot, spatially separated. `type` names the alignment on the usual scheme.

    `e_well` and `h_well` are the single-particle well depths in the electron convention, both
    positive when the dot confines that carrier: e_well = E_c(matrix) - E_c(dot) and
    h_well = E_v(dot) - E_v(matrix). A negative one means that carrier is EXPELLED from the dot,
    which is the normal situation for the electron in this antimonide family.

    Only the dot is strained. That is the right thing for a coherently embedded island in a
    thick matrix: the matrix carries the reaction strain, but spread over a much larger volume
    and decaying away from the island, so a single number for it would mislead. Pass the mean
    Tr(eps) inside the dot from a real solve, or leave it 0 for the unstrained picture.
    """
    Ev_m, Ec_m = 0.0, matrix['Eg']
    Ev_d = (dot['VBO'] - matrix['VBO']) - dot['a_v'] * trace_strain
    Ec_d = (dot['VBO'] - matrix['VBO'] + dot['Eg']) + dot['a_c'] * trace_strain

    # Classify on the SIGNS of the two offsets, with a 1 meV deadband. Type I is the case where
    # both carriers are confined in the same material, i.e. the offsets have OPPOSITE signs;
    # type II is where they have the same sign and the carriers separate. The deadband matters:
    # without it, an alloy evaluated at the endpoint where it IS the matrix compares two exactly
    # equal edges and `>` sends the degenerate case down an arbitrary branch.
    dv, dc, tol = Ev_d - Ev_m, Ec_d - Ec_m, 1e-3

    if Ev_d - Ec_m > tol:
        kind = 'broken gap (III)'
    elif abs(dv) <= tol and abs(dc) <= tol:
        kind = 'no offset'
    elif dv * dc > 0.0:
        kind = 'staggered (II)'
    else:
        kind = 'straddling (I)'

    return dict(Ev_matrix=Ev_m, Ec_matrix=Ec_m, Ev_dot=Ev_d, Ec_dot=Ec_d,
                gap_dot=Ec_d - Ev_d, overlap=Ev_d - Ec_m, type=kind,
                misfit=misfit(dot, matrix),
                e_well=Ec_m - Ec_d, h_well=Ev_d - Ev_m)


def alignment_table(system='InGaSb', compositions=(0.0, 0.2, 0.35, 0.5, 0.75, 1.0),
                    matrix='InAs', trace_strain=None):
    """Print where each composition of `system` puts the band edges in `matrix`.

    `system` may be an alloy name or a binary, in which case `compositions` is ignored.
    `trace_strain` is a single representative Tr(eps) inside the dot -- pass the mean from a real
    solve to see where the edges actually land, or leave it None for the unstrained alignment.
    """
    mat = MATERIALS[matrix]
    tr = 0.0 if trace_strain is None else trace_strain
    xs = (None,) if system in MATERIALS else tuple(compositions)

    print(f"energy zero = unstrained {matrix} valence edge; {matrix}: "
          f"E_v = 0.000, E_c = {mat['Eg']:.3f} eV   (T = {TEMPERATURE:g} K)")
    print("unstrained alignment" if trace_strain is None else
          f"dot strained with Tr(eps) = {tr:+.4f} (matrix taken unstrained)")
    print(f"{'x':>5} {'name':>14} {'misfit':>8} {'Eg':>7} {'E_v':>8} {'E_c':>8} "
          f"{'E_v(dot)-E_c(mat)':>19}  alignment")
    for x in xs:
        d = material(system, x)
        a = alignment(d, mat, tr)
        label = '  -  ' if x is None else f"{x:>5.2f}"
        print(f"{label} {d['name']:>14} {a['misfit'] * 100:>7.2f}% {d['Eg']:>7.3f} "
              f"{a['Ev_dot']:>8.3f} {a['Ec_dot']:>8.3f} {a['overlap']:>+19.3f}  {a['type']}")
    print(f"\npositive last column = broken gap: the dot's valence edge lies above the {matrix}\n"
          "conduction edge, so electrons stay in the matrix and holes in the dot.")


def audit():
    """Print every parameter with its provenance, plus the four checks worth making.

    Run this after changing anything in the tables. It is the cheapest way to catch a
    transcription error, because the checks below are joint constraints on several parameters at
    once rather than restatements of single values.
    """
    print(f"source workbooks: {', '.join(WORKBOOKS)}")
    print(f"table evaluated at T = {TEMPERATURE:g} K\n")

    print(f"{'parameter':>10}" + ''.join(f"{m:>10}" for m in BINARIES) + "   provenance")
    for k in PROVENANCE:
        row = ''.join(f"{MATERIALS[m][k]:>10.4f}" for m in BINARIES)
        print(f"{k:>10}{row}   {PROVENANCE[k]}")

    print("\nCHECK 1  a_gap = a_c + a_v must be NEGATIVE (compression opens the gap).")
    print("         This is the convention-independent combination; a positive value means an")
    print("         a_v was imported from a source using the other sign convention.")
    for m in BINARIES:
        g = a_gap(MATERIALS[m])
        print(f"         {m:>5} {g:>+7.2f} eV   {'OK' if g < 0 else '*** WRONG SIGN ***'}")

    print("\nCHECK 2  the Kane expression must reproduce the workbook's own tabulated m*_e.")
    print("         A joint check on Eg, delta_so, Ep and F, and it only passes at 0 K -- at")
    print("         300 K gaps it is 24% out on InSb, which is how the temperature")
    print("         inconsistency in the previous parameter set was found.")
    print(f"         {'':>16} {'derived':>9} {'tabulated':>10} {'ratio':>8}")
    for m in BINARIES:
        mat = MATERIALS[m]
        der = electron_mass(mat)
        print(f"         {m:>16} {der:>9.4f} {mat['m_e']:>10.4f} {der / mat['m_e']:>8.3f}")

    print("\nCHECK 3  the same expression for the ALLOYS, against their bowed table mass.")
    print("         These do NOT agree, and the disagreement is in the source values, not here.")
    print("         Bowing m_e directly and bowing the ingredients of the Kane expression are")
    print("         two prescriptions, and the review's values are not consistent between them:")
    print("         the strong Eg bowing pulls the derived mass down faster than the tabulated")
    print("         m_e bowing does. electron_mass() stays authoritative -- the eight-band")
    print("         Hamiltonian is built from Ep, F, Eg and delta and has no other option.")
    print("         Read the spread as the real uncertainty on an alloy electron mass; it lands")
    print("         squarely on the confinement threshold, which goes as 1/m.")
    print(f"         {'':>16} {'derived':>9} {'tabulated':>10} {'ratio':>8}")
    for name in ALLOYS:
        for x in (0.25, 0.5, 0.75):
            a = alloy(name, x)
            der = electron_mass(a)
            print(f"         {a['name']:>16} {der:>9.4f} {a['m_e']:>10.4f} "
                  f"{der / a['m_e']:>8.3f}")

    print("\nCHECK 4  every material must still be direct at Gamma (unstrained).")
    for m in BINARIES:
        v, g = lowest_valley(MATERIALS[m])
        print(f"         {m:>16} lowest valley {v}  ({g:.3f} eV)   "
              f"{'OK' if v == 'G' else '*** INDIRECT ***'}")
    for name in ALLOYS:
        worst = max((alloy(name, i / 20.0) for i in range(21)),
                    key=lambda a: a['Eg'] - min(a['Eg_X'], a['Eg_L']))
        v, g = lowest_valley(worst)
        print(f"         {worst['name']:>16} lowest valley {v}  ({g:.3f} eV)   "
              f"{'OK' if v == 'G' else '*** INDIRECT ***'}  (worst case over x for {name})")

    print("\nanything tagged ADDED in the provenance column is not from the review's own band-")
    print("parameter tables -- vary it before trusting a result that depends on it. That is")
    print("e14 and eps_R, i.e. everything the piezoelectric potential is built from.")


if __name__ == '__main__':
    audit()
    print()
    alignment_table('InGaSb', matrix='InAs')
    print()
    alignment_table('InAsSb', matrix='InAs')
