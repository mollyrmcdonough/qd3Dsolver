"""The InAs / In(x)Ga(1-x)Sb antimonide system: the narrative layer over `materials.py`.

THE DATA NOW LIVES IN `materials.py`. This module holds no numbers of its own except the legacy
switches described below, and every value it exposes is the one `materials` carries. New code
should import `materials` directly:

    import materials as mt
    dot    = mt.material('InGaSb', 0.35)     # or mt.alloy('InGaSb', 0.35)
    matrix = mt.material('InAs')
    mt.alignment_table('InAsSb', matrix='InAs')
    mt.set_temperature(77.0)

This module remains because `ingasb_dot`, three notebooks and five scripts import it, because
its public names appear throughout their prose, and because the history below is worth keeping
attached to the system it is about rather than buried in a general database.

`SB_MATERIALS` is `materials.MATERIALS` -- the SAME dict objects, not a copy, so
`materials.set_temperature` is visible through it immediately. It now carries GaAs as well as
the three antimonides; nothing here assumes otherwise, and it makes In(x)Ga(1-x)Sb on GaAs
available for the price of an argument.

What makes this system worth its own module
-------------------------------------------
The alignment is BROKEN GAP (type III). The In(x)Ga(1-x)Sb valence edge sits ABOVE the InAs
conduction edge at every composition -- by 143 meV at x = 0 and 173 meV at x = 1, unstrained --
so electrons stay in the InAs matrix while holes sit in the dot, spatially separated. The
electron is not confined by the dot at all; it is expelled from it. What binds it, if anything,
is a shallow pocket the dot digs in the matrix AROUND itself: the dot is under strong
compression (misfit -0.5% to -6.5% with composition) and by reaction the InAs immediately
around it is put into tension, which with a_c negative pulls the InAs conduction edge DOWN.

That is why this system needed the inhomogeneous elasticity solver rather than the Fourier one.
The pocket lives in the matrix, where a homogeneous solver gets the stiffness contrast wrong,
and the In(x)Ga(1-x)Sb is substantially softer than the InAs around it.

The parameter set had three errors, and they were 130 meV
---------------------------------------------------------
The numbers here originally came from `database.py` in the sibling `aestimo` checkout, which
attributes its III-V entries to Vurgaftman, Meyer and Ram-Mohan (2001) but had transcribed three
of them wrongly. Checking against the review's own tables -- now `bandparameters_vurgaftman2001
.xlsx`, see `materials.py` -- found:

    a_v    SIGN FLIPPED for InAs (+1.00 vs -1.00) and InSb (+0.31 vs -0.36)
    a_c    InSb -6.04 vs -6.94, GaSb -9.33 vs -7.50
    VBO    GaSb 130 meV too low

The consequence is the hydrostatic gap deformation potential a_gap = a_c + a_v:

                database    review
        InAs      -4.08     -6.08     (Pryor 1998 Table I, independently: -6.00)
        GaSb      -8.01     -8.30
        InSb      -5.73     -7.30

-7.30 for InSb is what `scripts/insb_pryor_pistol_check.py` had already back-derived as -7.29 eV
from Pryor & Pistol's Table I well, BEFORE the workbook was consulted. That 10 meV agreement is
what pins the error to the parameters rather than to the elasticity or the k.p machinery.

A fourth inconsistency was a temperature mismatch rather than a wrong number: the GaSb offset
was derived from Pryor & Pistol's 0 K table while the gaps in force were aestimo's ~300 K
values. The whole set is 0 K now, and `materials.electron_mass` reproducing all four tabulated
band-edge masses to the digit is the check that says so.

Results computed before those fixes are wrong by of order 130 meV. See `archive/`.

The legacy switches
-------------------
`GASB_AV_CONVENTION`, `GASB_VBO_SOURCE` and `GAP_SOURCE` predate the audit. They exist to
reproduce pre-audit behaviour on demand -- for a sensitivity study, or to re-derive how an old
number came out the way it did -- and every one of them is now known to be a worse choice than
the default. They are kept working rather than deleted because two scripts use them as exhibits,
and because a switch that silently stops switching is worse than one that is merely obsolete.
"""
import materials as mt

# --------------------------------------------------------------------------------------
# Re-exports. These are `materials`' own objects; this module adds nothing to them.
# --------------------------------------------------------------------------------------
from materials import (                                          # noqa: F401
    MATERIALS as SB_MATERIALS,
    PROVENANCE,
    a_gap,
    alignment,
    band_edge_fields,
    elastic,
    electron_mass,
    kp_params,
    lowest_valley,
    material,
    misfit,
    set_temperature,
)

#: Gap bowing for In(x)Ga(1-x)Sb, eV. Read from `materials` so there is one copy.
INGASB_EG_BOWING = mt._BOWING['InGaSb']['bowing']['Eg']

#: Valence-band-offset bowing for In(x)Ga(1-x)Sb, eV. Zero -- the review lists no VBO bowing for
#: this alloy, so the interpolation is linear. Exposed because it was once a live question:
#: aestimo's `InGaSb` entry carries `AVb_E = -2.1`, which cannot be an absolute energy (the
#: parents' are near -6 eV on that scale) and must have been meant as a bowing parameter, but a
#: 2.1 eV valence bowing is implausibly large and nothing in that file says which it is.
INGASB_VBO_BOWING = 0.0


def ingasb(x, eg_bowing=None, vbo_bowing=None):
    """In(x)Ga(1-x)Sb parameters. `x` is the InSb fraction, so x = 0 is GaSb and x = 1 is InSb.

    Equivalent to `materials.alloy('InGaSb', x)`, which is what new code should call. The two
    bowing arguments are kept so a notebook can vary them without touching the tables; pass None
    (the default) to use the module values.

    Vegard's law for the lattice constant is what makes composition a strain knob here: a0 runs
    6.0817 -> 6.4690 Angstrom at 0 K, i.e. the misfit against InAs runs -0.52% -> -6.48%.
    """
    if eg_bowing is None and vbo_bowing is None:
        return mt.alloy('InGaSb', x)

    spec = mt._BOWING['InGaSb']
    bow = dict(spec['bowing'])
    if eg_bowing is not None:
        bow['Eg'] = eg_bowing
    if vbo_bowing is not None:
        bow['VBO'] = vbo_bowing

    A, B = (SB_MATERIALS[p] for p in spec['parents'])
    out = {k: x * A[k] + (1 - x) * B[k] - bow.get(k, 0.0) * x * (1 - x)
           for k in mt._INTERPOLATED}
    out['system'], out['x'] = 'InGaSb', x
    out['name'] = ({1.0: 'InSb', 0.0: 'GaSb'}.get(x) or spec['formula'](x))
    return out


def alignment_table(compositions=(0.0, 0.2, 0.35, 0.5, 0.75, 1.0), matrix='InAs',
                    trace_strain=None):
    """In(x)Ga(1-x)Sb alignment in `matrix`. See `materials.alignment_table` for the general
    form, which takes the alloy name first and also handles InAs(1-x)Sb(x) and the binaries."""
    return mt.alignment_table('InGaSb', compositions, matrix, trace_strain)


# --------------------------------------------------------------------------------------
# Legacy switches. All three reproduce pre-audit behaviour; none is a better choice than
# the default. See the module docstring.
# --------------------------------------------------------------------------------------

#: DEPRECATED, and it was based on a WRONG DIAGNOSIS. Kept only so old notebooks still import.
#:
#: The reasoning was: aestimo gives a_v = +1.00 (InAs), +0.31 (InSb) and -1.32 (GaSb), GaSb is
#: the odd sign out, therefore GaSb is the error. Checked against the review directly, it is the
#: other way round -- a_v is NEGATIVE for all of them, and aestimo had flipped the sign for InAs
#: and InSb while leaving GaSb's alone. GaSb was the only correct one, and 'signfixed' made it
#: worse. See `materials.py` for the convention and why it is the one that holds.
GASB_AV_CONVENTION = 'deprecated'

_GASB_AV = {'database': -1.32, 'signfixed': 1.32, 'vurgaftman': -0.80}

#: GaSb's valence-band edge, on the review's scale where InSb is zero.
#:
#: 'database'     -- -0.16 eV, i.e. aestimo's -6.25 rescaled. 130 meV too low.
#: 'pryor_pistol' -- -0.03 eV, from C. E. Pryor and M.-E. Pistol, Phys. Rev. B 72, 205311
#:                   (2005), arXiv:cond-mat/0501090, whose Table I gives unstrained edges on a
#:                   scale with the InSb valence edge at zero. It coincides with the review's
#:                   own tabulated value, which is what `materials.py` carries.
#:
#: These were quoted as -6.25 and -6.12 eV before the energy zero moved to the review's scale;
#: the shift is uniform and only differences of VBO are physical, so nothing changed but the
#: printed numbers.
_GASB_VBO = {'database': -0.16, 'pryor_pistol': -0.03}

GASB_VBO_SOURCE = 'pryor_pistol'

#: The gaps as transcribed from aestimo's `database.py`, kept so the switch is reversible. These
#: are ~300 K values: Varshni at 300 K gives 0.354 / 0.727 / 0.174, which matches GaSb and InSb
#: but not InAs, whose entry of 0.400 sits between the 0 K and 300 K values.
_DATABASE_EG = {'InAs': 0.400, 'GaSb': 0.726, 'InSb': 0.174}

#: A flat view of the Gamma-valley Varshni parameters, in the shape this module used to expose
#: them. `materials._VARSHNI` is keyed by valley as well and is the one to read in new code.
_VARSHNI = {name: dict(zip(('Eg0', 'alpha', 'beta'), vals['G']))
            for name, vals in mt._VARSHNI.items()}

#: Which gaps are in force: 'database' or 'varshni'. See `set_gap_source`.
GAP_SOURCE = 'varshni'

#: Temperature the Varshni gaps are evaluated at, K. None when GAP_SOURCE = 'database'.
GAP_TEMPERATURE = 0.0


def varshni(name, T):
    """Eg(T) at the Gamma point for one binary. `materials.varshni` takes a valley too."""
    return mt.varshni(name, 'G', T)


def set_gap_source(source, T=0.0):
    """Switch the band gaps between aestimo's values and Varshni at temperature `T`.

    A function rather than a constant because `SB_MATERIALS` is populated at import, so
    reassigning a module constant afterwards would be a silent no-op.

    'varshni' delegates to `materials.set_temperature`, which also moves the LATTICE CONSTANTS.
    'database' overrides the gaps ONLY and leaves the lattice wherever the current temperature
    put it -- which is exactly what this function did before the lattice became temperature
    dependent, and is the behaviour `scripts/insb_band_edges_vs_size.py` relies on to isolate
    the gap inconsistency from everything else. For a faithful all-300 K legacy reproduction,
    call `set_temperature(300.0)` first.

    Changing Eg is not local: it feeds the conduction edge, the derived Kane mass, and the
    eight-band Hamiltonian. `ingasb(x)` re-reads the table on every call, so alloys follow.

    Returns the gaps now in force.
    """
    if source not in ('database', 'varshni'):
        raise ValueError(f"unknown source {source!r}; choose 'database' or 'varshni'")
    global GAP_SOURCE, GAP_TEMPERATURE

    if source == 'varshni':
        set_temperature(T)
        GAP_TEMPERATURE = float(T)
    else:
        for name, Eg in _DATABASE_EG.items():
            SB_MATERIALS[name]['Eg'] = Eg
        GAP_TEMPERATURE = None
    GAP_SOURCE = source
    return {name: SB_MATERIALS[name]['Eg'] for name in _DATABASE_EG}


def set_gasb_vbo(source):
    """Switch GaSb's valence-band edge at runtime; returns the value now in force."""
    if source not in _GASB_VBO:
        raise ValueError(f"unknown source {source!r}; choose from {list(_GASB_VBO)}")
    global GASB_VBO_SOURCE
    GASB_VBO_SOURCE = source
    SB_MATERIALS['GaSb']['VBO'] = _GASB_VBO[source]
    return SB_MATERIALS['GaSb']['VBO']


def set_gasb_av(convention):
    """Switch GaSb's a_v convention at runtime and return the value now in force.

    DEPRECATED -- the diagnosis behind it was wrong; see `GASB_AV_CONVENTION`. Kept working so a
    sensitivity study that flips it does not silently find nothing.
    """
    if convention not in _GASB_AV:
        raise ValueError(f"unknown convention {convention!r}; choose from {list(_GASB_AV)}")
    global GASB_AV_CONVENTION
    GASB_AV_CONVENTION = convention
    SB_MATERIALS['GaSb']['a_v'] = _GASB_AV[convention]
    return SB_MATERIALS['GaSb']['a_v']


def audit():
    """`materials.audit()`, plus the state of the legacy switches this module still exposes."""
    mt.audit()
    print("\n" + "=" * 78)
    print("legacy switches in `materials_sb` (all three reproduce pre-audit behaviour):")
    print(f"  GASB_AV_CONVENTION = {GASB_AV_CONVENTION!r:>14}  ->  GaSb a_v = "
          f"{SB_MATERIALS['GaSb']['a_v']:+.2f} eV   (deprecated: the diagnosis was wrong)")
    print(f"  GASB_VBO_SOURCE    = {GASB_VBO_SOURCE!r:>14}  ->  GaSb VBO = "
          f"{SB_MATERIALS['GaSb']['VBO']:+.2f} eV")
    print(f"  GAP_SOURCE         = {GAP_SOURCE!r:>14}  ->  T = {GAP_TEMPERATURE} K")
    print(f"  INGASB_EG_BOWING   = {INGASB_EG_BOWING} eV")
    print(f"  INGASB_VBO_BOWING  = {INGASB_VBO_BOWING} eV  (0 = linear VBO interpolation)")


if __name__ == '__main__':
    audit()
    print()
    alignment_table()
