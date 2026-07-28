"""Material parameters and band-edge construction for the Pryor (1998) InAs/GaAs benchmark.

Source: C. Pryor, "Eight-band calculations of strained InAs/GaAs quantum dots compared with
one-, four-, and six-band approximations", Phys. Rev. B 57, 7190 (1998); preprint freely
available as arXiv:cond-mat/9710304. All numbers in PRYOR_TABLE_I below were read off that
paper's Table I directly (the preprint PDF was fetched and read, not recalled), so unlike the
rest of this package these citations *are* pinned to a specific table. Per the citation policy
in qdsolver_core.py, equation numbers are still not quoted -- the equations were read but the
numbering in the published PRB version may differ from the preprint's.

Why this module exists separately from qdsolver_core.MATERIALS
-------------------------------------------------------------
Two things differ from our default parameterization and BOTH are sign/convention traps:

1. Band alignment. Our MATERIALS dict builds band edges from aestimo's `Band_offset` fraction
   of the gap. Pryor instead specifies the unstrained valence-band offset directly,
   E_vbo = E_v(InAs) - E_v(GaAs) = 85 meV, derived (his Sec. III) from Mn impurity ground-state
   energies 0.028 eV (InAs) and 0.113 eV (GaAs) above the respective valence bands, on the
   argument that transition-metal impurity levels are roughly fixed relative to the vacuum
   level; he notes this agrees with Au Schottky-barrier data. All energies here are therefore
   measured from the *unstrained GaAs valence-band edge*, which is also the zero used in his
   Figs. 4 and 7.

2. Sign convention for the valence-band hydrostatic deformation potential. Pryor's Table I
   lists a_g together with a_c and a_v such that a_g = a_v + a_c (InAs: 0.66 - 6.66 = -6.0;
   GaAs: 0.7 - 9.3 = -8.6, both matching his a_g row). That means his valence-band edge shifts
   by -a_v*Tr(eps), i.e. UP under the compressive strain of a buried InAs dot. The convention
   used elsewhere in this package (inherited from aestimo, and the more common Van de Walle
   one) instead has E_v shift by +a_v*Tr(eps), with a_g = a_c - a_v. Both give the same gap
   deformation potential once the matching a_v sign is used, but mixing Pryor's a_v into our
   line would move the valence band the wrong way and inflate the strained gap by ~150 meV.
   `band_edge_fields` below implements PRYOR's convention explicitly; do not reuse it for the
   MATERIALS dict.

What this benchmark can and cannot test
---------------------------------------
Pryor solves linear continuum elasticity numerically on the grid, so his strain is
inhomogeneous: the conduction well is 0.4 eV deep at the base of the island and tapers to
0.27 eV at the tip (his Sec. IV / Fig. 2). Our Davies "simple picture" gives a single uniform
Tr(eps) inside the island, ~0.30 eV deep -- in between, but flat. So the electron ground state
(which sits low in the pyramid, where his well is deeper) is expected to come out somewhat
less bound than his. The hole is affected far more: hole confinement in his calculation is
dominated by the *shear* strain terms (his q, r, s, built from b and d), which the
hydrostatic-only approximation discards entirely. Hole numbers from a one-band solve with
these parameters are therefore not a meaningful comparison against his Fig. 4b/7b -- those
need the shear strain and a multiband valence Hamiltonian.
"""
import numpy as np

# Table I. Elastic constants converted from Pryor's dyne/cm^2 to GPa for consistency with the
# rest of this package (1e11 dyne/cm^2 = 10 GPa); lattice constants converted nm -> Angstrom,
# again to match qdsolver_core.MATERIALS. Every other value is verbatim.
PRYOR_TABLE_I = {
    'InAs': dict(
        gamma1L=19.67, gamma2L=8.37, gamma3L=9.29,
        Eg=0.418,          # eV
        delta_so=0.38,     # eV  (Pryor's Delta)
        Ep=22.2,           # eV
        a_g=-6.0,          # eV  = a_v + a_c in Pryor's convention
        a_c=-6.66,         # eV
        a_v=0.66,          # eV  -- PRYOR sign convention, see module docstring
        b=-1.8, d=-3.6,    # eV, shear deformation potentials (unused by the one-band model)
        e14=0.045,         # C/m^2, piezoelectric constant (unused: piezo not implemented)
        eps_R=15.15,
        C11=83.29, C12=45.26, C44=39.59,   # GPa (8.329/4.526/3.959 x 1e11 dyne/cm^2)
        a0=6.0583,         # Angstrom (0.60583 nm)
        E_vbo=0.085,       # eV, E_v(InAs) - E_v(GaAs), unstrained
    ),
    'GaAs': dict(
        gamma1L=6.85, gamma2L=2.1, gamma3L=2.9,
        Eg=1.519,
        delta_so=0.33,
        Ep=25.7,
        a_g=-8.6,
        a_c=-9.3,
        a_v=0.7,
        b=-2.0, d=-5.4,
        e14=0.159,
        eps_R=15.15,       # Table I footnote: the InAs value is used throughout the structure
        C11=121.1, C12=54.8, C44=60.4,     # GPa (12.11/5.48/6.04 x 1e11 dyne/cm^2)
        a0=5.6532,         # Angstrom (0.56532 nm)
        E_vbo=0.0,         # reference material: zero of energy
    ),
}

# Conduction-band effective masses used for the one-band comparisons in Pryor's Sec. VI.
# These are stated in the text rather than Table I:
#   - 0.023 m0 (InAs) / 0.0665 m0 (GaAs): the unstrained bulk values (his case i; the 0.023
#     figure also appears as the "unstrained" arrow on his Fig. 3 histogram).
#   - 0.040 m0 in the InAs: the value predicted for bulk InAs under the *average* hydrostatic
#     strain in the island, from the pseudopotential calculation of Cusack, Briddon and Jaros,
#     Phys. Rev. B 54, 39 (1996); Pryor notes it agrees with the peak of his own Fig. 3
#     strain-dependent-mass histogram. This is his case (ii).
PRYOR_ELECTRON_MASSES = {
    'unstrained': dict(dot=0.023, matrix=0.0665),   # Pryor Sec. VI case (i)
    'strain_averaged': dict(dot=0.040, matrix=0.0665),  # Pryor Sec. VI case (ii)
}


def heavy_hole_mass_001(params):
    """[001] heavy-hole mass 1/(gamma1 - 2*gamma2) from Pryor's Table I Luttinger parameters.

    NOTE this is our own extension, not part of the benchmark: Pryor computes valence states
    with four-, six- and eight-band Hamiltonians only, never a one-band hole. Provided so the
    single-band machinery can be run end-to-end with a consistent parameter set, not so the
    result can be compared to his Fig. 4b.

    References: the [001] HH/LH masses in terms of Luttinger parameters are the standard
    Luttinger-Kohn result -- J. M. Luttinger and W. Kohn, Phys. Rev. 97, 869 (1955).
    """
    return 1.0 / (params['gamma1L'] - 2 * params['gamma2L'])


def band_edge_fields(inside_mask, trace_strain, mass_case='unstrained'):
    """Band-edge potentials (eV) and mass fields for a dot described by `inside_mask`, using
    Pryor's parameters and his conventions.

    Energy zero = unstrained GaAs valence-band edge (the zero of his Figs. 4 and 7), so the
    returned V_e is directly comparable to his E_c axis and V_h to his E_v axis.

    Unstrained edges:      E_v = E_vbo,            E_c = E_vbo + Eg
    Hydrostatic shifts:    dE_c = a_c * Tr(eps),   dE_v = -a_v * Tr(eps)   [Pryor convention]

    `trace_strain` should now be strain_fourier.solve_strain(...).trace, which is spatially
    varying inside the dot AND nonzero in the barrier -- real continuum elasticity does leak
    strain into the matrix, so the barrier band edges bend near the dot. That is handled
    correctly here: every term is evaluated point by point.

    The older qdsolver_core.trace_strain_from_mask field (uniform inside, exactly zero outside)
    still works as input, but see that function's docstring -- it is superseded and overstates
    the shift by a factor of ~1.17.

    Returns (V_e, V_h, m_e_field, m_h_field).
    """
    dot, matrix = PRYOR_TABLE_I['InAs'], PRYOR_TABLE_I['GaAs']

    Ev_unstrained = np.where(inside_mask, dot['E_vbo'], matrix['E_vbo'])
    Ec_unstrained = Ev_unstrained + np.where(inside_mask, dot['Eg'], matrix['Eg'])

    a_c = np.where(inside_mask, dot['a_c'], matrix['a_c'])
    a_v = np.where(inside_mask, dot['a_v'], matrix['a_v'])

    V_e = Ec_unstrained + a_c * trace_strain
    V_h = Ev_unstrained - a_v * trace_strain   # Pryor sign convention; see module docstring

    masses = PRYOR_ELECTRON_MASSES[mass_case]
    m_e_field = np.where(inside_mask, masses['dot'], masses['matrix'])
    m_h_field = np.where(inside_mask, heavy_hole_mass_001(dot), heavy_hole_mass_001(matrix))

    return V_e, V_h, m_e_field, m_h_field


# Benchmark targets read from Pryor's Sec. VI and Fig. 7a, for the b = 14 nm island. His
# figures put the GaAs conduction-band edge at Eg(GaAs) = 1.519 eV on this energy scale, so
# "binding" below means 1.519 - E.
BENCHMARK_B14 = dict(
    E0_unstrained_mass=1.41,      # eV, Fig. 7a leftmost point (m = 0.023/0.0665); ~110 meV bound
    n_bound_unstrained_mass=1,    # Sec. VI: "For the simple unstrained effective mass only a
                                  # single state is found"
    extra_binding_strain_averaged=0.030,  # eV; Sec. VI: with m = 0.04 m0 the ground-state
                                          # binding energy increases by 30 meV
    E1_minus_E0_strain_averaged=0.110,    # eV; Sec. VI: the one-band models with m = 0.04 m0
                                          # and m_eff(r) both give E1 - E0 ~ 110 meV
    exciton_binding=0.0215,       # eV, Hartree exciton binding read off Fig. 5 at b = 14 nm
                                  # (his curve runs 27 meV at b = 9 nm to ~17.5 meV at b = 18)
    GaAs_CB_edge=1.519,           # eV on this energy scale
)
