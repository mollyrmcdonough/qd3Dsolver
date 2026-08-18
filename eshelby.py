"""Closed-form Eshelby strain inside an oblate-spheroidal inclusion, isotropic elasticity.

Transcribed from the Appendix of

    S. I. Rybchenko, G. Yeap, R. Gupta, I. E. Itskevich and S. K. Haywood,
    "Importance of aspect ratio over shape in determining the quantization potential of
    self-assembled zinc-blende III-V quantum dots", J. Appl. Phys. 102, 013706 (2007),

which specialises Eshelby's general ellipsoidal-inclusion solution (J. D. Eshelby, Solid State
Phys. 3, 79 (1956); Proc. R. Soc. London A 241, 376 (1957)) to an oblate spheroid a = b > c with
the (001) growth plane along c.

**Why this is transcribed from Rybchenko rather than Eshelby.** Per the citation policy in
`qdsolver_core.py`, equation numbers are only quoted where there is a verifiable provenance
chain. Eshelby's originals are paywalled and were not read; Rybchenko's appendix WAS read
directly (`013706_1_online.pdf` in this repo), so the transcription here has a checkable source
and the section numbers below refer to that appendix.

What this is for
----------------
Two things the finite-element solver in `elasticity_fd` cannot give:

1. **A grid-free interior strain.** Eshelby's theorem makes the strain inside an ellipsoidal
   inclusion uniform, so there is no staircasing, no padding, and no periodic-supercell image
   interaction. `elasticity_fd` solves a periodic cell, and the measured interior scatter in
   Tr(eps) is ~1e-2 at pad/base = 1 on a quantity that is provably constant.
2. **A reference at intermediate aspect ratio.** The existing elasticity validation covers the
   sphere and the clamped slab. Nothing covered the flat-lens regime in between, which is exactly
   where self-assembled dots live.

Limitations, stated plainly
---------------------------
* **Isotropic elasticity only.** Rybchenko use the Voigt shear average for the matrix, their
  Eq. 3, which is bit-identical to `elasticity_fd.voigt_moduli`. They compare this against their
  own cubic-anisotropic FEM and report the difference as negligible for the valence band edge and
  a minor systematic overestimation for the conduction band edge. That is their assessment of
  their own results, adopted here rather than independently verified.
* **Oblate spheroid only** (a = b > c). Prolate and triaxial need different I integrals.
* The matrix field is available only AT the interface (section 3 of their appendix), not
  throughout the matrix.
"""
import numpy as np

FOURPI = 4.0 * np.pi

#: Below this value of u = 1 - (c/a)^2 the exact expressions lose precision to cancellation and
#: the series forms are used instead.
#:
#: Set from the MEASURED crossover, which is a sharp V. Relative disagreement between the two
#: branches in I_ac, on a = 3:
#:
#:     u        1e-6    1e-5    1e-4    3e-4    1e-3    3e-3    1e-2    3e-2    1e-1
#:     rel      1.0     4.3e-2  4.5e-5  3.5e-6  6.2e-7  4.7e-6  5.2e-5  4.6e-4  4.8e-3
#:
#: Left of the minimum the EXACT form is the wrong one -- at u = 1e-6 it returns I_ac = -7.62
#: against a true value near +0.0931, having lost the answer entirely to cancellation. Right of
#: it the truncated series runs out. u = 1e-3 is the crossing, good to ~6e-7 either side, and the
#: window u = 3e-4 .. 3e-3 is where they can be meaningfully cross-checked.
#:
#: This only matters within a hair of a sphere: AR = 1.5 already gives u = 0.56. The sphere
#: itself (u = 0 exactly) is handled by the series limits, which are exact.
_SERIES_U = 1.0e-3


def _bracket(t, u):
    """arccos(t) - t*sqrt(u) with t = c/a, u = 1 - t^2, stable as u -> 0.

    The two terms both tend to sqrt(u) and cancel to O(u^1.5), so the direct form loses about
    half its significant digits by u ~ 1e-8. The series is
        (2/3) u^1.5 + (1/5) u^2.5 + (3/28) u^3.5 + ...
    obtained from arcsin(sqrt u) - sqrt(1-u) sqrt(u).
    """
    if u < _SERIES_U:
        return u ** 1.5 * (2.0 / 3.0 + u / 5.0 + 3.0 * u * u / 28.0)
    return np.arccos(t) - t * np.sqrt(u)


def shape_integrals(a, c):
    """The I integrals of the appendix, section 1, for an oblate spheroid a = b > c.

    Returns (I_a, I_c, I_ac, I_cc, I_aa, I_ab). Units are 1/length^2 for the double-index ones.
    Sphere limits (a = c) are exact: I_a = I_c = 4pi/3, I_ac = I_ab = 4pi/15a^2,
    I_aa = I_cc = 4pi/5a^2.
    """
    a = float(a)
    c = float(c)
    if c > a:
        raise ValueError(f"oblate spheroid requires c <= a (got a={a}, c={c}); "
                         "prolate needs different I integrals")
    t = c / a
    u = 1.0 - t * t

    if u < _SERIES_U:
        # I_a = 2 pi t / u^1.5 * bracket, with the bracket's leading u^1.5 cancelling.
        I_a = 2.0 * np.pi * t * (2.0 / 3.0 + u / 5.0 + 3.0 * u * u / 28.0)
        # I_ac = (I_c - I_a) / (3(a^2 - c^2)) = (4pi - 3 I_a) / (3 a^2 u). The numerator cancels
        # to O(u), so u is divided out ANALYTICALLY rather than numerically -- otherwise the
        # sphere (u = 0 exactly) divides by zero, and near it the result is noise. Using
        # 1 - sqrt(1-u) = u/2 + u^2/8 + u^3/16 + ..., the numerator over u is
        num_over_u = (FOURPI * (0.5 + u / 8.0 + u * u / 16.0)
                      - 6.0 * np.pi * t * (1.0 / 5.0 + 3.0 * u / 28.0))
        I_ac = num_over_u / (3.0 * a * a)
    else:
        I_a = 2.0 * np.pi * a * a * c / (a * a - c * c) ** 1.5 * _bracket(t, u)
        I_ac = (FOURPI - 3.0 * I_a) / (3.0 * (a * a - c * c))

    I_c = FOURPI - 2.0 * I_a
    I_cc = FOURPI / (3.0 * c * c) - 2.0 * I_ac
    # I_aa = 4pi/3a^2 - I_ab - I_ac with I_ab = I_aa/3  =>  I_aa = pi/a^2 - (3/4) I_ac
    I_aa = np.pi / (a * a) - 0.75 * I_ac
    I_ab = I_aa / 3.0
    return I_a, I_c, I_ac, I_cc, I_aa, I_ab


def eshelby_tensor(a, c, sigma):
    """The S_ijkl of the appendix, section 1, as a dict keyed by the paper's index strings.

    `sigma` is the MATRIX Poisson ratio. Only the components the interior solution needs are
    returned, plus the shear ones for completeness.
    """
    I_a, I_c, I_ac, I_cc, I_aa, I_ab = shape_integrals(a, c)
    Q = 3.0 / (8.0 * np.pi * (1.0 - sigma))
    R = (1.0 - 2.0 * sigma) / (8.0 * np.pi * (1.0 - sigma))
    a2, c2 = a * a, c * c
    return dict(
        S1111=Q * a2 * I_aa + R * I_a,
        S2222=Q * a2 * I_aa + R * I_a,
        S3333=Q * c2 * I_cc + R * I_c,
        S1122=Q * a2 * I_ab - R * I_a,
        S2211=Q * a2 * I_ab - R * I_a,
        S1133=Q * c2 * I_ac - R * I_a,
        S2233=Q * c2 * I_ac - R * I_a,
        S3311=Q * a2 * I_ac - R * I_c,
        S3322=Q * a2 * I_ac - R * I_c,
        S1212=Q * a2 * I_ab + R * I_a,
        S1313=0.5 * Q * (a2 + c2) * I_ac + 0.5 * R * (I_a + I_c),
        S2323=0.5 * Q * (a2 + c2) * I_ac + 0.5 * R * (I_a + I_c),
        Q=Q, R=R,
    )


def isotropic_moduli(C):
    """(k, mu) Voigt averages from cubic (C11, C12, C44) -- Rybchenko Eq. 3 for mu.

    Delegates to `elasticity_fd.voigt_moduli`, which computes exactly
    mu = (C11 - C12)/5 + 3 C44/5, verified identical in `scripts/convention_audit.py`.
    """
    import elasticity_fd as ef
    return ef.voigt_moduli(*C)


def oblate_inclusion_strain(eps_star, C_dot, C_matrix, a, c):
    """Uniform TOTAL strain inside an oblate-spheroidal inclusion. Appendix section 2.

    Parameters
    ----------
    eps_star : float
        Lattice mismatch (a_matrix - a_dot) / a_dot -- the paper's e*, and exactly what
        `materials.misfit(dot, matrix)` returns (checked in `scripts/convention_audit.py`).
        Negative for a dot with the larger lattice constant, i.e. compressed.
    C_dot, C_matrix : (C11, C12, C44)
        Cubic constants; reduced to isotropic (k, mu) internally.
    a, c : float
        Semi-axes. a = b is the in-plane semi-axis (base/2), c the semi-axis along [001]
        (height/2). Requires c <= a.

    Returns
    -------
    dict with e11 = e22 (in-plane), e33 (growth direction), and trace.

    Sign convention note: the returned components are the strain INSIDE the dot relative to the
    dot's own unstrained lattice, matching `elasticity_fd.solve_strain_fd`'s elastic strain in
    the sense used by the deformation potentials.
    """
    k, mu = isotropic_moduli(C_matrix)
    k_i, mu_i = isotropic_moduli(C_dot)
    if abs(mu - mu_i) < 1e-12 or abs(k - k_i) < 1e-12:
        raise ValueError("dot and matrix moduli are equal; the appendix's A-E coefficients "
                         "divide by (mu - mu_i) and (k - k_i). Use the homogeneous-inclusion "
                         "result (qdsolver_core.trace_strain_from_mask) instead.")
    sigma = (3.0 * k - 2.0 * mu) / (2.0 * (3.0 * k + mu))
    S = eshelby_tensor(a, c, sigma)

    m_ratio = mu / (mu - mu_i)
    k_ratio = k / (k - k_i)
    A = 2.0 * m_ratio + 6.0 * S['S3311'] - 2.0 * k_ratio
    B = -2.0 * m_ratio + 3.0 * S['S3333'] - k_ratio
    C = k_ratio
    D = S['S1111'] + S['S1122'] - 2.0 * S['S3311'] - m_ratio
    E = S['S1133'] - S['S3333'] + m_ratio

    e33 = 3.0 * eps_star * C * D / (A * E - B * D)
    e11 = -(E / D) * e33

    # The appendix's e^T is the EQUIVALENT EIGENSTRAIN, not the strain inside the dot.
    #
    # This is the one thing in the transcription that cannot be read off the page, and getting it
    # wrong is silent: e^T has the right magnitude and the right aspect-ratio trend, so it looks
    # like a strain. Measured on a sphere it is 1.11407 eps* while the true elastic strain is
    # 0.46700 eps* -- and the ratio, 0.41919, is exactly |1 - alpha| with
    # alpha = (1+sigma)/(3(1-sigma)) the dilatational Eshelby factor. That is the signature of an
    # eigenstrain awaiting contraction with S, not of a botched coefficient.
    #
    # Applying Eshelby's construction: the constrained strain is e^C = S : e**, and the elastic
    # strain the deformation potentials act on is e^C - e**. At the sphere that gives
    # (alpha - 1) * 1.11407 = -0.46700 eps*, reproducing `inhomogeneous_sphere_trace` exactly.
    ec11 = S['S1111'] * e11 + S['S1122'] * e11 + S['S1133'] * e33
    ec33 = S['S3311'] * e11 + S['S3322'] * e11 + S['S3333'] * e33
    # Sign: e** - e^C, not e^C - e**. Both orderings give the same magnitude, and which one is
    # "the" elastic strain depends on whether eps* is referred to the dot's lattice or the
    # matrix's -- an ambiguity the appendix does not resolve and that cannot be settled by
    # algebra. It is settled by physics instead: InSb in InAs is COMPRESSED, so the elastic
    # strain must carry the same sign as eps* (negative). This ordering reproduces
    # `elasticity_fd.inhomogeneous_sphere_trace` to machine precision and matches the sign of the
    # FE solver; the other ordering matches the magnitude and gets the sign backwards, which
    # would flip every deformation-potential shift downstream.
    el11 = e11 - ec11
    el33 = e33 - ec33
    return dict(e11=el11, e22=el11, e33=el33, trace=2.0 * el11 + el33,
                eigen11=e11, eigen33=e33, constrained11=ec11, constrained33=ec33,
                sigma=sigma, k=k, mu=mu, k_i=k_i, mu_i=mu_i)


def _selftest(verbose=True):
    """Internal consistency: sphere limits of the I integrals, and series/exact agreement.

    Does NOT check the physics -- that is `scripts/ellipsoid_strain_validation.py`, which
    compares against the finite-element solver and the known inhomogeneous-sphere result.
    """
    ok = True

    def chk(name, got, want, tol):
        nonlocal ok
        good = abs(got - want) <= tol
        ok &= good
        if verbose:
            print(f"  [{'PASS' if good else 'FAIL'}] {name}: {got:.10g} vs {want:.10g}")

    a = 3.0
    I_a, I_c, I_ac, I_cc, I_aa, I_ab = shape_integrals(a, a)
    chk("sphere I_a == 4pi/3", I_a, FOURPI / 3.0, 1e-12)
    chk("sphere I_c == 4pi/3", I_c, FOURPI / 3.0, 1e-12)
    chk("sphere I_ac == 4pi/15a^2", I_ac, FOURPI / (15.0 * a * a), 1e-12)
    chk("sphere I_aa == 4pi/5a^2", I_aa, FOURPI / (5.0 * a * a), 1e-12)
    chk("sphere I_cc == I_aa", I_cc, I_aa, 1e-12)
    chk("sphere I_ab == I_ac", I_ab, I_ac, 1e-12)

    # Series and exact branches must agree in the OVERLAP where both are well conditioned.
    #
    # Neither form is valid everywhere, so this is not a range that can be widened at will.
    # Below u ~ 1e-4 the exact form loses the answer to cancellation -- measured, the two differ
    # by 4.3% in I_ac at u = 1e-5, and there it is the EXACT branch that is wrong. Above u ~ 1e-1
    # the truncated series runs out. The window between is where they can be cross-checked, and
    # agreement there is what says the transcription and the expansion describe the same
    # function.
    global _SERIES_U
    keep, worst = _SERIES_U, 0.0
    for u in (3e-4, 1e-3, 3e-3):
        c = a * np.sqrt(1.0 - u)
        _SERIES_U = 0.0
        exact = shape_integrals(a, c)
        _SERIES_U = 1.0
        series = shape_integrals(a, c)
        for nm, x, y in zip(('I_a', 'I_c', 'I_ac', 'I_cc', 'I_aa', 'I_ab'), exact, series):
            worst = max(worst, abs(x - y) / max(abs(x), 1e-30))
    _SERIES_U = keep
    good = worst < 1e-5
    ok &= good
    if verbose:
        print(f"  [{'PASS' if good else 'FAIL'}] series/exact agree over u = 3e-4..3e-3: "
              f"worst relative difference {worst:.2e} (tolerance 1e-5)")
    return ok


if __name__ == '__main__':
    import sys
    sys.path.insert(0, __file__.rsplit('\\', 1)[0] if '\\' in __file__ else '.')
    print("eshelby.py self-test (shape integrals only; physics is in "
          "scripts/ellipsoid_strain_validation.py)")
    sys.exit(0 if _selftest() else 1)
