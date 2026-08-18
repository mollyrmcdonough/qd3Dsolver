"""One coherently strained island in a matrix: strain, band landscape, confinement, states.

General in the material pair. `ingasb_dot.py` was the original of this code and is now the
InAs/In(x)Ga(1-x)Sb narrative layer over it; everything here works for any dot and any matrix
`materials.py` can supply, in any band alignment.

    import heterostructure as hs
    env = hs.build(hs.spherical_lens(10.0, 5.0), 'InSb', matrix='GaAs')
    hs.confinement(env)

Why the confinement test does not branch on alignment type
----------------------------------------------------------
The obvious design is to ask whether the system is type I, type II or broken gap and then look
for the electron in the dot or in the matrix accordingly. That is fragile: strain moves the band
edges by hundreds of meV and can change the answer relative to the unstrained alignment, the
alignment can differ between the middle of the island and its rim, and a "type" is a property of
two bulk materials rather than of a strained 3D structure.

So `confinement` never asks. For each carrier it builds a BINDING DEPTH FIELD -- how far the
local edge lies on the confining side of the far-field matrix value --

    electron:  D = E_c(far) - E_c(r)
    hole:      D = E_v(r)   - E_v(far)          (E_v from the exact k = 0 valence edge)

and the well is wherever D > 0, whatever region that turns out to be. A type-I dot puts it inside
the mask; the antimonides put the electron's outside it, in the tensile shell the island's
compression creates around itself. Same code, and `frac_in_dot` reports which happened rather
than assuming it. `alignment_type` is carried alongside as a LABEL, from `materials.alignment`,
and nothing branches on it.

The binding criterion, and what it is worth
-------------------------------------------
A finite spherical well of radius R and depth V0 holds at least one bound state only if

    V0 R^2  >  pi^2 hbar^2 / (8 m*)

`binding_threshold` returns the right-hand side. This is a necessary condition applied to an
equivalent sphere of the same volume as the real well, so:

  FAILING it is decisive -- the real well is never better at binding than a compact sphere of
  the same volume, and a thin shell (which is what the antimonide electron pocket is) is much
  worse.
  PASSING it is only suggestive. Confirm with a real solve, `electron_states` or
  `eight_band_states`.

It is used instead of "just solve for the ground state" because in a shallow well a finite-box
solve returns box zero-point energy dressed up as a binding energy: it barely responds to the
well depth, and it changes if the padding changes. `V0 R^2` against a threshold says the same
thing without the box in it. That failure mode was real here -- see `ingasb_dot`, where doubling
the pocket depth moved the "ground state" by under 2 meV.

Critical size, and why one solve is enough
------------------------------------------
Continuum elasticity has no length scale, so at fixed SHAPE the strain field and every band edge
are independent of the island's size -- verified to 0.0 meV over a 4x range in
`scripts/insb_band_edges_vs_size.py`, and exactly, not approximately: scaling the grid and box
with the island makes every size the same discrete problem. V0 is therefore size-independent and
R is exactly proportional to the island's linear size, so from ONE converged solve

    L_critical = L * sqrt( threshold / (V0 R^2) )

`critical_size` returns it. Below `L_critical` the well cannot hold the carrier at all; above it,
binding is possible and worth a real solve. That is the whole point of the scale-invariance
result: it converts an expensive size sweep into an extrapolation.

The caveat is that this is exact for the BAND EDGES and the elasticity, not for everything. It
assumes the shape is held fixed, the box stays converged (see `scripts/insb_box_convergence.py`
-- a tight periodic box makes the island stiffer and biases the edges), and it ignores Coulomb
binding entirely. A carrier that fails the single-particle test here can still be bound to its
partner as an exciton, which is exactly what happens in the broken-gap antimonides.
"""
import time

import numpy as np
import scipy.sparse.linalg as spla

import qdsolver_core as qd
import elasticity_fd as ef
import piezoelectric as pz
import kp_confined as kpc
import kp_pryor as kp
import eigensolvers as eig
import materials as mt


# --------------------------------------------------------------------------------------
# Shapes
# --------------------------------------------------------------------------------------

def lens(radius, height):
    """Circular lens (half-ellipsoid dome) of base radius `radius` and height `height`."""
    return dict(
        kind='lens',
        label=f"lens r = {radius:g} nm, h = {height:g} nm",
        extent=(2 * radius, 2 * radius, height),
        volume=qd.lens_volume(radius, height),
        mask=lambda X, Y, Z: qd.lens_mask(X, Y, Z, radius, height),
        params=dict(radius=radius, height=height),
        scale=2.0 * radius,
    )


def ellipsoid(base, height):
    """Oblate ellipsoid of base DIAMETER `base` and total height `height`.

    The dot geometry of Yeap et al., Phys. Rev. B 79, 075305 (2009), Fig. 1 -- a full ellipsoid,
    not the half-ellipsoid `lens` returns.

    **AR = d/h**, diameter over height, so AR = 2 means an island twice as wide as tall and a
    LARGER AR is a FLATTER island. This is the literature convention (Yeap, Rybchenko) and, since
    2026-08-05, this package's convention everywhere. Earlier work here quoted h/d -- the
    reciprocal -- so a stored record or a plot axis predating that change reads backwards. Sweeps
    written before it carry `aspect` (h/d); everything since carries `AR`.

    Arguments are the base diameter and total height rather than semi-axes, to match how the
    paper (and experiment) quote island dimensions.
    """
    rx = base / 2.0
    rz = height / 2.0
    return dict(
        kind='ellipsoid',
        label=f"ellipsoid base = {base:g} nm, h = {height:g} nm (AR = {base/height:.3g})",
        extent=(base, base, height),
        volume=4.0 / 3.0 * np.pi * rx * rx * rz,
        mask=lambda X, Y, Z: qd.ellipsoid_mask(X, Y, Z, rx, rx, rz),
        params=dict(base=base, height=height, aspect_ratio=base / height),
        scale=base,
    )


def spherical_lens(radius, height):
    """Lens cut from a sphere, the dot geometry of Pryor & Pistol Fig. 1(a).

    Use this, not `lens`, when comparing against their tables: `lens` is a half-ellipsoid, which
    stands vertically at its rim and encloses 23% more volume at AR = 4 (h = d/4, the aspect ratio
    Pryor & Pistol quote as h/d = 1/4). See `qdsolver_core.spherical_cap_mask`.

    AR is reported as d/h throughout this package -- see `ellipsoid` for the convention note.
    """
    return dict(
        kind='spherical_lens',
        label=f"spherical lens r = {radius:g} nm, h = {height:g} nm (AR = {2*radius/height:.3g})",
        extent=(2 * radius, 2 * radius, height),
        volume=qd.spherical_cap_volume(radius, height),
        mask=lambda X, Y, Z: qd.spherical_cap_mask(X, Y, Z, radius, height),
        params=dict(radius=radius, height=height, aspect_ratio=2.0 * radius / height),
        scale=2.0 * radius,
    )


def dash(length, width, height, contact_angle_deg=qd.DASH_CONTACT_ANGLE_DEG):
    """Elongated faceted island (truncated rectangular pyramid), after Tersoff and Tromp,
    Phys. Rev. Lett. 70, 2782 (1993). See `qdsolver_core.dash_mask` for the geometry and for the
    provenance limits on it."""
    return dict(
        kind='dash',
        label=(f"dash {length:g} x {width:g} x {height:g} nm, "
               f"facets {contact_angle_deg:g} deg, aspect {length/width:.1f}:1"),
        extent=(length, width, height),
        volume=qd.dash_volume(length, width, height, contact_angle_deg),
        mask=lambda X, Y, Z: qd.dash_mask(X, Y, Z, length, width, height, contact_angle_deg),
        params=dict(length=length, width=width, height=height,
                    contact_angle_deg=contact_angle_deg),
        scale=float(width),
    )


def dash_base_for_aspect(aspect, height, target_volume,
                         contact_angle_deg=qd.DASH_CONTACT_ANGLE_DEG):
    """Base dimensions of a dash with a given aspect ratio, holding BOTH volume and height fixed.

    Returns (length, width), or None if no base can hold `target_volume` at this height and
    facet angle.

    This is the controlled way to vary shape. The obvious alternative -- fix the base area and
    solve for height -- forces the narrow end of the series toward its geometric ceiling, where
    the flat top has almost vanished and the island is really a pointed ridge. Then the series
    varies aspect ratio AND top-face fraction AND height at once, and any trend in the answer
    cannot be attributed to elongation. Holding height and volume fixed leaves aspect ratio as
    the only thing that changes.

    The width still has a floor: the facets must not meet below `height`, i.e.
    width >= 2*height/tan(theta). Past that the shape is impossible at any length.
    """
    t = np.tan(np.deg2rad(contact_angle_deg))
    w_min = 2.0 * height / t
    lo, hi = w_min, max(w_min * 2.0, 10.0)
    vol = lambda w: qd.dash_volume(w * aspect, w, height, contact_angle_deg)
    while vol(hi) < target_volume:
        hi *= 2.0
        if hi > 1e5:
            return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if vol(mid) < target_volume:
            lo = mid
        else:
            hi = mid
    w = 0.5 * (lo + hi)
    return w * aspect, w


def dash_height_for_volume(length, width, target_volume,
                           contact_angle_deg=qd.DASH_CONTACT_ANGLE_DEG):
    """Height that gives a dash of `target_volume`, or None if the island is too narrow to hold
    it at this facet angle. Use to build constant-volume aspect-ratio series, where any change
    in the answer is shape rather than size -- the facet set-back removes proportionally more
    material from a narrow island, so fixing length*width*height does NOT fix the volume."""
    lo, hi = 1e-9, (width / 2.0) * np.tan(np.deg2rad(contact_angle_deg))
    if qd.dash_volume(length, width, hi, contact_angle_deg) < target_volume:
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if qd.dash_volume(length, width, mid, contact_angle_deg) < target_volume:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------

def material_spec(spec, alloy='InGaSb'):
    """Resolve a dot or matrix specification into a material dict.

        material_spec(0.35)                -> In0.35Ga0.65Sb   (a float is an alloy fraction)
        material_spec(0.5, 'InAsSb')       -> InAs0.50Sb0.50
        material_spec(('InAsSb', 0.5))     -> the same, naming the alloy inline
        material_spec('GaSb')              -> the binary
        material_spec(mt.material('InAs')) -> passed through

    The float case exists because every caller of `ingasb_dot.build` predates the generalisation
    and passes a bare In(x)Ga(1-x)Sb fraction positionally. Keeping that meaning is what lets the
    notebooks and the benchmark scripts run unchanged. The tuple form is what to use in new code
    that mixes alloys, since it does not depend on the `alloy` argument being set correctly.
    """
    if isinstance(spec, (dict, str, tuple, list)):
        return mt.as_material(spec)
    return mt.alloy(alloy, float(spec))


def build(shape, dot, h=0.5, pad=6.0, z_pad=None, use_piezo=True, matrix='InAs',
          alloy='InGaSb', vol_tol=0.15, verbose=True, require_converged=True):
    """Grid, mask, strain, piezoelectric potential and band-edge fields for one configuration.

    `dot` may be an alloy fraction (float, interpreted in `alloy`), a binary name, or a material
    dict -- see `material_spec`. `matrix` likewise. `shape` comes from `lens`, `spherical_lens`
    or `dash`.

    The strain is solved with `elasticity_fd`, i.e. Pryor's real-space method with
    position-dependent elastic constants, not the Fourier solver. That is not a stylistic choice:
    a homogeneous solver gets the stiffness contrast wrong, and it is wrong exactly where it
    matters most for a type-II or broken-gap system, in the matrix immediately around the island
    where the reaction strain digs the pocket that holds the expelled carrier.

    `vol_tol` guards the mask: a faceted island on a cubic grid is staircased, and the volume
    error is first order in h. Exceeding the tolerance raises rather than warns, because a mask
    that is 15% oversized makes every energy below wrong by about as much.

    Energy zero is the UNSTRAINED MATRIX VALENCE EDGE, so `Ev_far = 0` and `Ec_far = Eg(matrix)`
    whatever the materials are.
    """
    dot = material_spec(dot, alloy)
    mat = material_spec(matrix, alloy)

    cx, cy, cz, X, Y, Z = qd.island_grid(shape['extent'], h, pad, z_pad=z_pad)
    mask = shape['mask'](X, Y, Z)
    if not mask.any():
        raise ValueError(f"empty mask: the island is smaller than one cell at h = {h}")

    vol_err = qd.mask_volume_error(mask, h, shape['volume'])
    if abs(vol_err) > vol_tol:
        raise ValueError(
            f"mask volume error {vol_err:+.3f} exceeds {vol_tol}: refine h (currently {h}) or "
            f"enlarge the island. Volume error is first order in h for a faceted shape.")
    for ax in (0, 1):
        a = qd.mirror_asymmetry(mask, ax)
        if a:
            raise ValueError(f"mask is not mirror symmetric about axis {ax} ({a} voxels); the "
                             f"grid axis is not symmetric to machine precision")

    eps_star = mt.misfit(dot, mat)
    t0 = time.time()
    # require_converged is the default and is left on deliberately: an unconverged or
    # preconditioner-filtered strain field produces band edges that look plausible and are wrong,
    # and it used to be reported only under verbose -- which is how a whole box-convergence study
    # got run on two silently broken large grids.
    strain = ef.solve_strain_fd(mask, eps_star, mt.elastic(dot), mt.elastic(mat), h, tol=1e-10,
                                require_converged=require_converged)
    t_strain = time.time() - t0

    phi = np.zeros(mask.shape)
    if use_piezo:
        e14 = np.where(mask, dot['e14'], mat['e14'])
        phi = pz.potential(strain, e14, mat['eps_R'], h)

    Ec, Ev = mt.band_edge_fields(mask, strain.trace, dot, mat)
    U = -phi                       # electrostatic potential ENERGY of an electron
    Ec, Ev = Ec + U, Ev + U

    g = strain.at(mask)
    env = dict(shape=shape, dot=dot, matrix=mat, h=h, pad=pad,
               x=dot.get('x'), alloy=dot.get('system'),
               cx=cx, cy=cy, cz=cz, X=X, Y=Y, Z=Z, mask=mask,
               strain=strain, phi=phi, Ec=Ec, Ev=Ev, eps_star=eps_star,
               vol_err=vol_err, use_piezo=use_piezo,
               Ec_far=float(mat['Eg']), Ev_far=0.0)   # unstrained matrix edges, the energy zero
    env['alignment'] = mt.alignment(dot, mat, g['exx'] + g['eyy'] + g['ezz'])

    if verbose:
        tr = g['exx'] + g['eyy'] + g['ezz']
        a = env['alignment']
        print(f"{shape['label']}   dot = {dot['name']} in {mat['name']}")
        print(f"  grid {mask.shape} = {mask.size:,} pts, h = {h} nm, "
              f"box {len(cx)*h:.0f} x {len(cy)*h:.0f} x {len(cz)*h:.0f} nm")
        print(f"  island {mask.sum():,} cells, volume {shape['volume']:.1f} nm^3, "
              f"vol err {vol_err:+.3f}")
        print(f"  misfit {eps_star*100:+.2f}%  ->  <Tr eps> = {tr:+.5f}  "
              f"(exx {g['exx']:+.5f}, eyy {g['eyy']:+.5f}, ezz {g['ezz']:+.5f})")
        print(f"  strain: CG {strain.cg_iterations} iters, rel {strain.cg_residual:.1e}, "
              f"{strain.cg_modes_dropped} mode(s) dropped, {t_strain:.1f}s")
        if use_piezo:
            print(f"  piezo {phi.min()*1e3:+.1f} .. {phi.max()*1e3:+.1f} meV")
        print(f"  alignment (strained, hydrostatic only): {a['type']}, "
              f"e-well {a['e_well']*1e3:+.0f} meV, h-well {a['h_well']*1e3:+.0f} meV")
        print(f"  E_c: dot {Ec[mask].mean():.3f}, matrix min {Ec[~mask].min():.3f}, "
              f"far {env['Ec_far']:.3f} eV")
    return env


# --------------------------------------------------------------------------------------
# Band landscape
# --------------------------------------------------------------------------------------

def valence_edge(env, hydrostatic_applied=True):
    """Exact k = 0 top of the local valence band, everywhere on the grid.

    This is the largest eigenvalue of the 6x6 valence block of the strain Hamiltonian, so it
    includes the full Bir-Pikus shear terms, not just the hydrostatic shift already in `Ev`. The
    shear is not a correction here: it splits heavy and light hole by hundreds of meV in a dot
    strained by several percent, and it is what sets the hole well depth.
    """
    m = env['mask']
    pick = lambda k: np.where(m, env['dot'][k], env['matrix'][k])
    return kp.local_band_edges(env['strain'], Ev=env['Ev'], Ec=env['Ec'],
                               delta_so=pick('delta_so'), a_c=pick('a_c'), a_v=pick('a_v'),
                               b=pick('b'), d=pick('d'),
                               hydrostatic_applied=hydrostatic_applied)


def cut_indices(env, through='centre'):
    """Index tuples for the three principal cuts through the island."""
    ix, iy = len(env['cx']) // 2, len(env['cy']) // 2
    iz = int(np.argmin(np.abs(env['cz'] - env['shape']['extent'][2] / 2.0)))
    return dict(x=(slice(None), iy, iz), y=(ix, slice(None), iz), z=(ix, iy, slice(None)))


def band_profiles(env):
    """Conduction and valence edges along [100], [010] and [001] through the island."""
    edges = valence_edge(env)
    cuts = cut_indices(env)
    axes = dict(x=env['cx'], y=env['cy'], z=env['cz'])
    out = {}
    for k, idx in cuts.items():
        out[k] = dict(r=axes[k], Ec=edges['cb'][idx], v1=edges['v1'][idx],
                      v2=edges['v2'][idx], v3=edges['v3'][idx],
                      inside=env['mask'][idx])
    out['kramers'] = edges['kramers']
    return out


# --------------------------------------------------------------------------------------
# Confinement
# --------------------------------------------------------------------------------------

def binding_threshold(m_star):
    """V0 R^2 (eV nm^2) a finite spherical well needs to hold ANY bound state: pi^2 hbar^2/(8 m).

    Note it scales as 1/m: a light carrier is much harder to confine. The InAs electron at
    m = 0.026 needs 3.66 eV nm^2, a heavy hole at m = 0.35 needs 0.27 -- a factor of 13, and it
    is why these systems bind holes long before they bind electrons.
    """
    return np.pi ** 2 * qd.HBAR2_OVER_2M0 / (4.0 * m_star)


def hole_mass(mat):
    """Spherical-approximation heavy-hole mass, 1/(gamma1 - 2*gamma_bar), gamma_bar =
    (2 gamma2 + 3 gamma3)/5.

    The Baldereschi-Lipari spherical average, used because the binding criterion is written for
    an isotropic well and the true [001]/[111] masses differ by a factor of ~2 in these
    materials. The HEAVY hole is the right one for a binding threshold: a heavier carrier binds
    more easily, so if even the heavy hole fails the test, nothing in the valence band binds.
    Under strong shear the top of the valence band is not purely heavy-hole-like, so treat this
    as the necessary-condition estimate it is.
    """
    g_bar = (2.0 * mat['gamma2L'] + 3.0 * mat['gamma3L']) / 5.0
    return 1.0 / (mat['gamma1L'] - 2.0 * g_bar)


def _depth_field(env, carrier, edges=None):
    """(D, mass) where D > 0 marks where `carrier` is bound, and by how much (eV).

    Electron: D = E_c(far) - E_c(r), from the strained conduction edge including piezo.
    Hole:     D = E_v(r) - E_v(far), from the EXACT k = 0 valence edge, so the Bir-Pikus shear
              is included -- it is what sets the hole well depth and omitting it is a
              hundreds-of-meV error in a strongly strained island.
    """
    m = env['mask']
    if carrier == 'electron':
        D = env['Ec_far'] - env['Ec']
        mass = np.where(m, mt.electron_mass(env['dot']), mt.electron_mass(env['matrix']))
    elif carrier == 'hole':
        e = valence_edge(env) if edges is None else edges
        D = e['v1'] - env['Ev_far']
        mass = np.where(m, hole_mass(env['dot']), hole_mass(env['matrix']))
    else:
        raise ValueError(f"carrier must be 'electron' or 'hole', got {carrier!r}")
    return D, mass


def _touches_boundary(sel):
    """Does the selected region reach the edge of the periodic cell?

    If it does, the region is not localised by the structure -- it wraps, and its volume is set
    by how much matrix was included rather than by any physics. See `well_metrics`.
    """
    return bool(sel[0].any() or sel[-1].any()
                or sel[:, 0].any() or sel[:, -1].any()
                or sel[:, :, 0].any() or sel[:, :, -1].any())


def well_core(env, carrier, frac=0.9, edges=None):
    """Where the DEEPEST part of the well sits: the region within `frac` of the maximum depth.

    This is the right thing to ask when the question is "which side of the interface does this
    carrier live on", and `well_metrics`' `frac_in_dot` is the wrong thing. That column is
    computed per contour, and the contour `critical_size` selects is the one maximising V0R^2 --
    which is the shallowest surviving one, and therefore the most contaminated by the diffuse
    halo the strained matrix creates. Read off that contour, an InAs dot in GaAs appears to put
    its hole in the matrix, which is nonsense: it is the textbook type-I dot. Read off the well
    core, the hole is 100% inside.

    Returns the maximum depth, the core volume, and the fraction of the core inside the island.
    """
    D, _ = _depth_field(env, carrier, edges)
    dmax = float(D.max())
    if dmax <= 0.0:
        return dict(depth_max=dmax, volume=0.0, frac_in_dot=float('nan'))
    sel = D > frac * dmax
    n = int(sel.sum())
    return dict(depth_max=dmax, volume=n * env['h'] ** 3,
                frac_in_dot=float((sel & env['mask']).sum()) / n)


def well_metrics(env, carrier, depths=(0.0, 0.01, 0.025, 0.05, 0.10), edges=None):
    """Volume, equivalent radius and binding verdict of the well that holds `carrier`.

    One row per entry in `depths`: the region where the carrier is bound by MORE than that much,
    which strips the shallow fringe off the well and shows how fast it shrinks.

    Each row carries `frac_in_dot`, the fraction of the well volume inside the island. That is
    the number that says whether this is a dot-confined carrier or one expelled into a pocket in
    the matrix -- read it instead of assuming from the alignment type.

    THE `box_limited` FLAG IS NOT DECORATION. A row is flagged when its region reaches the edge
    of the periodic cell, and such a row must be discarded: its volume is set by how much matrix
    the box happens to include, not by the structure. This is easy to hit and it looks like a
    result. The island's compression puts the surrounding matrix into tension, which RAISES the
    local valence edge over a broad shallow halo, so for an InSb lens in InAs the depth = 0 hole
    "well" measures

        pad 7.5 nm -> 23,357 nm^3      pad 12 nm -> 53,870      pad 18 nm -> 124,418

    -- growing without bound, and 96% of it outside the island. Trim to a deeper contour and it
    converges: the 25 meV region is 7,300 / 7,760 / 7,785 nm^3 over the same paddings and the
    100 meV region is flat at ~1,440. Those are wells; the depth = 0 region is the box.

    The equivalent radius is that of a sphere of the same volume, and `mass` is the well's own
    volume-weighted mean, so a well straddling the interface gets a sensible mixture rather than
    one material's value.
    """
    D, mass = _depth_field(env, carrier, edges)
    m, h = env['mask'], env['h']
    cell = h ** 3

    rows = []
    for t in depths:
        sel = D > t
        n = int(sel.sum())
        if n == 0:
            rows.append(dict(depth=t, volume=0.0, R=0.0, mean_depth=0.0, VR2=0.0,
                             mass=float('nan'), threshold=float('nan'), frac_in_dot=float('nan'),
                             box_limited=False, binds=False))
            continue
        vol = n * cell
        R = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        mean_depth = float(D[sel].mean())
        m_star = float(mass[sel].mean())
        thr = binding_threshold(m_star)
        rows.append(dict(depth=t, volume=vol, R=R, mean_depth=mean_depth,
                         VR2=mean_depth * R ** 2, mass=m_star, threshold=thr,
                         frac_in_dot=float((sel & m).sum()) / n,
                         box_limited=_touches_boundary(sel),
                         binds=mean_depth * R ** 2 > thr))
    return rows


def critical_size(env, carrier, depths=(0.0, 0.01, 0.025, 0.05, 0.10), edges=None):
    """Smallest island of THIS SHAPE that could confine `carrier`, in nm.

    Returns `scale` (the island's current linear size, `shape['scale']`), `factor` (how much it
    must grow), `critical` (scale * factor), and the row the answer came from.

    Exact within the model, from a single solve, because continuum elasticity has no length
    scale: at fixed shape every band edge is size-independent, so V0 is fixed and R is exactly
    proportional to the island's linear size. `V0 R^2 > threshold` is then a condition on size
    alone, and the factor is sqrt(threshold / V0R^2). See the module docstring for what this does
    NOT account for -- Coulomb binding above all, which is what actually holds the carrier in
    the broken-gap antimonides.

    ROWS FLAGGED `box_limited` ARE DISCARDED, which is the single most important thing this
    function does. Such a region reaches the wall of the periodic cell, so its volume is a
    property of the padding rather than of the island -- see `well_metrics` for the measured
    growth. Using one would produce a critical size that changes when you change the box, with
    nothing in the output to say so.

    Of what survives it takes the LARGEST V0R^2 rather than the depth = 0 row, because the
    shallowest surviving contour is still the most halo-contaminated one: trimming deeper finds
    the compact well underneath, at a higher V0R^2 and a much higher fraction inside the island.
    Taking the maximum makes a pass easier and a failure harder to reach, which is the right
    direction -- the criterion is only ever decisive when it FAILS, so it should be evaluated on
    the most favourable sub-well available. For the antimonide electron every contour fails,
    0.5 to 0.7 against a threshold of 3.61, so that conclusion cannot be blamed on the choice of
    contour.

    `factor <= 1` means the current island already clears the threshold. `factor` is infinite,
    with `box_limited=True`, when every contour was discarded -- enlarge the box and re-run
    rather than reading anything into it.
    """
    all_rows = well_metrics(env, carrier, depths, edges=edges)
    rows = [r for r in all_rows if r['volume'] > 0 and not r['box_limited']]
    scale = env['shape'].get('scale')
    if not rows or scale is None:
        return dict(scale=scale, factor=float('inf'), critical=float('inf'),
                    VR2=0.0, threshold=float('nan'), carrier=carrier, at_depth=None,
                    box_limited=any(r['box_limited'] for r in all_rows))
    best = max(rows, key=lambda r: r['VR2'])
    factor = float(np.sqrt(best['threshold'] / best['VR2']))
    core = well_core(env, carrier, edges=edges)
    return dict(scale=float(scale), factor=factor, critical=float(scale) * factor,
                VR2=best['VR2'], threshold=best['threshold'], mass=best['mass'],
                frac_in_dot=best['frac_in_dot'], core_in_dot=core['frac_in_dot'],
                depth_max=core['depth_max'], at_depth=best['depth'], carrier=carrier,
                box_limited=False)


def confinement(env, depths=(0.0, 0.01, 0.025, 0.05, 0.10), verbose=True):
    """Where each carrier is confined, how strongly, and the smallest island that could hold it.

    Returns {'electron': ..., 'hole': ...}, each with `rows` from `well_metrics` and `critical`
    from `critical_size`, plus the alignment label.

    Nothing here branches on the alignment type -- see the module docstring. The `in dot` column
    is what tells you where the carrier actually sits.
    """
    edges = valence_edge(env)
    out = dict(alignment=env['alignment'])

    if verbose:
        a = env['alignment']
        print(f"{env['shape']['label']}   {env['dot']['name']} in {env['matrix']['name']}")
        print(f"  alignment {a['type']}   (unstrained overlap "
              f"E_v(dot)-E_c(matrix) = {a['overlap']*1e3:+.0f} meV)")

    for carrier in ('electron', 'hole'):
        rows = well_metrics(env, carrier, depths, edges=edges)
        crit = critical_size(env, carrier, depths, edges=edges)
        out[carrier] = dict(rows=rows, critical=crit)

        if not verbose:
            continue
        top = rows[0]
        print(f"\n  {carrier.upper()}")
        if top['volume'] <= 0:
            print("    no region binds it at all: the far-field matrix edge is never crossed.")
            continue
        print(f"    m* = {top['mass']:.4f} m0  ->  needs V0 R^2 > "
              f"{top['threshold']:.2f} eV nm^2")
        print(f"    {'below':>7} {'volume':>10} {'equiv R':>9} {'mean depth':>11} "
              f"{'V0R^2':>8} {'in dot':>8}  verdict")
        for r in rows:
            if r['volume'] <= 0:
                print(f"    {r['depth']*1e3:>6.0f}m {'--':>10}")
                continue
            verdict = ('REACHES THE BOX WALL -- discarded' if r['box_limited']
                       else 'could bind' if r['binds'] else 'too weak')
            print(f"    {r['depth']*1e3:>6.0f}m {r['volume']:>10.1f} {r['R']:>9.2f} "
                  f"{r['mean_depth']*1e3:>10.1f}m {r['VR2']:>8.2f} "
                  f"{r['frac_in_dot']*100:>7.0f}%  {verdict}")
        if crit.get('box_limited'):
            print("    every contour reaches the box wall, so no size can be quoted: the region")
            print("    is bounded by the padding, not by the island. Enlarge pad and re-run.")
            continue
        at = '' if crit['at_depth'] is None else f" [best contour: {crit['at_depth']*1e3:.0f} meV]"
        if crit['factor'] <= 1.0:
            print(f"    clears the threshold at this size "
                  f"({crit['scale']:.1f} nm, {1/crit['factor']:.2f}x margin){at}")
        elif np.isfinite(crit['factor']):
            print(f"    would need the island {crit['factor']:.2f}x larger: "
                  f"{crit['critical']:.1f} nm against {crit['scale']:.1f} nm now{at}")

    if verbose:
        print("\n  Failing this test is decisive; passing it is only suggestive -- it is a")
        print("  necessary condition on an equivalent sphere, and a real well of the same")
        print("  volume binds no better. Confirm a pass with electron_states or")
        print("  eight_band_states. Coulomb binding is not included.")
    return out


# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------

def electron_states(env, k=4, verbose=True, mass_mode='differential'):
    """Single-band conduction states. Bound iff E < `env['Ec_far']` = Eg(matrix).

    `mass_mode` is passed to `materials.electron_mass_field`: 'unstrained' (the behaviour before
    this argument existed), 'absolute', or 'differential' (default -- the tabulated band-edge mass
    scaled by the Kane response to the local strained gap).

    The potential is the strained conduction edge including the piezoelectric term; the mass is
    the Kane-derived band-edge mass of each material (`materials.electron_mass`). Energies are on
    the same zero as everything else, the unstrained matrix valence edge.

    `inside` is the fraction of each state's density within the island. In a type-I system that
    should be large; in a broken-gap one it should be SMALL, because the electron is expelled
    from the dot and lives in the tensile shell around it. Either way it is the check that the
    state is the one you meant to find.
    """
    m = env['mask']
    # `mass_mode` folds the strain-induced band-gap change into the electron mass, which Yeap et
    # al. state they include. It is not a small correction to the MASS -- coherent strain opens
    # the InSb gap fourfold and triples m_e -- but the electron here lives in the matrix, where
    # the strain is far weaker, so its effect on the LEVEL is much smaller. Report the spread
    # across modes rather than trusting one; see `materials.electron_mass_field`.
    m_e = mt.electron_mass_field(m, env['dot'], env['matrix'], env['strain'].trace,
                                 mode=mass_mode)
    E, V = qd.solve_states(m_e, env['Ec'], env['h'], n_states=k)

    inside = []
    for j in range(V.shape[1]):
        rho = np.abs(V[:, j]) ** 2
        inside.append(float(rho.reshape(m.shape)[m].sum() / rho.sum()))
    inside = np.array(inside)

    if verbose:
        print(f"  electron states (bound below E_c({env['matrix']['name']}, far) = "
              f"{env['Ec_far']:.3f} eV):")
        for j, (e, f) in enumerate(zip(E, inside)):
            tag = 'bound' if e < env['Ec_far'] else 'unbound (box state)'
            print(f"    {j}: E = {e:.4f} eV, binding {(env['Ec_far']-e)*1e3:+7.1f} meV, "
                  f"{f*100:5.1f}% inside the dot   {tag}")
    return dict(E=E, V=V, inside=inside, n_bound=int((E < env['Ec_far']).sum()))


def embed_electron_fields(env, L, mass_mode='differential'):
    """Conduction edge and electron mass on a box of side >= `L` nm, padded with FAR-FIELD values.

    The inverse of `kp_planewave.crop_env`, and it exists for the opposite reason. The hole is
    bound by hundreds of meV and decays within a nanometre, so it wants a SMALL box. The electron
    in a broken-gap system is a weakly bound MATRIX state in the shallow tensile shell outside the
    island; at ~17 meV binding its decay length sqrt(G0/(m E)) is 9.3 nm at m = 0.026, so it wants
    a box of several times that. Solving both on one grid is impossible: fine enough for the dot
    and large enough for the electron is millions of points.

    Padding with constants rather than resampling is exact here, and that is the point. The strain
    of an inclusion decays as 1/r^3, so ten nanometres out from a 2.5 nm island the fields ARE
    their far-field values -- Ec_far and the unstrained matrix mass. The grid spacing never
    changes, so nothing is interpolated and the inner region is bit-identical to `env`.

    The caller is responsible for `env` itself being padded far enough that this is true;
    `verbose` on `build` reports the strain at the box faces.
    """
    h = env['h']
    n_want = int(np.ceil(L / h))
    m_e = mt.electron_mass_field(env['mask'], env['dot'], env['matrix'], env['strain'].trace,
                                 mode=mass_mode)
    # The far field is unstrained matrix, so evaluate the same expression there rather than
    # assuming which branch of `mass_mode` collapses to the tabulated value.
    m_far = float(mt.electron_mass_field(np.zeros((1, 1, 1), bool), env['dot'], env['matrix'],
                                         0.0, mode=mass_mode).ravel()[0])

    pads, shape = [], []
    for n in env['mask'].shape:
        extra = max(n_want - n, 0)
        lo = extra // 2
        pads.append((lo, extra - lo))
        shape.append(n + extra)
    Ec = np.pad(env['Ec'], pads, constant_values=env['Ec_far'])
    me = np.pad(m_e, pads, constant_values=m_far)
    mask = np.pad(env['mask'], pads, constant_values=False)
    return dict(Ec=Ec, m_e=me, mask=mask, h=h, m_far=m_far, pads=pads,
                L=[s * h for s in shape], Ec_far=float(env['Ec_far']))


def electron_states_bigbox(env, L, k=2, mass_mode='differential', tol=1e-7, maxiter=600,
                           verbose=True):
    """Lowest conduction states on a box of side >= `L` nm. Matrix-free; LOBPCG, not shift-invert.

    `heterostructure.electron_states` factorizes the Hamiltonian with a sparse LU, which is the
    right choice up to a few hundred thousand points and impossible past it -- 3D LU fill-in. The
    boxes this needs are 1-2 million points, so the solve is iterative.

    The preconditioner is what makes that cheap, and it is close to exact here: outside a few
    nanometres of the island the operator IS the constant-coefficient Laplacian at the matrix
    mass, and `build_hamiltonian` uses hard-wall (Dirichlet) boundaries, which the type-I discrete
    sine transform diagonalizes EXACTLY. So the preconditioner inverts the far-field operator in
    one DST pair and only the small strained region is left for the iteration to deal with.

    WHY THE BOX IS THE WHOLE PROBLEM. A hard-walled box adds 3*G0*pi^2/(m L^2) of pure
    quantisation energy, which at m = 0.026 is 645 meV at L = 8.1 nm -- measured, and it is why
    this state was previously reported as unbound. It falls as 1/L^2: 4.0 meV at L = 65 nm. Since
    the binding being measured is ~17 meV, the box must be run out and the trend shown, never
    assumed. `scripts/electron_binding.py` does that.
    """
    from scipy.fft import dstn, idstn

    f = embed_electron_fields(env, L, mass_mode=mass_mode)
    Ec, me, h = f['Ec'], f['m_e'], f['h']
    H = qd.build_hamiltonian(me, Ec, h)
    n = H.shape[0]
    sigma = float(Ec.min()) - 1e-3

    # Exact symbol of the Dirichlet second difference at the FAR-FIELD mass: the DST-I basis
    # function j has eigenvalue 2t(1 - cos(pi (j+1)/(N+1))) along each axis.
    t_far = qd.HBAR2_OVER_2M0 / (f['m_far'] * h ** 2)
    lam = np.zeros(Ec.shape)
    for ax, N in enumerate(Ec.shape):
        j = np.arange(1, N + 1)
        w = 2.0 * t_far * (1.0 - np.cos(np.pi * j / (N + 1)))
        lam = lam + w.reshape([-1 if a == ax else 1 for a in range(3)])
    denom = lam + f['Ec_far'] - sigma

    def precond(x):
        out = np.empty_like(x)
        for c in range(x.shape[1]):
            v = x[:, c].reshape(Ec.shape)
            out[:, c] = idstn(dstn(v, type=1, norm='ortho') / denom,
                              type=1, norm='ortho').ravel()
        return out

    M = spla.LinearOperator((n, n), matvec=lambda v: precond(v.reshape(-1, 1)).ravel(),
                            matmat=precond, dtype=float)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, k))
    E, V = spla.lobpcg(H, X, M=M, largest=False, tol=tol, maxiter=maxiter)
    order = np.argsort(E)
    E, V = E[order], V[:, order]

    inside, ring = [], []
    for j in range(V.shape[1]):
        rho = (V[:, j] ** 2).reshape(Ec.shape)
        rho = rho / rho.sum()
        inside.append(float(rho[f['mask']].sum()))
        ring.append(float(rho[(~f['mask']) & (Ec < f['Ec_far'] - 1e-4)].sum()))
    box_quantum = 3.0 * qd.HBAR2_OVER_2M0 * np.pi ** 2 / (f['m_far'] * min(f['L']) ** 2)

    if verbose:
        print(f"  box {f['L'][0]:.1f} x {f['L'][1]:.1f} x {f['L'][2]:.1f} nm = {n:,} pts, "
              f"empty-box quantum {box_quantum*1e3:.1f} meV")
        for j, (e, i, r) in enumerate(zip(E, inside, ring)):
            print(f"    {j}: E = {e:.4f} eV, binding {(f['Ec_far']-e)*1e3:+7.1f} meV, "
                  f"{i*100:4.1f}% in dot, {r*100:5.1f}% in the tensile shell, "
                  f"{'bound' if e < f['Ec_far'] else 'UNBOUND (box state)'}")
    return dict(E=E, V=V, inside=np.array(inside), in_shell=np.array(ring),
                n_pts=n, L=f['L'], box_quantum=float(box_quantum),
                Ec_far=f['Ec_far'], binding=float((f['Ec_far'] - E[0]) * 1e3))


def hole_well(env):
    """Depth and location of the hole well, from the exact k = 0 valence edge.

    Holes are confined where the local valence edge is HIGHEST. Returns the edge inside the dot,
    the best value outside it, and the depth -- all in the electron convention, so a larger
    positive depth is a more strongly confined hole. `broken_gap` is the overlap with the matrix
    conduction edge; positive means the two bands cross in energy.
    """
    edges = valence_edge(env)
    m = env['mask']
    v_in = float(edges['v1'][m].max())
    v_out = float(edges['v1'][~m].max())
    return dict(v_in=v_in, v_out=v_out, far=env['Ev_far'],
                depth=v_in - env['Ev_far'], above_matrix=v_in - v_out,
                broken_gap=v_in - env['Ec_far'])


def six_band_holes(env, k=8, tol=1e-8, maxiter=20000, verbose=True):
    """Hole states from the SIX-band (valence-only) Hamiltonian. Seconds, and use this one.

    Prefer this to `eight_band_states(band='vb')` for holes here. Two reasons, one of them a
    correctness reason.

    **Spurious solutions.** The eight-band Hamiltonian couples conduction and valence, and on a
    finite-difference grid that coupling produces extra wrong-curvature branches at large k which
    appear as grid-scale-oscillatory states inside the gap. Measured on the InAs/GaAs control
    (spherical lens r = 10 nm, h = 1 nm): an eight-band solve seeded at the exact island valence
    top converged to a 7.4e-08 eV residual and returned twelve states in perfect Kramers pairs,
    14-52 meV below the edge, each with **0.6-0.8%** of its density in the island -- against a
    random-vector baseline of 8.2%. Those states sit between the matrix valence top (0.2351 eV)
    and the island valence top (0.3245 eV), i.e. classically forbidden in the matrix, yet 99%
    of their weight is there. No physical bound state does that. The same geometry solved
    six-band returns the ladder at 80-97% localization in seven seconds.

    So the eight-band failure was NOT sigma placement (the valence maximum is inside the island
    by 89-567 meV in every system here), not the block size, and not convergence. Suppressing
    spurious modes properly -- rescaling Ep so the extra branches leave the Brillouin zone -- is
    the fix if eight-band is needed; until then this is the trustworthy route.

    **Speed.** With no conduction band the hole ground state is the LARGEST eigenvalue, so it is
    extremal: plain Lanczos finds it with no folded spectrum and no sigma at all. That removes
    the entire sigma-scanning apparatus `ingasb_dot.hole_ladder` exists to work around, and with
    it the squared-gap conditioning penalty of the folded operator.

    The cost is the conduction-valence coupling itself, which is a real approximation in a
    narrow-gap material: it pushes valence states down, so this UNDERESTIMATES hole confinement,
    mildly in InGaSb (Eg = 0.42 eV) and more in InSb (Eg = 0.235 eV). Confinement energies from
    here are therefore conservative, and so is any critical size derived from them.

    Returns energies descending (most weakly confined first, i.e. the hole ground state at
    index 0) with the localization fraction that decides whether each is an island state.
    """
    import scipy.sparse.linalg as spla
    m, h = env['mask'], env['h']
    n = m.size
    ops = kpc.GridOperators(m.shape, h, periodic=False)
    fields = kp.material_fields(m, mt.kp_params(env['dot']), mt.kp_params(env['matrix']),
                                env['Ev'], env['Ec'], n_bands=6)
    H = kp.confined_hamiltonian(ops, fields, n_bands=6, strain=env['strain'])
    nb = H.shape[0] // n
    top = valence_edge_top(env)

    t0 = time.time()
    # 'LA' = largest algebraic. In the electron convention the valence states are at the top,
    # so the hole ground state is near the maximum -- no interior targeting needed. But the
    # maximum is NOT automatically a hole state; see below. Solve for a margin of extra states
    # so there is something left after the unphysical ones are discarded.
    n_solve = min(max(2 * k, k + 8), H.shape[0] - 2)
    E, V = spla.eigsh(H, k=n_solve, which='LA', tol=tol, maxiter=maxiter)
    order = np.argsort(-E)
    E, V = E[order], V[:, order]
    inside = np.array([float((np.abs(V[:, j].reshape(nb, n)) ** 2).sum(axis=0)
                             .reshape(m.shape)[m].sum()
                             / (np.abs(V[:, j]) ** 2).sum()) for j in range(V.shape[1])])

    # Discard eigenvalues ABOVE the island's own valence band edge.
    #
    # A hole bound in the island must lie below the top of the well that binds it, so E > `top`
    # is not a bound state, whatever its residual says. This is not hypothetical: at fine grids
    # the solve returns interface-pinned modes whose energy DIVERGES as ~1/h^2 (measured on a
    # 2.5 nm island: +0.117 eV at h = 0.50 nm rising to +1.566 eV at h = 0.156), and taking
    # `which='LA'` at face value returns those instead of the ladder.
    #
    # Localization does NOT separate them -- they sit on the dot boundary and come out 55-60%
    # "inside", indistinguishable from a real state by that test alone. The band edge does
    # separate them, because they are above it and no bound state can be.
    #
    # THIS IS A FILTER, NOT A FIX, and it is weaker than it looks. Those modes are not a
    # discretisation defect: the symmetrized six-band operator has no maximum at an abrupt
    # interface, because its kinetic tensor is Legendre-Hadamard elliptic but not strongly
    # elliptic (`kp_planewave.strong_ellipticity`). They are genuine eigenvalues of the operator
    # being solved, a plane-wave basis with no stencil at all produces them too, and removing the
    # ones above the edge does not un-contaminate what lies below -- they hybridize with the
    # ladder rather than sitting cleanly on top of it.
    #
    # `spurious` counts what was thrown away. A nonzero count means the runaway has reached the
    # band edge on this grid, so the level below it is not to be quoted.
    keep = E <= top + 1e-6
    n_spurious = int((~keep).sum())
    E, V, inside = E[keep][:k], V[:, keep][:, :k], inside[keep][:k]

    if verbose:
        print(f"  six-band vb: {H.shape[0]:,} unknowns, {time.time()-t0:.0f}s, "
              f"island valence top = {top:.4f} eV")
        if n_spurious:
            print(f"    discarded {n_spurious} eigenvalue(s) above the band edge "
                  f"(interface artifacts); check convergence in h")
        for j, (e, f) in enumerate(zip(E, inside)):
            print(f"    h{j}: E = {e:+.4f} eV, confinement {(top-e)*1e3:6.1f} meV, "
                  f"{f*100:5.1f}% inside the dot")
    return dict(E=E, V=V, inside=inside, loc=inside, top=top, n_bands=nb,
                n_spurious=n_spurious, seconds=time.time() - t0)


def valence_edge_top(env, reduce='max', erode=0, report=False):
    """The k = 0 top of the local valence band INSIDE the island -- `kp_pryor.hole_sigma`.

    Confinement energies are quoted downward from this, so it is the reference every hole number
    in this package depends on. `inside_mask` is not optional: without it this returns the value
    over the whole box, which in a strained system can sit in the matrix.

    **`reduce='max'` is h-DEPENDENT and biased high. Use `reduce='mean'` for an ellipsoid.**

    Eshelby's theorem makes the strain inside an ellipsoidal inclusion homogeneous, so this
    quantity is provably independent of h there -- and with `max` it is not. Measured on a 20 nm
    ellipsoid at AR 2 (InSb in InAs, 20 cells across the height), varying only how much of the
    boundary layer is excluded:

        erode    n_int     v1 std      max      mean    max - mean
            0    16840    16.20 m   0.7707    0.6498     120.96 m
            1    13936     4.62 m   0.6746    0.6470      27.60 m
            2    11296     1.99 m   0.6575    0.6467      10.88 m
            3     8912     1.13 m   0.6521    0.6466       5.54 m

    The **mean moves 3.2 meV** across that range while the **max moves 119 meV**. The problem is
    the estimator, not the boundary layer: a maximum tracks the upper tail of whatever scatter the
    staircased mask leaves, so refining the grid makes it drift rather than converge and a
    convergence study in h does not reveal it. The mean is the estimator of a constant, and is
    already within 3 meV on the raw mask.

    `reduce` is left at `'max'` by default because for a NON-ellipsoidal island (lens, dash) the
    interior is genuinely inhomogeneous and the maximum is the physically meaningful top of the
    well -- and because it is what `six_band_holes` uses as its filter threshold, where an
    over-tight bound would discard real states. Pass `'mean'` when the shape guarantees
    homogeneity and you want the h-independent number.

    `report` returns diagnostics instead of the bare value, including the interior standard
    deviation -- which by the same theorem must be ~0 for an ellipsoid, making it a free check on
    the strain solve, the padding and the mask.
    """
    if reduce not in ('max', 'mean'):
        raise ValueError(f"reduce must be 'max' or 'mean', got {reduce!r}")
    m = env['mask']
    sel = m
    if erode:
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(m, iterations=int(erode))
        if eroded.any():
            sel = eroded
    pick = lambda key: np.where(m, env['dot'][key], env['matrix'][key])
    edges = kp.local_band_edges(env['strain'], env['Ev'], np.zeros_like(env['Ev']),
                                pick('delta_so'), 0.0, 0.0, pick('b'), pick('d'),
                                hydrostatic_applied=True)
    v1 = edges['v1']
    top = float(v1[sel].max() if reduce == 'max' else v1[sel].mean())
    if not report:
        return top
    return dict(top=top, top_max=float(v1[sel].max()), top_mean=float(v1[sel].mean()),
                top_raw_max=float(v1[m].max()), interior_std=float(v1[sel].std()),
                n_interior=int(sel.sum()), n_mask=int(m.sum()), reduce=reduce)


def eight_band_states(env, band='vb', k=4, tol=1e-7, maxiter=8000, verbose=True):
    """OPT-IN, minutes not seconds: eight-band states, the same machinery as the Pryor benchmark.

    For HOLES prefer `six_band_holes`: this routine is subject to spurious in-gap solutions that
    are converged, Kramers-paired, and wrong. See that function's docstring for the measurement.

    Needed for the hole, which a single band cannot describe in a narrow-gap strained alloy.

    Read this before believing the output. In a **broken-gap** system the notion of "the states
    near the valence edge" is not clean: the dot's valence edge lies above the matrix's
    conduction edge, so at the same energy there are dot-like valence states and matrix-like
    conduction states, and they hybridize. A folded-spectrum solve returns the eigenvalues
    nearest sigma and its residual certifies only that they ARE eigenpairs, not that they are the
    ones wanted. So the localization fraction returned here is not decoration -- it is the only
    thing distinguishing a hole state from a matrix electron state at the same energy. Always
    re-run with sigma moved and check the spectrum is unchanged.
    """
    m, h = env['mask'], env['h']
    ops = kpc.GridOperators(m.shape, h, periodic=False)
    fields = kp.material_fields(m, mt.kp_params(env['dot']), mt.kp_params(env['matrix']),
                                env['Ev'], env['Ec'], n_bands=8)
    H = kp.confined_hamiltonian(ops, fields, n_bands=8, strain=env['strain'])

    if band == 'vb':
        sigma = kp.hole_sigma(env['Ev'], env['strain'],
                              np.where(m, env['dot']['b'], env['matrix']['b']),
                              np.where(m, env['dot']['d'], env['matrix']['d']),
                              np.where(m, env['dot']['delta_so'], env['matrix']['delta_so']),
                              inside_mask=m)
    else:
        sigma = float(env['Ec'][~m].min())

    if verbose:
        print(f"  eight-band {band}: {H.shape[0]:,} unknowns, sigma = {sigma:.4f} eV")
    E, V, info = eig.solve_interior(H, k=k, sigma=sigma, tol=tol, maxiter=maxiter,
                                    verbose=verbose)

    n = m.size
    inside = np.array([float((np.abs(V[:, j].reshape(8, n)) ** 2).sum(axis=0)
                             .reshape(m.shape)[m].sum()
                             / (np.abs(V[:, j]) ** 2).sum()) for j in range(V.shape[1])])
    if verbose:
        for j, (e, f) in enumerate(zip(E, inside)):
            print(f"    {j}: E = {e:.4f} eV, {f*100:5.1f}% inside the dot")
    return dict(E=E, V=V, inside=inside, sigma=sigma, info=info, band=band)


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------

def plot_bands(env, figsize=(13, 4.0)):
    """Band edges from the local strain along the three principal directions.

    Same construction as Pryor's Fig. 2 -- eigenvalues of the strain Hamiltonian at k = 0. The
    dashed and dotted lines are the unstrained matrix edges, i.e. the two thresholds a carrier
    has to get past to be bound; where a band crosses them is where a well exists.
    """
    import matplotlib.pyplot as plt
    p = band_profiles(env)
    name = env['matrix']['name']
    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)
    titles = {'x': '[100] through the centre', 'y': '[010] through the centre',
              'z': '[001] through the centre'}
    for ax, k in zip(axes, ('x', 'y', 'z')):
        d = p[k]
        ax.plot(d['r'], d['Ec'], 'k-', lw=1.5, label='$E_c$')
        ax.plot(d['r'], d['v1'], '-', color='#b03030', lw=1.5, label='$E_v$ (top)')
        ax.plot(d['r'], d['v2'], '-', color='#b03030', lw=0.8, alpha=0.6)
        ax.plot(d['r'], d['v3'], '-', color='#b03030', lw=0.8, alpha=0.6)
        ax.axhline(env['Ec_far'], color='0.55', ls='--', lw=0.8)
        ax.axhline(env['Ev_far'], color='0.55', ls=':', lw=0.8)
        for e in np.flatnonzero(np.diff(d['inside'].astype(int))):
            ax.axvline((d['r'][e] + d['r'][e + 1]) / 2, color='0.85', lw=0.8, zorder=0)
        ax.set_xlabel(f"{k} (nm)")
        ax.set_title(titles[k], fontsize=9)
    axes[0].set_ylabel('E (eV)')
    axes[0].legend(fontsize=8, loc='center left')
    axes[-1].text(0.98, 0.06, f"dashed: unstrained {name} $E_c$\ndotted: unstrained {name} $E_v$",
                  transform=axes[-1].transAxes, ha='right', fontsize=7, color='0.4')
    fig.suptitle(f"{env['shape']['label']}   |   {env['dot']['name']} in {name}   |   "
                 f"local band edges at $k=0$", fontsize=10)
    fig.tight_layout()
    return fig


def _allowed_segments(r, profile, E, side):
    """Contiguous stretches of `r` where a carrier at energy `E` is classically allowed.

    `side` is 'below' for electrons (allowed where E_c < E) and 'above' for holes (allowed where
    the valence edge lies above E, since a hole's energy runs the other way).
    """
    ok = (profile < E) if side == 'below' else (profile > E)
    if not ok.any():
        return []
    idx = np.flatnonzero(ok)
    breaks = np.flatnonzero(np.diff(idx) > 1)
    groups = np.split(idx, breaks + 1)
    return [(r[g[0]], r[g[-1]]) for g in groups if len(g) > 1]


def plot_levels(env, electron=None, hole=None, cut='x', ax=None, figsize=(8.0, 5.2),
                max_levels=6, palette=None, legend_levels=None):
    """Band edges along one principal cut, with computed level energies drawn where each state
    is classically allowed.

    `electron` is the dict from `electron_states`; `hole` is a dict with an `E` array from
    `eight_band_states` or `ingasb_dot.hole_ladder`. Either may be omitted.

    Each level is a horizontal segment spanning the classically allowed region at that energy --
    where E_c dips below the level for an electron, where the valence edge rises above it for a
    hole. That is the honest way to draw "where the state is" from a 1D cut: it is defined by the
    potential alone, so it cannot silently disagree with the wavefunction the way a hand-placed
    line can. A level that is not bound (electron at or above the far-field matrix edge, hole at
    or below it) is drawn dashed and spans the whole box, because that is exactly what it does --
    it is a box state, not a confined one.

    By default level lines take the colour of the band they belong to rather than a new hue per
    level: the identity that matters is electron-vs-hole, and the index is a direct label. That
    default is right when the question is "is this state in the dot"; it is wrong when the
    question is "which level is which", because near-degenerate levels overlap and one colour
    makes the ladder read as a smear. Pass `palette` for the latter -- a colormap name or an
    explicit list of colours -- and each level gets its own hue plus a legend entry carrying its
    energy. `legend_levels` caps how many appear in the legend (default: all drawn ones).
    """
    import matplotlib.pyplot as plt
    p = band_profiles(env)[cut]
    r, Ec, Ev = p['r'], p['Ec'], p['v1']

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(r, Ec, '-', color='k', lw=1.6, label='$E_c$', zorder=4)
    ax.plot(r, Ev, '-', color='#b03030', lw=1.6, label='$E_v$ (top)', zorder=4)
    ax.axhline(env['Ec_far'], color='0.55', ls='--', lw=0.9, zorder=1,
               label='$E_c$ matrix (far)')
    ax.axhline(env['Ev_far'], color='0.55', ls=':', lw=0.9, zorder=1,
               label='$E_v$ matrix (far)')

    # Shade the island so "inside" is visible without a second axis.
    inside = p['inside']
    for e in np.flatnonzero(np.diff(inside.astype(int))):
        ax.axvline((r[e] + r[e + 1]) / 2, color='0.85', lw=0.9, zorder=0)

    # Near-degenerate levels are the norm here -- a p-like doublet sits within a few meV -- so
    # labels are staggered outward when they would land on each other. Without this the doublet
    # prints one label on top of the other and reads as a single level.
    span = float(np.ptp(Ec.tolist() + Ev.tolist()))
    placed = []

    def label(x, y, text, colour, alpha=1.0):
        step = 0
        while any(abs(y - py) < 0.022 * span and abs(step - ps) < 1 for py, ps in placed):
            step += 1
        placed.append((y, step))
        ax.text(x + step * 0.05 * float(r[-1] - r[0]), y, text, va='center', fontsize=7,
                color=colour, alpha=alpha)

    def hues(n, band_colour):
        """One colour per level from `palette`, or the band colour repeated (the default)."""
        if palette is None:
            return [band_colour] * n
        if isinstance(palette, str):
            cm = plt.get_cmap(palette)
            return [cm(0.12 + 0.76 * (j / max(n - 1, 1))) for j in range(n)]
        return [palette[j % len(palette)] for j in range(n)]

    for states, colour, side, far, tag in (
            (electron, 'k', 'below', env['Ec_far'], 'e'),
            (hole, '#b03030', 'above', env['Ev_far'], 'h')):
        if states is None:
            continue
        E = np.atleast_1d(np.asarray(states['E'] if isinstance(states, dict) else states)).real
        E = E[:max_levels]
        cols = hues(len(E), colour)
        n_leg = len(E) if legend_levels is None else legend_levels
        for j, e in enumerate(E):
            col = cols[j]
            bound = (e < far) if side == 'below' else (e > far)
            segs = _allowed_segments(r, Ec if side == 'below' else Ev, e, side)
            # The legend entry carries the energy, so the reader never has to match a hue
            # against an axis by eye -- that is the whole point of colouring per level.
            leg = (f"{tag}{j} = {e:.4f} eV" + ("" if bound else "  (unbound)")
                   if j < n_leg else None)
            if not segs or not bound:
                ax.plot([r[0], r[-1]], [e, e], ls=':', color=col, lw=1.2, alpha=0.75,
                        zorder=3, label=leg)
                label(r[-1], e, f" {tag}{j} unbound", col, alpha=0.8)
                continue
            for i, (a, b) in enumerate(segs):
                ax.plot([a, b], [e, e], '-', color=col, lw=2.2, alpha=0.95, zorder=5,
                        label=leg if i == 0 else None)
            label(segs[-1][1], e, f" {tag}{j}", col)

    ax.set_xlabel(f"{cut} (nm)")
    ax.set_ylabel('E (eV)')
    ax.set_title(f"{env['shape']['label']}\n{env['dot']['name']} in {env['matrix']['name']} — "
                 f"band edges and confined levels along [{'100' if cut=='x' else '001'}]",
                 fontsize=9)
    # A per-level legend runs long, so it moves outside the axes rather than covering the bands.
    if palette is None:
        ax.legend(fontsize=7.5, loc='center left', frameon=False)
    else:
        ax.legend(fontsize=7.0, loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  frameon=False, borderaxespad=0.0)
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.figure.tight_layout()
    return ax.figure


def plot_maps(env, figsize=(13, 3.6)):
    """Maps of Tr(eps), the conduction edge and the valence edge in the plane y = 0."""
    import matplotlib.pyplot as plt
    edges = valence_edge(env)
    iy = len(env['cy']) // 2
    ext = [env['cz'][0], env['cz'][-1], env['cx'][0], env['cx'][-1]]
    panels = [(env['strain'].trace[:, iy, :], r'Tr $\varepsilon$', 'RdBu_r', None),
              (edges['cb'][:, iy, :], '$E_c$ (eV)', 'viridis', None),
              (edges['v1'][:, iy, :], '$E_v$, top (eV)', 'magma', None)]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (data, title, cmap, _) in zip(axes, panels):
        vmax = np.abs(data).max() if cmap == 'RdBu_r' else None
        im = ax.imshow(data, origin='lower', extent=ext, aspect='auto', cmap=cmap,
                       vmin=-vmax if vmax else None, vmax=vmax)
        ax.contour(env['cz'], env['cx'], env['mask'][:, iy, :].astype(float), [0.5],
                   colors='w', linewidths=1.0)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel('z (nm)')
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0].set_ylabel('x (nm)')
    fig.suptitle(f"{env['shape']['label']}   |   {env['dot']['name']} in "
                 f"{env['matrix']['name']}   |   slice at y = 0", fontsize=10)
    fig.tight_layout()
    return fig
