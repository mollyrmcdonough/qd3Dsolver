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


def spherical_lens(radius, height):
    """Lens cut from a sphere, the dot geometry of Pryor & Pistol Fig. 1(a).

    Use this, not `lens`, when comparing against their tables: `lens` is a half-ellipsoid, which
    stands vertically at its rim and encloses 23% more volume at h/d = 1/4. See
    `qdsolver_core.spherical_cap_mask`.
    """
    return dict(
        kind='spherical_lens',
        label=f"spherical lens r = {radius:g} nm, h = {height:g} nm (h/d = {height/(2*radius):.3g})",
        extent=(2 * radius, 2 * radius, height),
        volume=qd.spherical_cap_volume(radius, height),
        mask=lambda X, Y, Z: qd.spherical_cap_mask(X, Y, Z, radius, height),
        params=dict(radius=radius, height=height),
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
    if isinstance(spec, dict):
        return spec
    if isinstance(spec, str):
        return mt.material(spec)
    if isinstance(spec, (tuple, list)):
        return mt.material(*spec)
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

def electron_states(env, k=4, verbose=True):
    """Single-band conduction states. Bound iff E < `env['Ec_far']` = Eg(matrix).

    The potential is the strained conduction edge including the piezoelectric term; the mass is
    the Kane-derived band-edge mass of each material (`materials.electron_mass`). Energies are on
    the same zero as everything else, the unstrained matrix valence edge.

    `inside` is the fraction of each state's density within the island. In a type-I system that
    should be large; in a broken-gap one it should be SMALL, because the electron is expelled
    from the dot and lives in the tensile shell around it. Either way it is the check that the
    state is the one you meant to find.
    """
    m = env['mask']
    m_e = np.where(m, mt.electron_mass(env['dot']), mt.electron_mass(env['matrix']))
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


def eight_band_states(env, band='vb', k=4, tol=1e-7, maxiter=8000, verbose=True):
    """OPT-IN, minutes not seconds: eight-band states, the same machinery as the Pryor benchmark.

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
