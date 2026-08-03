"""InAs / In(x)Ga(1-x)Sb / InAs quantum dots and dashes: strain, band landscape, electron states.

Driver for the two notebooks `ingasb_lens.ipynb` and `ingasb_dash.ipynb`, which differ only in
the shape passed to `build`. Material parameters and their provenance live in `materials_sb.py`;
read that module's docstring before trusting any number out of here, because two entries in the
source database are demonstrably wrong and one of them (GaSb's a_v sign) sets the hole well depth.

What makes this system different from the Pryor InAs/GaAs benchmark
-------------------------------------------------------------------
The alignment is **broken gap**. The In(x)Ga(1-x)Sb valence edge sits above the InAs conduction
edge at every composition -- by ~20 meV at x = 0 and ~180 meV at x = 1 -- so this is not a type-I
dot with an electron and a hole in the same box. Holes are confined in the dot; electrons are
*expelled* from it, because the dot's conduction edge is far above the matrix's.

The only thing that could bind the electron is strain, not a band offset. The dot is compressed by
the matrix (the misfit runs -0.6% to -6.5% with composition), and by reaction the InAs immediately
around it is put into tension. With a_c negative, tension pulls the InAs conduction edge DOWN, so
the dot digs a conduction-band pocket in the matrix wrapped around itself -- a genuinely
inhomogeneous-strain effect that a model with uniform strain inside the dot and none outside would
not have at all, which is why the FD elasticity solver is used here.

**That pocket does not bind an electron, anywhere in this parameter range.** Measured rather than
assumed: over compositions x = 0.35 to 1.0 the deepest point of the pocket goes from 83 to 190 meV
and the island volume was varied by 4x on top of that, and the computed ground state moved by
3.6 meV -- in the wrong direction, because it is box zero-point energy, not binding. The reason is
geometric and `pocket_metrics` quantifies it: binding any state in a spherical well of depth V0
and radius R needs V0*R^2 > pi^2*hbar^2/(8m) = 3.78 eV nm^2 at the InAs electron mass, and the
pocket reaches at most 2.45 -- and that estimate is optimistic, since the pocket is a thin shell
wrapped around a large repulsive barrier and a shell binds worse than a compact sphere of the same
volume.

So the electron here is bound by Coulomb attraction to the hole in the island, not by a
single-particle well. That is the standard picture for a type-II dot, and it means any "electron
binding energy" from a single-particle solve in a finite box is measuring the box. `electron_states`
is still provided, because seeing the states sit above the barrier edge and drift with box size is
the evidence for the statement, but do not read a binding energy off it.

Cost, and why the default is single band
----------------------------------------
`build` + `electron_states` is seconds per configuration, so composition, size and aspect ratio
are explorable. That is the point of the default path. The single-band conduction Hamiltonian is
defensible for the electron here -- the electron sits in InAs, away from the dot, in a shallow
pocket, and its mass is taken from the same Kane parameters as the eight-band model.

It is NOT defensible for the hole, which sits in a narrow-gap heavily strained alloy where the
valence bands are strongly mixed. So no single-band hole solve is offered. What is offered
cheaply is `valence_edge`, the exact k = 0 top of the local valence band including the full
Bir-Pikus shear terms, which gives the hole well depth and its shape without solving anything.
For actual hole states use `eight_band_states`, which is the same machinery as the Pryor
benchmark and costs minutes rather than seconds.
"""
import json
import os
import time

import numpy as np

import qdsolver_core as qd
import elasticity_fd as ef
import piezoelectric as pz
import strain_fourier as sf
import kp_confined as kpc
import kp_pryor as kp
import eigensolvers as eig
import materials_sb as ms


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
    # V(w) = aspect*w^2*height - k*(aspect+1)*w*height^2/2 + k^2*height^3/3, k = 2/tan
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

def build(shape, x, h=0.5, pad=6.0, z_pad=None, use_piezo=True, matrix='InAs',
          vol_tol=0.15, verbose=True, require_converged=True):
    """Grid, mask, strain, piezoelectric potential and band-edge fields for one configuration.

    `x` is the InSb fraction of the In(x)Ga(1-x)Sb dot. `shape` comes from `lens` or `dash`.

    The strain is solved with `elasticity_fd`, i.e. Pryor's real-space method with
    position-dependent elastic constants, not the Fourier solver. That is not a stylistic choice
    here: the electron pocket this system relies on lives in the *matrix*, where a homogeneous
    solver gets the stiffness contrast wrong, and In(x)Ga(1-x)Sb differs from InAs in bulk
    modulus by up to ~20% across the composition range.

    `vol_tol` guards the mask: a faceted island on a cubic grid is staircased, and the volume
    error is first order in h. Exceeding the tolerance raises rather than warns, because a mask
    that is 15% oversized makes every energy below wrong by about as much.
    """
    dot = ms.ingasb(x)
    mat = ms.SB_MATERIALS[matrix]

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

    eps_star = ms.misfit(dot, mat)
    t0 = time.time()
    # require_converged is the default and is left on deliberately: an unconverged or
    # preconditioner-filtered strain field produces band edges that look plausible and are wrong,
    # and it used to be reported only under verbose -- which is how a whole box-convergence study
    # got run on two silently broken large grids.
    strain = ef.solve_strain_fd(mask, eps_star, ms.elastic(dot), ms.elastic(mat), h, tol=1e-10,
                                require_converged=require_converged)
    t_strain = time.time() - t0

    phi = np.zeros(mask.shape)
    if use_piezo:
        e14 = np.where(mask, dot['e14'], mat['e14'])
        phi = pz.potential(strain, e14, mat['eps_R'], h)

    Ec, Ev = ms.band_edge_fields(mask, strain.trace, dot, mat)
    U = -phi                       # electrostatic potential ENERGY of an electron
    Ec, Ev = Ec + U, Ev + U

    env = dict(shape=shape, x=x, dot=dot, matrix=mat, h=h, pad=pad,
               cx=cx, cy=cy, cz=cz, X=X, Y=Y, Z=Z, mask=mask,
               strain=strain, phi=phi, Ec=Ec, Ev=Ev, eps_star=eps_star,
               vol_err=vol_err,
               Ec_far=float(mat['Eg']), Ev_far=0.0)   # unstrained matrix edges, the energy zero

    if verbose:
        g = strain.at(mask)
        tr = g['exx'] + g['eyy'] + g['ezz']
        print(f"{shape['label']}   dot = {dot['name']}")
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
        print(f"  E_c: dot {Ec[mask].mean():.3f}, matrix min {Ec[~mask].min():.3f}, "
              f"far {env['Ec_far']:.3f} eV   "
              f"-> electron pocket depth {env['Ec_far'] - Ec[~mask].min():.3f} eV")
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


def plot_bands(env, figsize=(13, 4.0)):
    """Band edges from the local strain along the three principal directions.

    Same construction as Pryor's Fig. 2 -- eigenvalues of the strain Hamiltonian at k = 0 -- but
    for this system the interesting feature is the opposite one: the conduction edge dipping
    BELOW the far-field InAs value in the matrix around the dot, which is the electron pocket.
    """
    import matplotlib.pyplot as plt
    p = band_profiles(env)
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
    axes[-1].text(0.98, 0.06, 'dashed: unstrained InAs $E_c$\ndotted: unstrained InAs $E_v$',
                  transform=axes[-1].transAxes, ha='right', fontsize=7, color='0.4')
    fig.suptitle(f"{env['shape']['label']}   |   dot = {env['dot']['name']}   |   "
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
    fig.suptitle(f"{env['shape']['label']}   |   dot = {env['dot']['name']}   |   "
                 f"slice at y = 0", fontsize=10)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# States
# --------------------------------------------------------------------------------------

def pocket_metrics(env, depths=(0.0, 0.01, 0.025, 0.05, 0.10), verbose=True):
    """Size and depth of the strain-induced conduction pocket, and whether it can bind at all.

    The obvious summary of the pocket, `min(E_c)` over the matrix, is misleading and was
    misleading here: it reports the extreme value at the island's rim where the tensile strain
    peaks, which is a sharp local dip rather than a well. Doubling that number by raising the In
    content changes the electron energy by under 2 meV, because a light electron cannot localize
    in a thin shell no matter how deep it is.

    What matters instead is how much matrix volume lies below the far-field conduction edge, and
    at what depth. This tabulates that, and compares each (depth, volume) pair against the
    textbook criterion for a finite spherical well of radius R and depth V0 to hold any bound
    state at all,

        V0 * R^2  >  pi^2 * hbar^2 / (8 m)

    with R taken as the radius of a sphere of the same volume. That is a necessary-condition
    style estimate, not a solve -- a shell is worse at binding than a compact sphere of the same
    volume, so failing it is decisive while passing it is only suggestive.
    """
    Ec, m, h = env['Ec'], env['mask'], env['h']
    far = env['Ec_far']
    outside = ~m
    m_e = ms.electron_mass(env['matrix'])
    crit = np.pi ** 2 * qd.HBAR2_OVER_2M0 / (4.0 * m_e)      # eV nm^2, = pi^2 hbar^2 / (8 m)

    rows = []
    for t in depths:
        sel = outside & (Ec < far - t)
        vol = float(sel.sum()) * h ** 3
        if vol <= 0:
            rows.append(dict(depth=t, volume=0.0, R=0.0, VR2=0.0, binds=False))
            continue
        R = (3.0 * vol / (4.0 * np.pi)) ** (1.0 / 3.0)
        # Mean depth below the far-field edge within the selected region.
        mean_depth = float((far - Ec[sel]).mean())
        rows.append(dict(depth=t, volume=vol, R=R, mean_depth=mean_depth,
                         VR2=mean_depth * R ** 2, binds=mean_depth * R ** 2 > crit))

    if verbose:
        print(f"  pocket: deepest point {(far - Ec[outside].min())*1e3:.1f} meV below the "
              f"far-field InAs edge")
        print(f"  binding a state in a spherical well needs V0*R^2 > {crit:.2f} eV nm^2 "
              f"(m_e = {m_e:.4f})")
        print(f"    {'below':>8} {'volume':>10} {'equiv R':>9} {'mean depth':>11} "
              f"{'V0*R^2':>9}  verdict")
        for r in rows:
            if r['volume'] <= 0:
                print(f"    {r['depth']*1e3:>7.0f}m {'--':>10}")
                continue
            print(f"    {r['depth']*1e3:>7.0f}m {r['volume']:>10.1f} {r['R']:>9.2f} "
                  f"{r['mean_depth']*1e3:>10.1f}m {r['VR2']:>9.2f}  "
                  f"{'could bind' if r['binds'] else 'too weak'}")
    return rows


def electron_states(env, k=4, verbose=True):
    """Single-band conduction states in the strain-induced pocket.

    The potential is the strained conduction edge including the piezoelectric term; the mass is
    the Kane-derived band-edge mass of each material (see `materials_sb.electron_mass`). Energies
    are on the same zero as everything else, the unstrained InAs valence edge, so a state is
    bound iff its energy is below `env['Ec_far']` = Eg(InAs).

    Returns E, V and the fraction of each state's density inside the dot -- which for this
    system should be SMALL, since the electron is expelled from the dot and lives in the tensile
    InAs shell around it. A large fraction means something is wrong with the alignment.
    """
    m = env['mask']
    m_e = np.where(m, ms.electron_mass(env['dot']), ms.electron_mass(env['matrix']))
    E, V = qd.solve_states(m_e, env['Ec'], env['h'], n_states=k)

    inside = []
    for j in range(V.shape[1]):
        rho = np.abs(V[:, j]) ** 2
        inside.append(float(rho.reshape(m.shape)[m].sum() / rho.sum()))
    inside = np.array(inside)

    if verbose:
        print(f"  electron states (bound below E_c(InAs, far) = {env['Ec_far']:.3f} eV):")
        for j, (e, f) in enumerate(zip(E, inside)):
            tag = 'bound' if e < env['Ec_far'] else 'unbound (box state)'
            print(f"    {j}: E = {e:.4f} eV, binding {(env['Ec_far']-e)*1e3:+7.1f} meV, "
                  f"{f*100:5.1f}% inside the dot   {tag}")
    return dict(E=E, V=V, inside=inside, n_bound=int((E < env['Ec_far']).sum()))


def hole_well(env):
    """Depth and location of the hole well, from the exact k = 0 valence edge.

    Holes are confined where the local valence edge is HIGHEST. Returns the edge inside the dot,
    the far-field matrix value, and the depth -- all in the electron convention, so a larger
    positive depth is a more strongly confined hole.
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
    fields = kp.material_fields(m, ms.kp_params(env['dot']), ms.kp_params(env['matrix']),
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
# Sweeps
# --------------------------------------------------------------------------------------

def sweep(shape_of, values, label, h=0.5, pad=6.0, use_piezo=True, verbose=False):
    """Run the cheap path over a parameter. `shape_of(v)` returns (shape, x) for each value.

    Returns a list of dicts with the electron ground state, its binding and localization, the
    hole well depth, and the mean strain -- enough to plot any of them against the swept
    parameter.
    """
    rows = []
    for v in values:
        # shape_of is inside the guard too: a geometrically impossible island (too narrow to
        # hold the requested volume at this facet angle) should be reported and skipped, not
        # abort the whole sweep.
        try:
            shape, x = shape_of(v)
            env = build(shape, x, h=h, pad=pad, use_piezo=use_piezo, verbose=verbose)
        except ValueError as exc:
            print(f"  {label} = {v}: skipped -- {exc}")
            continue
        pm = pocket_metrics(env, verbose=False)
        hw = hole_well(env)
        g = env['strain'].at(env['mask'])
        best = max(r['VR2'] for r in pm)
        row = dict(x=x,
                   pocket_max=env['Ec_far'] - float(env['Ec'][~env['mask']].min()),
                   pocket_VR2=best, binds=best > POCKET_CRITERION,
                   hole_depth=hw['depth'], broken_gap=hw['broken_gap'],
                   tr=float(g['exx'] + g['eyy'] + g['ezz']),
                   volume=shape['volume'], vol_err=env['vol_err'])
        # Set the swept key last, and by assignment rather than as a keyword: when the sweep IS
        # over composition, `label` is 'x' and passing both as keywords is a TypeError.
        row[label] = v
        rows.append(row)
        r = row
        print(f"  {label} = {v!r:>10}: pocket {r['pocket_max']*1e3:6.1f} meV deep, "
              f"V0*R^2 = {best:5.2f} of the {POCKET_CRITERION:.2f} eV nm^2 needed "
              f"({'BINDS' if r['binds'] else 'no bound electron'}), "
              f"hole well {hw['depth']*1e3:6.1f} meV, Tr eps {r['tr']:+.4f}")
        del env
    return rows


#: V0*R^2 (eV nm^2) needed for a spherical well to hold any bound state, at the InAs electron
#: mass: pi^2 * hbar^2 / (8 m). Compare `pocket_metrics` output against this.
POCKET_CRITERION = np.pi ** 2 * qd.HBAR2_OVER_2M0 / (4.0 * ms.electron_mass(
    ms.SB_MATERIALS['InAs']))


def plot_sweep(rows, label, figsize=(11, 3.6)):
    """Pocket strength against the binding threshold, hole well depth, and mean strain.

    The first panel deliberately plots V0*R^2 rather than an electron binding energy. In this
    system the single-particle pocket never reaches the threshold, so a "binding energy" from a
    finite-box solve would be box zero-point energy dressed up as physics -- it barely responds
    to a 2.4x change in pocket depth. V0*R^2 against the threshold says the same thing honestly
    and shows how far short the pocket falls.
    """
    import matplotlib.pyplot as plt
    v = [r[label] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    axes[0].plot(v, [r['pocket_VR2'] for r in rows], 'o-', color='k', ms=4, label='$V_0R^2$')
    axes[0].axhline(POCKET_CRITERION, color='#b03030', lw=1.0, ls='--',
                    label='binding threshold')
    axes[0].set_ylim(0, max(POCKET_CRITERION * 1.15,
                            max(r['pocket_VR2'] for r in rows) * 1.15))
    axes[0].set_ylabel(r'$V_0R^2$  (eV nm$^2$)')
    axes[0].set_title('can the strain pocket bind an electron?', fontsize=9)
    axes[0].legend(fontsize=7)

    ax2 = axes[1]
    ax2.plot(v, [r['hole_depth'] * 1e3 for r in rows], 'o-', color='#b03030', ms=4)
    ax2.set_ylabel('hole well depth (meV)')
    ax2.set_title('$E_v$(dot) above unstrained InAs $E_v$', fontsize=9)

    axes[2].plot(v, [r['tr'] for r in rows], 'o-', color='#30609b', ms=4)
    axes[2].set_ylabel(r'$\langle$Tr $\varepsilon\rangle$ in the dot')
    axes[2].set_title('mean hydrostatic strain', fontsize=9)
    for ax in axes:
        ax.set_xlabel(label)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------------------
# Size sweep with eight-band hole levels -- the analogue of Pryor's Fig. 4
# --------------------------------------------------------------------------------------
#
# Pryor's Fig. 4 plots bound-state energies against island size for both bands. Only the
# valence panel has a counterpart here: this system is broken gap, the electron is not bound by
# any single-particle well (see `pocket_metrics` and the module docstring), so its "levels" would
# be box quantization. What is swept for the conduction band instead is the box-independent
# binding criterion, which is a statement about the pocket rather than about the box.

def kramers_levels(E, loc, split_tol=1e-4):
    """Collapse the exactly-twofold spectrum into physical levels.

    Time reversal makes every eigenvalue of this Hamiltonian doubly degenerate, so one "state"
    is a *pair* of eigenvalues. This checks the pairing rather than trusting it: it returns
    (level energies, mean localization per level, largest within-pair splitting in eV), and a
    within-pair splitting above the solver residual means the degeneracy is unresolved and the
    level count is not to be believed. Same construction as `pryor_fig4.kramers_levels`.
    """
    E = np.asarray(E)
    n = len(E) // 2 * 2
    if n == 0:
        return np.array([]), np.array([]), 0.0
    pairs = E[:n].reshape(-1, 2)
    lloc = np.asarray(loc)[:n].reshape(-1, 2).mean(axis=1)
    return pairs.mean(axis=1), lloc, float(np.abs(np.diff(pairs, axis=1)).max())


#: Fraction of sum_i |psi_i|^2 that must lie inside the island for a state to count as a hole
#: state rather than a matrix box state. See `hole_ladder` for why this is load-bearing here.
HOLE_LOC_MIN = 0.35


def valence_edge_top(env):
    """The k = 0 top of the local valence band inside the island -- `kp_pryor.hole_sigma`.

    In the Pryor benchmark this is the right sigma for a hole solve. Here it is NOT, and
    `hole_ladder` explains why at length. Kept as a named quantity because it is still the
    correct *band edge*, and the gap between it and the actual ladder top is the confinement
    energy, which is worth reporting.
    """
    m = env['mask']
    pick = lambda key: np.where(m, env['dot'][key], env['matrix'][key])
    return float(kp.hole_sigma(env['Ev'], env['strain'], pick('b'), pick('d'),
                               pick('delta_so'), inside_mask=m))


def _degenerate_groups(E, tol=1e-5):
    """Indices of `E` grouped into runs of (near-)equal eigenvalues, sorted descending.

    Time reversal makes every eigenvalue of the eight-band Hamiltonian exactly twofold
    degenerate, so a group of two is one physical level and its Kramers partner. Groups of one
    happen at the edges of a solver's block, where a partner fell outside the k states returned;
    they are kept as levels but contribute no degeneracy check.
    """
    order = np.argsort(-np.asarray(E))
    groups, cur = [], [order[0]] if len(order) else []
    for j in order[1:]:
        if abs(E[j] - E[cur[-1]]) <= tol:
            cur.append(j)
        else:
            groups.append(np.array(cur)); cur = [j]
    if len(cur):
        groups.append(np.array(cur))
    return groups


def hole_ladder(env, k=8, probe=24, loc_min=HOLE_LOC_MIN, sigmas=None, n_sigma=6,
                tol=1e-7, maxiter=3000, degeneracy_tol=1e-5,
                locate=True, locate_probe=8, locate_maxiter=250, verbose=True):
    """Eight-band hole levels for one environment, by scanning sigma and keeping what localizes.

    Sigma placement is the entire problem in this system, and the obvious choice is wrong
    ---------------------------------------------------------------------------------------
    In the Pryor InAs/GaAs benchmark, `kp_pryor.hole_sigma` -- the exact k = 0 top of the local
    valence band -- is the right target: the hole states sit a few tens of meV below it and a
    solve for the k eigenvalues nearest it returns them. `pryor_fig4.bound_states` does exactly
    that.

    Here that fails, and it fails silently. Measured on a lens at x = 1 (dense diagonalization
    of the full 8-band matrix, so this is not an artifact of an iterative solver):

      * `hole_sigma` came out at 0.806 eV, and the spectrum there is a **dense matrix continuum**
        -- levels ~1 meV apart, each with ~1% of its density in the island. Those are InAs
        electron box states, which exist at that energy because the alignment is broken gap.
      * The island-localized states (40-84% inside) sit between -0.04 and +0.44 eV, i.e.
        **370 to 850 meV BELOW** `hole_sigma`.
      * A folded-spectrum solve seeded at `hole_sigma` ran to its 8000-iteration limit, returned
        a residual of 6e-4 eV, and reported four states 0.6% localized. Converged-looking,
        certified as eigenpairs, and not hole states.
      * The same solver seeded at 0.44 eV converged in 652 iterations and 10 seconds to a
        residual of 1.5e-7 eV, and returned exactly the localized states the dense
        diagonalization found.

    So the solver was never the problem and the conditioning was never the problem: sigma was.
    Two things put the ladder that far below the band edge -- the confinement energy is large
    (small island, deep well, heavy-hole mass), and in a broken-gap system the island's valence
    states are degenerate with the matrix conduction continuum and hybridize with it rather than
    sitting cleanly in a gap.

    What this does instead
    ----------------------
    Scans sigma down from the k = 0 valence-band top toward the matrix valence edge, solving for
    `probe` states at each, collapsing each solve's spectrum to physical levels, keeping the
    levels at least `loc_min` localized in the island, then merging across sigmas and taking the
    top `k`. Every returned level therefore has to have been found by an actual solve near its
    own energy, and `sigma_hits` records which sigma found what -- if the top level came from
    the lowest sigma in the scan, the scan did not bracket the ladder and should be extended.

    `k` counts LEVELS, not eigenvalues: each level already has its Kramers partner folded in.

    Two stages, because most of the scan is otherwise wasted
    -------------------------------------------------------
    A sigma in the matrix continuum cannot converge -- that is the whole reason the scan exists --
    so it runs to `maxiter` and everything it returns is then rejected. Measured on the r = 6 nm
    lens: the two continuum sigmas burned 6002 of 8646 total iterations, 69% of the wall clock,
    for zero kept levels.

    So `locate=True` runs a cheap first pass at every sigma (`locate_probe` states,
    `locate_maxiter` iterations) purely to ask "is there anything island-like near here?", and
    only sigmas that answer yes get the full `probe`/`maxiter` solve. Localization is a much
    coarser property than an eigenvalue and converges long before one does, which is what makes
    the cheap pass trustworthy for this question and not for any other -- no energy from a locate
    pass is ever reported.

    The promotion threshold is deliberately `loc_min / 2`, not `loc_min`: a half-converged vector
    can understate its own localization, and promoting a borderline sigma costs one extra solve
    whereas missing one loses a level silently. Set `locate=False` to force the full solve
    everywhere.

    Coverage is still computed over EVERY sigma, using the locate-pass window for the ones that
    were skipped. Skipping a sigma must not make the scan look better covered than it is.

    Holes are read *downward* from the valence edge, so the most strongly confined hole is the
    HIGHEST energy in the electron convention, and everything is sorted descending.
    """
    m, h = env['mask'], env['h']
    top = valence_edge_top(env)
    if sigmas is None:
        # Skip the band edge itself: that is where the continuum is, and it is what fails.
        sigmas = np.linspace(top, env['Ev_far'], n_sigma + 1)[1:]
    sigmas = np.asarray(sigmas, dtype=float)

    ops = kpc.GridOperators(m.shape, h, periodic=False)
    fields = kp.material_fields(m, ms.kp_params(env['dot']), ms.kp_params(env['matrix']),
                                env['Ev'], env['Ec'], n_bands=8)
    H = kp.confined_hamiltonian(ops, fields, n_bands=8, strain=env['strain'])
    n = m.size
    if verbose:
        print(f"  eight-band vb: {H.shape[0]:,} unknowns; k=0 valence top = {top:.4f} eV")
        print(f"  scanning sigma over " +
              ", ".join(f"{s:.3f}" for s in sigmas) + " eV")

    def localization(V):
        return np.array([float((np.abs(V[:, j].reshape(8, n)) ** 2).sum(axis=0)
                               .reshape(m.shape)[m].sum() / (np.abs(V[:, j]) ** 2).sum())
                         for j in range(V.shape[1])])

    lv_E, lv_loc, lv_sig, lv_res, lv_deg = [], [], [], [], []
    spans = []
    n_probed, n_localized, n_iters, n_skipped = 0, 0, 0, 0
    t0 = time.time()
    for sg in sigmas:
        # ---- stage 1: cheap locate. Only ever asked "is anything island-like near here?" ----
        if locate:
            Eq, Vq, infoq = eig.solve_interior(H, k=locate_probe, sigma=float(sg), tol=tol,
                                               maxiter=locate_maxiter, verbose=False)
            n_probed += len(Eq)
            n_iters += infoq['iterations']
            locq = localization(Vq)
            promote = bool((locq >= loc_min / 2.0).any())
            if not promote:
                # Record what this pass actually looked at, so coverage is not overstated.
                spans.append((float(sg), float(Eq.min()), float(Eq.max())))
                n_skipped += 1
                if verbose:
                    print(f"    sigma {sg:7.4f}: locate {infoq['iterations']:>4} iters, "
                          f"max localization {locq.max()*100:4.1f}% -- skipped")
                del Vq
                continue
            del Vq

        # ---- stage 2: the real solve, only where stage 1 saw something ----
        E, V, info = eig.solve_interior(H, k=probe, sigma=float(sg), tol=tol, maxiter=maxiter,
                                        verbose=False)
        spans.append((float(sg), float(E.min()), float(E.max())))
        loc = localization(V)
        res = np.asarray(info['residuals'])
        n_probed += len(E)
        n_iters += info['iterations']

        # Collapse to LEVELS inside this solve, before anything is merged across sigmas.
        #
        # This ordering matters and getting it wrong is not obvious. Kramers partners have
        # IDENTICAL energies, and so do the copies of one level returned by two overlapping
        # sigma windows -- energy alone cannot tell those apart. Deduplicating the merged list
        # first therefore throws away one member of every Kramers pair, after which pairing the
        # survivors pairs each level with its NEIGHBOUR and reports a level spacing as a
        # "Kramers splitting". That happened here: 35 meV, which is a spacing, not a splitting.
        # Grouping within a single solve keeps the two cases separate.
        deg = _degenerate_groups(E, tol=degeneracy_tol)
        for grp in deg:
            if loc[grp].mean() < loc_min:
                continue
            n_localized += len(grp)
            lv_E.append(float(E[grp].mean()))
            lv_loc.append(float(loc[grp].mean()))
            lv_sig.append(float(sg))
            lv_res.append(float(res[grp].max()))
            lv_deg.append(float(np.ptp(E[grp])) if len(grp) > 1 else np.nan)
        if verbose:
            kept = sum(1 for g in deg if loc[g].mean() >= loc_min)
            print(f"    sigma {sg:7.4f}: solve  {info['iterations']:>4} iters, "
                  f"residual {res.max():.1e} eV, {len(deg):>2} levels, "
                  f"{kept:>2} island-localized")

    lv_E = np.array(lv_E); lv_loc = np.array(lv_loc)
    lv_sig = np.array(lv_sig); lv_res = np.array(lv_res); lv_deg = np.array(lv_deg)

    # NOW merge across sigmas: same level, found twice by neighbouring windows.
    order = np.argsort(-lv_E)
    lv_E, lv_loc = lv_E[order], lv_loc[order]
    lv_sig, lv_res, lv_deg = lv_sig[order], lv_res[order], lv_deg[order]
    keep = []
    for j in range(len(lv_E)):
        if not keep or abs(lv_E[j] - lv_E[keep[-1]]) > degeneracy_tol:
            keep.append(j)
    keep = np.array(keep[:k], dtype=int) if keep else np.array([], dtype=int)

    worst = float(np.nanmax(lv_deg[keep])) if len(keep) and not np.all(
        np.isnan(lv_deg[keep])) else float('nan')

    # Did the scan actually COVER the energy range, or does it have holes in it?
    #
    # Each solve returns only `probe` eigenvalues around its own sigma, so it covers a finite
    # energy window. If two neighbouring sigmas' windows do not overlap, everything between them
    # is invisible to the scan -- and a level can sit there. That is not hypothetical: at
    # r = 6 nm with n_sigma = 5 and probe = 10, the two topmost levels fell into the gap between
    # the sigma = 0.48 and sigma = 0.32 windows and were silently absent from the ladder, while
    # every other check (sigma_hits not at the bottom of the scan, Kramers at 2e-11 meV,
    # residuals at 2e-7 eV) still passed. Coverage is the check that catches it, so it is
    # computed here rather than left to the caller.
    # `degeneracy_tol` as the threshold, not zero: two windows that abut to within the tolerance
    # used to call eigenvalues equal have not left room for a level between them, and reporting
    # a zero-width "gap" would be crying wolf. Seen in practice as [0.0420, 0.0420].
    spans_sorted = sorted(spans, key=lambda s: -s[0])
    gaps = [(spans_sorted[i][1], spans_sorted[i + 1][2])
            for i in range(len(spans_sorted) - 1)
            if spans_sorted[i][1] - spans_sorted[i + 1][2] > degeneracy_tol]
    covered_top = spans_sorted[0][2] if spans_sorted else float('nan')

    if verbose:
        print(f"    {len(lv_E)} island-localized levels found across the scan "
              f"({n_localized} states out of {n_probed} probed), {len(keep)} kept after "
              f"merging, {n_iters:,} iterations total, {time.time()-t0:.0f}s")
        if locate and n_skipped:
            print(f"    {n_skipped} of {len(sigmas)} sigmas skipped by the locate pass "
                  f"(continuum); their windows are still counted in the coverage check below")
        if not len(keep):
            print("    NOTHING localized anywhere in the scan. Widen `sigmas` or lower "
                  "`loc_min` -- the ladder is empty for a numerical reason, not a physical one.")
        else:
            print(f"    ladder top {lv_E[keep][0]:.4f} eV, i.e. "
                  f"{(top - lv_E[keep][0])*1e3:.0f} meV below the k=0 valence edge")
        if gaps:
            print(f"    WARNING: the sigma windows leave {len(gaps)} uncovered energy gap(s): " +
                  ", ".join(f"[{lo:.4f}, {hi:.4f}]" for hi, lo in gaps))
            print("      A level inside a gap is invisible to this scan and will be missing "
                  "from the ladder\n      WITHOUT any other check failing. Raise `probe` (wider "
                  "windows) or `n_sigma`.")
        else:
            print(f"    sigma windows overlap continuously up to {covered_top:.4f} eV: "
                  "no uncovered gaps")
    return dict(E=lv_E[keep], loc=lv_loc[keep], levels=lv_E[keep], level_loc=lv_loc[keep],
                kramers_split=worst, sigma=top, sigmas=sigmas,
                sigma_hits=lv_sig[keep] if len(keep) else np.array([]),
                residual=float(lv_res[keep].max()) if len(keep) else float('nan'),
                n_probed=int(n_probed), n_localized=int(n_localized),
                spans=spans, gaps=gaps, covered_top=covered_top,
                iterations=int(n_iters), n_skipped=int(n_skipped),
                seconds=float(time.time() - t0))


def run_size(shape, x, h=0.75, pad=5.0, k=8, use_piezo=True, matrix='InAs', z_pad=None,
             vol_tol=0.15, verbose=True, **kw):
    """One point of the size sweep: strain, pocket verdict, hole well, eight-band hole ladder.

    The `build` arguments are listed explicitly rather than swept up into `**kw`, which is
    forwarded to `hole_ladder`. Sharing one bag between the two would send `vol_tol` to the
    eigensolver and `probe` to the grid builder, and the resulting TypeError would land several
    frames from the caller that caused it.

    Returns a JSON-serializable dict, so `hole_size_sweep` can checkpoint it.
    """
    env = build(shape, x, h=h, pad=pad, z_pad=z_pad, use_piezo=use_piezo, matrix=matrix,
                vol_tol=vol_tol, verbose=verbose)
    pm = pocket_metrics(env, verbose=False)
    hw = hole_well(env)
    g = env['strain'].at(env['mask'])
    best = max(r['VR2'] for r in pm)

    t0 = time.time()
    lad = hole_ladder(env, k=k, verbose=verbose, **kw)
    out = dict(x=x, h=h, pad=pad, k=k, piezo=bool(use_piezo),
               label=shape['label'], kind=shape['kind'], params=dict(shape['params']),
               volume=float(shape['volume']), npts=int(env['mask'].size),
               vol_err=float(env['vol_err']),
               tr_mean=float(g['exx'] + g['eyy'] + g['ezz']),
               pocket_max=float(env['Ec_far'] - env['Ec'][~env['mask']].min()),
               pocket_VR2=float(best), pocket_binds=bool(best > POCKET_CRITERION),
               hole_well=float(hw['depth']), broken_gap=float(hw['broken_gap']),
               Ec_far=float(env['Ec_far']), Ev_far=float(env['Ev_far']),
               vb=dict(E=lad['E'].tolist(), loc=lad['loc'].tolist(),
                       levels=lad['levels'].tolist(), level_loc=lad['level_loc'].tolist(),
                       kramers_split=float(lad['kramers_split']),
                       valence_top=float(lad['sigma']), sigmas=lad['sigmas'].tolist(),
                       sigma_hits=np.asarray(lad['sigma_hits']).tolist(),
                       n_probed=lad['n_probed'], n_localized=lad['n_localized'],
                       residual=float(lad['residual']),
                       spans=[list(s) for s in lad['spans']],
                       gaps=[list(g) for g in lad['gaps']],
                       covered_top=float(lad['covered_top']),
                       iterations=int(lad['iterations']),
                       n_skipped=int(lad['n_skipped']),
                       seconds=float(time.time() - t0)))
    if verbose:
        print("    hole levels (eV, most confined first): " +
              ", ".join(f"{e:.4f}[{L*100:.0f}%]"
                        for e, L in zip(lad['levels'], lad['level_loc'])))
        print(f"    Kramers splitting within pairs: {lad['kramers_split']*1e3:.2e} meV")
    del env
    return out


def hole_size_sweep(shape_of, values, label='radius', x=1.0, cache=None, recompute=False,
                    **kw):
    """Run (or resume) a size sweep of the eight-band hole ladder.

    `shape_of(v)` returns the shape dict for each value of the swept parameter. Results are
    written to `cache` after every size, so an interrupted run resumes instead of restarting --
    the same arrangement as `pryor_fig4.sweep`, and for the same reason: this is the expensive
    part, minutes to tens of minutes per size.
    """
    done = {}
    if cache and os.path.exists(cache) and not recompute:
        with open(cache) as fh:
            done = {r[label]: r for r in json.load(fh)}
    for v in values:
        if v in done:
            print(f"{label} = {v}: loaded from {os.path.basename(cache)}")
            continue
        row = run_size(shape_of(v), x, **kw)
        row[label] = v
        done[v] = row
        if cache:
            with open(cache, 'w') as fh:
                json.dump([done[u] for u in sorted(done)], fh, indent=1)
    return [done[v] for v in sorted(done) if v in values]


def bound_hole_levels(results, loc_min=0.35):
    """(swept values, list of level arrays), keeping only the genuinely confined hole levels.

    Two tests, and a level has to pass both: it must lie above the unstrained InAs valence edge
    (the asymptotic barrier edge far from the island), and enough of its density must sit inside
    the island. In a broken-gap system the second test is not a refinement of the first, it is
    the load-bearing one -- the island's valence edge is hundreds of meV above the InAs
    *conduction* edge, so matrix-like electron box states exist at the same energies as the hole
    levels and a folded-spectrum solve will happily return them.

    Truncated at the first failure rather than filtered, because the levels come in order of
    confinement: once one is not bound, nothing above it is. Dropping one from the middle would
    silently renumber the rest, and "the n-th confined level" is what each curve is.
    """
    xs, lv = [], []
    for r in results:
        lvl = np.array(r['vb']['levels'])
        lloc = np.array(r['vb']['level_loc'])
        bad = np.flatnonzero((lvl <= r['Ev_far']) | (lloc < loc_min))
        n = int(bad[0]) if len(bad) else len(lvl)
        xs.append(r)
        lv.append(lvl[:n])
    return xs, lv


def plot_hole_ladder(results, xkey, loc_min=0.35, figsize=(6.2, 4.6), ax=None):
    """Hole levels against island size -- the counterpart of Pryor's Fig. 4(b).

    Energies run upward from the unstrained InAs valence edge, as in Pryor's panel, so a HIGHER
    point is a MORE strongly confined hole. The dashed line marking the InAs *conduction* edge
    has no counterpart in his figure and is the whole point of this one: every confined hole
    level sits above it, which is what a broken gap looks like.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    rows, lv = bound_hole_levels(results, loc_min)
    xs = [r[xkey] for r in rows]
    nmax = max((len(v) for v in lv), default=0)
    for j in range(nmax):
        x = [u for u, v in zip(xs, lv) if len(v) > j]
        y = [v[j] for v in lv if len(v) > j]
        ax.plot(x, y, 'o-', ms=3.5, lw=1.0, color='k', mfc='w' if j else 'k')
    ax.axhline(results[0]['Ev_far'], color='0.5', lw=0.9, ls='--')
    ax.axhline(results[0]['Ec_far'], color='#b03030', lw=0.9, ls='--')
    ax.text(xs[0], results[0]['Ev_far'], ' unstrained InAs $E_v$', va='bottom', ha='left',
            fontsize=8, color='0.4')
    ax.text(xs[0], results[0]['Ec_far'], ' unstrained InAs $E_c$', va='bottom', ha='left',
            fontsize=8, color='#b03030')
    ax.set(xlabel=f'{xkey}  (nm)', ylabel='$E$  (eV)')
    ax.margins(x=0.06, y=0.10)
    ax.set_title('Eight-band hole levels', fontsize=10)
    return ax


def plot_pocket_verdict(results, xkey, figsize=(6.2, 4.6), ax=None):
    """The conduction-band panel this system actually supports.

    Not an energy-vs-size plot: there is no bound electron to plot. What varies with size is how
    far the strain pocket falls short of being able to hold one, so that is what is drawn --
    V0*R^2 against the threshold, plus the pocket's deepest point on the right-hand axis to show
    that a deeper pocket is not the same thing as a binding one.
    """
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    xs = [r[xkey] for r in results]
    ax.plot(xs, [r['pocket_VR2'] for r in results], 'o-', color='k', ms=4,
            label=r'$V_0R^2$ of the pocket')
    ax.axhline(POCKET_CRITERION, color='#b03030', lw=1.0, ls='--',
               label='threshold to bind anything')
    ax.set_ylim(0, max(POCKET_CRITERION * 1.15,
                       max(r['pocket_VR2'] for r in results) * 1.15))
    ax.set(xlabel=f'{xkey}  (nm)', ylabel=r'$V_0R^2$  (eV nm$^2$)')
    ax.legend(fontsize=8, loc='center left')
    ax2 = ax.twinx()
    ax2.plot(xs, [r['pocket_max'] * 1e3 for r in results], 's:', color='#30609b', ms=4)
    ax2.set_ylabel('deepest point of the pocket (meV)', color='#30609b', fontsize=9)
    ax2.tick_params(axis='y', colors='#30609b')
    ax.set_title('Conduction band: can the strain pocket bind an electron?', fontsize=10)
    return ax
