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

What binds the electron at all is strain, not a band offset. The dot is compressed by the matrix
(the misfit runs -0.6% to -6.5% with composition), and by reaction the InAs immediately around it
is put into tension. With a_c negative, tension pulls the InAs conduction edge DOWN, so the dot
digs a shallow conduction-band pocket in the matrix wrapped around itself. `electron_states`
solves that pocket. It is a genuinely type-II, spatially indirect structure, and the electron
binding is a strain effect that a calculation with uniform strain inside the dot and zero outside
would miss entirely -- which is exactly why the inhomogeneous FD elasticity solver is used here.

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
          vol_tol=0.15, verbose=True):
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
    strain = ef.solve_strain_fd(mask, eps_star, ms.elastic(dot), ms.elastic(mat), h, tol=1e-10)
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
              f"{t_strain:.1f}s")
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
        shape, x = shape_of(v)
        try:
            env = build(shape, x, h=h, pad=pad, use_piezo=use_piezo, verbose=verbose)
        except ValueError as exc:
            print(f"  {label} = {v}: skipped -- {exc}")
            continue
        st = electron_states(env, k=2, verbose=False)
        hw = hole_well(env)
        g = env['strain'].at(env['mask'])
        rows.append(dict(**{label: v}, x=x, E0=float(st['E'][0]),
                         binding=env['Ec_far'] - float(st['E'][0]),
                         inside=float(st['inside'][0]), n_bound=st['n_bound'],
                         hole_depth=hw['depth'], broken_gap=hw['broken_gap'],
                         tr=float(g['exx'] + g['eyy'] + g['ezz']),
                         volume=shape['volume'], vol_err=env['vol_err']))
        print(f"  {label} = {v!r:>10}: E0 = {rows[-1]['E0']:.4f} eV "
              f"(binding {rows[-1]['binding']*1e3:6.1f} meV, {rows[-1]['inside']*100:4.1f}% in "
              f"dot), hole well {hw['depth']*1e3:6.1f} meV, Tr eps {rows[-1]['tr']:+.4f}")
        del env
    return rows


def plot_sweep(rows, label, figsize=(11, 3.6)):
    """Electron binding, hole well depth and mean strain against the swept parameter."""
    import matplotlib.pyplot as plt
    v = [r[label] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].plot(v, [r['binding'] * 1e3 for r in rows], 'o-', color='k', ms=4)
    axes[0].axhline(0, color='0.6', lw=0.8, ls='--')
    axes[0].set_ylabel('electron binding (meV)')
    axes[0].set_title('electron in the strain pocket', fontsize=9)
    axes[1].plot(v, [r['hole_depth'] * 1e3 for r in rows], 'o-', color='#b03030', ms=4)
    axes[1].set_ylabel('hole well depth (meV)')
    axes[1].set_title('$E_v$(dot) above unstrained InAs $E_v$', fontsize=9)
    axes[2].plot(v, [r['tr'] for r in rows], 'o-', color='#30609b', ms=4)
    axes[2].set_ylabel(r'$\langle$Tr $\varepsilon\rangle$ in the dot')
    axes[2].set_title('mean hydrostatic strain', fontsize=9)
    for ax in axes:
        ax.set_xlabel(label)
    fig.tight_layout()
    return fig
