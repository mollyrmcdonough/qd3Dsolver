"""Strain-induced piezoelectric potential for a zincblende quantum dot.

In a zincblende crystal the only independent piezoelectric constant is e14, and the polarization
is driven purely by the SHEAR strain,

    P_x = 2 e14 eps_yz ,   P_y = 2 e14 eps_zx ,   P_z = 2 e14 eps_xy

(indices cyclic). The bound charge rho_b = -div P then sources an electrostatic potential which
adds to the band edges. Pryor's Table I carries e14 for both materials and his Sec. II states
the effect is included: "In addition to the explicit strain dependence in H_s, there is a small
piezoelectric effect which is included. The strain induced polarization of the material
contributes an additional electrostatic potential which breaks the C_4 symmetry of the islands
to C_2." (C. Pryor, Phys. Rev. B 57, 7190 (1998); he cites O. Stier and D. Bimberg, Phys. Rev. B
55, 7726 (1997) alongside that statement.)

Elastic vs total strain is not an ambiguity here: the lattice-mismatch eigenstrain is purely
dilatational, so the elastic and total strain tensors have IDENTICAL shear components, which are
the only ones that enter. Either field gives the same polarization.

The symmetry statement is the sharp test
----------------------------------------
A square-based pyramid is C4v as a shape, but the zincblende [100]/[010] axes are
inequivalent, and P depends on shear, so the piezoelectric potential is ODD under the 90-degree
rotation that the shape is even under: phi(C4 r) = -phi(r). `c4_antisymmetry_error` measures
this directly and it must vanish to machine precision -- it is a property of the tensor
structure, not of the shape or the solver, so any violation is a bug (in practice: an
asymmetric coordinate grid, which is exactly how a real grid bug was found here; see
qdsolver_core.centered_axis).

That antisymmetry also means phi vanishes IDENTICALLY on the [001] axis, since points on the
axis are fixed by C4. Anything computed along [001] alone is therefore blind to piezoelectricity.

What is NOT included
--------------------
Second-order (nonlinear) piezoelectricity. First-order alone is now known to be an unreliable
approximation for these dots: the second-order response is comparable in magnitude and often
opposes it, so the net field can be much smaller than the linear estimate -- G. Bester,
A. Zunger, X. Wu and D. Vanderbilt, Phys. Rev. B 74, 081305(R) (2006), and G. Bester, X. Wu,
D. Vanderbilt and A. Zunger, Phys. Rev. Lett. 96, 187602 (2006). Pryor (1998) predates that
work and uses first order, so first order is what is implemented here for comparability.

Units: strain dimensionless, e14 in C/m^2, lengths in nm, potential in Volts (so a singly
charged carrier's potential energy in eV is numerically +/- phi).
"""
import numpy as np

from poisson_solver import E_CHARGE, POISSON_CONST

# rho = -div P with P in C/m^2 and lengths in nm gives C/(m^2 nm); converting to elementary
# charges per nm^3 costs a factor 1e9 (nm^-1 -> m^-1) over E_CHARGE * 1e27 (m^-3 -> nm^-3).
CHARGE_UNIT = 1.0 / (E_CHARGE * 1e18)


def _k_grids(shape, h):
    k = [2 * np.pi * np.fft.fftfreq(n, d=h) for n in shape]
    return np.meshgrid(*k, indexing='ij')


def polarization(strain, e14, sign=+1.0):
    """(P_x, P_y, P_z) in C/m^2 from the shear strain. `e14` may be a scalar or a grid field.

    `sign` flips the crystallographic convention for the sign of e14 (the anion/cation
    assignment of the [111] direction), which differs between sources. It is exposed rather
    than buried because the resulting potential changes sign with it, and the sign is not
    determined by anything else in this package.
    """
    return (2.0 * e14 * strain.eyz * sign,
            2.0 * e14 * strain.ezx * sign,
            2.0 * e14 * strain.exy * sign)


def charge_density(P, h):
    """Bound charge -div P in elementary charges per nm^3, by spectral differentiation.

    Spectral rather than finite-difference so that it is consistent with `potential`, which
    inverts the Laplacian in the same Fourier basis: mixing an FD divergence with a spectral
    Poisson solve leaves a discretization mismatch that shows up as spurious short-wavelength
    structure at the dot facets.
    """
    KX, KY, KZ = _k_grids(P[0].shape, h)
    div = (1j * KX * np.fft.fftn(P[0]) + 1j * KY * np.fft.fftn(P[1])
           + 1j * KZ * np.fft.fftn(P[2]))
    return -np.fft.ifftn(div).real * CHARGE_UNIT


def potential(strain, e14, eps_r, h, sign=+1.0):
    """Piezoelectric potential (Volts) on a periodic cell.

    Solves eps_r k^2 phi_k = rho_k * POISSON_CONST with the k = 0 mode dropped (a periodic cell
    must be charge neutral, and the mean potential is arbitrary). `eps_r` is taken as a scalar:
    Pryor's Table I footnote states the InAs dielectric constant is used throughout the
    structure, so there is no dielectric mismatch to resolve.
    """
    rho = charge_density(polarization(strain, e14, sign), h)
    KX, KY, KZ = _k_grids(rho.shape, h)
    k2 = KX ** 2 + KY ** 2 + KZ ** 2
    inv = np.zeros_like(k2)
    np.divide(1.0, k2, out=inv, where=k2 != 0.0)
    return np.fft.ifftn(np.fft.fftn(rho) * POISSON_CONST * inv / eps_r).real


def c4_rotate(field):
    """Sample `field` at C4-rotated coordinates: (x, y, z) -> (-y, x, z).

    With a mirror-symmetric axis (qdsolver_core.centered_axis), -y_j is y_{N-1-j}, so the
    rotated array is out[i,j,k] = field[N-1-j, i, k]. Requires the x and y axes to be identical,
    which every grid in this package satisfies.
    """
    f = np.asarray(field)
    assert f.shape[0] == f.shape[1], "C4 rotation needs identical x and y axes"
    return np.swapaxes(np.flip(f, axis=0), 0, 1)


def c4_antisymmetry_error(phi):
    """max |phi(C4 r) + phi(r)| / max|phi| -- must vanish to machine precision.

    See the module docstring: this is fixed by the tensor structure of the zincblende
    piezoelectric response, independently of dot shape, so a nonzero value is a bug rather than
    physics. It was in fact how a mirror-asymmetric coordinate grid was caught here.
    """
    scale = np.abs(phi).max()
    if scale == 0.0:
        return 0.0
    return float(np.abs(c4_rotate(phi) + phi).max() / scale)


def field_strength(phi, h):
    """(max |E|, in kV/cm) of the piezoelectric field, for comparison with the literature."""
    g = np.gradient(phi, h)          # V/nm
    mag = np.sqrt(sum(x ** 2 for x in g))
    return float(mag.max() * 1e7 / 1e3)   # V/nm -> V/cm -> kV/cm
