"""6-band Luttinger-Kohn k.p Hamiltonian, bulk -- SUPERSEDED AND KNOWN WRONG.

Use kp_pryor.bulk_hamiltonian instead. Two errors, neither caught by the validation described
below: the split-off band is placed at -delta instead of +delta on the hole-convention
diagonal, and the R/S off-diagonal elements are in the wrong positions (21 meV of spurious
anisotropy in the spherical approximation, where the valence bands must be exactly isotropic).
The [001] effective-mass checks that this module WAS validated against cannot detect either,
because R and S vanish along [001] by symmetry. Kept only so the earlier notebooks still
import; do not use for new work.

Original docstring follows.

6-band Luttinger-Kohn k.p Hamiltonian, bulk (k treated as a number, not yet an operator).

This is deliberately the *first* step towards a multiband QD solver: before ever trying to
discretize k -> -i d/dr on a grid (which requires careful, error-prone treatment of the
off-diagonal cross-derivative terms like kx*ky), validate the Hamiltonian itself against known
bulk physics -- Hermiticity, the zero-k split-off placement, and the known [100] heavy/light
hole effective masses (1/m_hh = gamma1-2*gamma2, 1/m_lh = gamma1+2*gamma2, the same formula
aestimo's own Strain_and_Masses function uses in aestimo.py). See multiband_bulk_validation.ipynb
for that validation.

Basis order: |3/2,3/2>, |3/2,1/2>, |3/2,-1/2>, |3/2,-3/2>, |1/2,1/2>, |1/2,-1/2>
(heavy hole, light hole, light hole, heavy hole, split-off, split-off).

References:
 - J. M. Luttinger and W. Kohn, Phys. Rev. 97, 869 (1955): the original k.p Hamiltonian for
   degenerate valence bands, whose gamma1/gamma2/gamma3 parameters are used here.
 - S. L. Chuang, "Physics of Optoelectronic Devices" (Wiley, 1995), Ch. 4: the explicit 6x6
   matrix in this |J, m_J> basis with the standard P/Q/R/S combinations used below. This is
   the same reference aestimo's own aeslibs/VBHM.py cites for its 1D quantum-well version of
   this Hamiltonian (its header points at p. 183 for the Hermitian operator-ordering form).
 - Per the citation policy in qdsolver_core.py: equation numbers from these sources are not
   quoted because they could not be verified against the originals; the matrix elements below
   were instead validated numerically (Hermiticity, k=0 split-off placement, [001] effective
   masses vs the gamma1 -/+ 2*gamma2 formulas) in multiband_bulk_validation.ipynb.
"""
import numpy as np

from qdsolver_core import HBAR2_OVER_2M0


def luttinger_kohn_bulk(kx, ky, kz, gamma1, gamma2, gamma3, delta_so):
    """Build the 6x6 bulk Luttinger-Kohn Hamiltonian (eV) at wavevector (kx,ky,kz) (1/nm).

    Only the upper triangle is written out explicitly from the standard P/Q/R/S combinations;
    the rest is filled in by Hermitian completion (H = triu + triu(strict)^dagger), so a
    transcription error in one of these formulas would show up as a wrong *eigenvalue*, never
    as a Hermiticity violation -- Hermiticity is guaranteed by construction, not by getting the
    algebra right, which is exactly why the bulk effective-mass/split-off checks in the
    validation notebook (not just a Hermiticity check) matter.
    """
    k2 = kx**2 + ky**2 + kz**2
    P = HBAR2_OVER_2M0 * gamma1 * k2
    Q = HBAR2_OVER_2M0 * gamma2 * (kx**2 + ky**2 - 2 * kz**2)
    R = HBAR2_OVER_2M0 * (-np.sqrt(3) * gamma2 * (kx**2 - ky**2) + 1j * 2 * np.sqrt(3) * gamma3 * kx * ky)
    S = HBAR2_OVER_2M0 * 2 * np.sqrt(3) * gamma3 * (kx - 1j * ky) * kz

    H = np.zeros((6, 6), dtype=complex)
    H[0, 0] = P + Q
    H[1, 1] = P - Q
    H[2, 2] = P - Q
    H[3, 3] = P + Q
    H[4, 4] = P - delta_so
    H[5, 5] = P - delta_so

    H[0, 1] = -S
    H[0, 2] = R
    H[0, 3] = 0
    H[0, 4] = -np.conj(S) / np.sqrt(2)
    H[0, 5] = np.sqrt(2) * R

    H[1, 2] = 0
    H[1, 3] = R
    H[1, 4] = -np.sqrt(2) * Q
    H[1, 5] = np.sqrt(1.5) * S

    H[2, 3] = S
    H[2, 4] = np.sqrt(1.5) * np.conj(S)
    H[2, 5] = np.sqrt(2) * Q

    H[3, 4] = -np.sqrt(2) * R
    H[3, 5] = S / np.sqrt(2)

    H[4, 5] = 0

    return np.triu(H) + np.triu(H, 1).conj().T


BAND_LABELS = ["HH+3/2", "LH+1/2", "LH-1/2", "HH-3/2", "SO+1/2", "SO-1/2"]
