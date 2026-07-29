"""Geometry checks for the dot-shape masks, against closed-form volumes and symmetries.

Cheap (seconds) and worth running before any expensive solve: a mask whose volume is wrong by
10% makes every confinement energy wrong by roughly the same amount, and the failure is invisible
downstream.

The dash is a truncated rectangular pyramid after J. Tersoff and R. M. Tromp, Phys. Rev. Lett.
70, 2782 (1993) -- rectangular base, sloping side facets, flat top.

Run: python scripts/shape_validation.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

import qdsolver_core as qd

TH = qd.DASH_CONTACT_ANGLE_DEG
ok = True


def report(label, mask, h, exact, tol):
    global ok
    err = qd.mask_volume_error(mask, h, exact)
    good = abs(err) < tol
    ok &= good
    print(f"  {label:46} vol err {err:+8.4f}   {'ok' if good else 'FAIL'} (tol {tol})")
    return err


print("TEST 1: dash volume against the closed form, on the cell-centred island grid")
L, W, H = 40.0, 12.0, 1.0
exact = qd.dash_volume(L, W, H, TH)
inset = 2 * H / np.tan(np.deg2rad(TH))
print(f"  truncated pyramid {L} x {W} x {H} nm, contact angle {TH} deg")
print(f"  top face {L-inset:.2f} x {W-inset:.2f} nm, exact volume {exact:.3f} nm^3")
errs = []
for h in (0.5, 0.25, 0.125):
    _, _, _, X, Y, Z = qd.island_grid((L, W, H), h, 4.0)
    m = qd.dash_mask(X, Y, Z, L, W, H, TH)
    errs.append(report(f"h = {h}", m, h, exact, 0.15))

# The absolute error is not the interesting quantity. A staircase approximation of a sloping
# facet is first-order accurate AT WORST, so what has to be true is that the error at least
# halves as h halves. A flat or erratic sequence would mean the grid is sampling the bounding
# planes rather than resolving the facets, which is the failure mode this grid convention exists
# to avoid -- see the contrast block below.
#
# There is deliberately no UPPER bound on the ratio. Converging faster than first order is not a
# failure, and it happens here: at the {111} angle the facet sets back h/tan(theta) = 0.707 cells
# per layer, which the cubic grid staircases far more cleanly than a shallow facet does, and the
# observed ratios run 2.5-4. An upper bound would flag that as broken.
print("  convergence: error should at least halve as h halves (first order is the worst case)")
for i in range(1, len(errs)):
    ratio = errs[i - 1] / errs[i]
    good = ratio > 1.7
    ok &= good
    print(f"    ratio {errs[i-1]:+.4f} / {errs[i]:+.4f} = {ratio:.2f}   "
          f"{'ok' if good else 'FAIL -- slower than first order'}")

print("\n  the same shape on a grid with sample points ON the bounding planes, for contrast:")
for h in (0.5, 0.25, 0.125):
    n = lambda e: int(round((e + 8.0) / h)) // 2 * 2 + 1
    cx, cy = qd.centered_axis(n(L), h), qd.centered_axis(n(W), h)
    cz = (np.arange(int(round(H / h)) + 3) - 1) * h
    X, Y, Z = np.meshgrid(cx, cy, cz, indexing='ij')
    err = qd.mask_volume_error(qd.dash_mask(X, Y, Z, L, W, H, TH), h, exact)
    print(f"    h = {h:<6} vol err {err:+8.4f}  <- does not converge away; it halves at best")

print("\nTEST 2: length == width gives the symmetric truncated pyramid, invariant under x<->y")
h = 0.25
_, _, _, X, Y, Z = qd.island_grid((12.0, 12.0, 1.0), h, 4.0)
sq = qd.dash_mask(X, Y, Z, 12.0, 12.0, 1.0, TH)
report("square dash 12 x 12 x 1", sq, h, qd.dash_volume(12.0, 12.0, 1.0, TH), 0.10)
diff = int((np.swapaxes(sq, 0, 1) != sq).sum())
print(f"  invariant under x<->y swap: {diff} differing voxels   {'ok' if diff == 0 else 'FAIL'}")
ok &= diff == 0

print("\nTEST 3: an elongated dash is C2, not C4 -- mirror symmetric but not swap symmetric")
_, _, _, X, Y, Z = qd.island_grid((L, W, H), h, 4.0)
d = qd.dash_mask(X, Y, Z, L, W, H, TH)
for ax in (0, 1):
    a = qd.mirror_asymmetry(d, ax)
    print(f"  mirror asymmetry about axis {ax}: {a}   {'ok' if a == 0 else 'FAIL'}")
    ok &= a == 0
nx, ny = d.shape[0], d.shape[1]
sub = d[:min(nx, ny), :min(nx, ny)]
c4 = int((np.swapaxes(sub, 0, 1) != sub).sum())
print(f"  differs under x<->y swap: {c4} voxels   {'ok -- C4 is broken, as intended' if c4 else 'FAIL'}")
ok &= c4 > 0

print("\nTEST 4: lens volume against the closed form")
for h in (0.5, 0.25):
    _, _, _, X, Y, Z = qd.island_grid((16.0, 16.0, 4.0), h, 4.0)
    report(f"lens r = 8, height = 4, h = {h}", qd.lens_mask(X, Y, Z, 8.0, 4.0), h,
           qd.lens_volume(8.0, 4.0), 0.03)

print("\nTEST 5: degenerate and impossible cases are rejected, not silently clipped")
tan_t = np.tan(np.deg2rad(TH))
h_max = (W / 2.0) * tan_t
_, _, _, X, Y, Z = qd.island_grid((L, W, h_max), 0.1, 4.0)
try:
    qd.dash_mask(X, Y, Z, L, W, h_max * 1.01, TH)
    print("  FAIL: over-tall dash accepted"); ok = False
except ValueError as exc:
    print(f"  over-tall dash rejected: {str(exc)[:66]}...   ok")
try:
    qd.dash_volume(10.0, 20.0, 0.5, TH)
    print("  FAIL: width > length accepted"); ok = False
except ValueError:
    print("  width > length rejected   ok")

ridge = qd.dash_mask(X, Y, Z, L, W, h_max, TH)
report(f"ridge limit, height = (W/2)tan(theta) = {h_max:.3f}", ridge, 0.1,
       qd.dash_volume(L, W, h_max, TH), 0.03)

print("\nTEST 6: constant-volume aspect-ratio series -- the Tersoff-Tromp elongation axis")
print("  Height is solved for at each aspect ratio so the volume is held fixed; any energy")
print("  difference along this series is then shape, not size.")
V0 = qd.dash_volume(20.0, 20.0, 1.0, TH)


def height_for_volume(length, width, target, angle=TH):
    """Bisect on height to hit a target volume -- the facet set-back removes more material from
    a narrow island, so holding L*W*h fixed does NOT hold the volume fixed."""
    lo, hi = 1e-6, (width / 2.0) * np.tan(np.deg2rad(angle))
    if qd.dash_volume(length, width, hi, angle) < target:
        return None
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if qd.dash_volume(length, width, mid, angle) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


print(f"  target volume {V0:.2f} nm^3")
print(f"  {'aspect':>7} {'length':>8} {'width':>8} {'height':>8} {'volume':>10} {'check':>8}")
for aspect in (1.0, 2.0, 4.0, 8.0):
    W_ = 20.0 / np.sqrt(aspect)
    L_ = W_ * aspect
    hh = height_for_volume(L_, W_, V0)
    if hh is None:
        print(f"  {aspect:>7.1f} {L_:>8.2f} {W_:>8.2f} {'--':>8} {'--':>10}   too narrow to "
              f"hold this volume at {TH} deg facets")
        continue
    v = qd.dash_volume(L_, W_, hh, TH)
    good = abs(v / V0 - 1) < 1e-9
    ok &= good
    print(f"  {aspect:>7.1f} {L_:>8.2f} {W_:>8.2f} {hh:>8.3f} {v:>10.2f} "
          f"{'ok' if good else 'FAIL':>8}")

print("\n" + ("ALL SHAPE TESTS PASSED" if ok else "SOME SHAPE TESTS FAILED"))
sys.exit(0 if ok else 1)
