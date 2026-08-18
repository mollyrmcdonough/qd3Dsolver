"""Convert a pre-2026-08-05 transition sweep from h/d to AR = d/h.

Aspect ratio in this package is now **AR = d/h** (diameter over height), the literature
convention: a larger AR is a flatter island. Sweeps written before the change stored `aspect`
= h/d, both as a record field and inside the resume key.

This rewrites both, in place, after taking a `.h_over_d.bak` copy. It is worth doing rather than
teaching every reader a compatibility shim: `_transitions_77K.json` cost 11,646 s of solves and
must stay resumable, but a file where half the records mean d/h and half mean h/d is exactly the
kind of silent reciprocal error the convention audit exists to catch.

The conversion is only a reciprocal and a rename -- no physics is touched, and `height` is already
stored per record, so `AR = base / height` is recomputed from the geometry rather than from the
ratio, which double-checks the old field on the way through.

Run:  python scripts/migrate_aspect_to_AR.py [_transitions_77K.json ...]
"""
import json
import os
import shutil
import sys


def new_key(old):
    """`x1.00_InAs_b20_a0.080_T77` -> `x1.00_InAs_b20_AR12.500_T77`."""
    parts = old.split('_')
    a = float(parts[-2][1:])            # 'a0.080'
    parts[-2] = f"AR{1.0 / a:.3f}"
    return '_'.join(parts)


def migrate(path):
    data = json.load(open(path))
    if not any('aspect' in r for r in data.values()):
        print(f"{path}: already on AR = d/h, nothing to do")
        return 0
    shutil.copy(path, path + '.h_over_d.bak')

    out = {}
    for k, rec in data.items():
        if 'aspect' not in rec:
            out[k] = rec
            continue
        aspect = rec.pop('aspect')
        AR = rec['base'] / rec['height']
        # The stored height is the ground truth; a disagreement means the record was hand-edited.
        assert abs(AR - 1.0 / aspect) < 1e-6, f"{k}: base/height = {AR}, 1/aspect = {1/aspect}"
        rec['AR'] = AR
        out[new_key(k)] = rec

    # Rebuild in a stable order so the diff is readable.
    out = dict(sorted(out.items(), key=lambda kv: (kv[1].get('x', 0), -kv[1].get('AR', 0),
                                                   kv[1].get('base', 0))))
    json.dump(out, open(path, 'w'), indent=1)
    print(f"{path}: {len(out)} record(s) -> AR = d/h  (backup at {os.path.basename(path)}"
          f".h_over_d.bak)")
    return len(out)


if __name__ == '__main__':
    paths = sys.argv[1:] or ['_transitions_77K.json']
    for p in paths:
        if os.path.exists(p):
            migrate(p)
        else:
            print(f"{p}: not found, skipped")
