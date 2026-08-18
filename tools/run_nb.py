"""Execute a notebook in place, saving after EVERY cell.

`jupyter nbconvert --execute` only writes its output when the whole notebook finishes, so a run
that dies at hour three leaves nothing behind. These solves take hours and one has already died
overnight, so this saves incrementally: every completed cell's output is on disk immediately.

Run:  python tools/run_nb.py hole_states.ipynb [--timeout SECONDS]
"""
import os
import sys
import time

import nbformat as nbf
from nbclient import NotebookClient

args = [a for a in sys.argv[1:] if not a.startswith('--')]
if not args:
    sys.exit(__doc__)
path = os.path.abspath(args[0])
timeout = 36000
if '--timeout' in sys.argv:
    timeout = int(sys.argv[sys.argv.index('--timeout') + 1])

nb = nbf.read(path, as_version=4)
n_code = sum(1 for c in nb.cells if c.cell_type == 'code')
print(f"executing {os.path.basename(path)}: {len(nb.cells)} cells "
      f"({n_code} code), timeout {timeout}s per cell", flush=True)

t0 = time.time()
done = [0]


def save(cell, cell_index, execute_reply):
    """Persist after each cell, so a death costs one cell rather than the run."""
    if cell.cell_type == 'code':
        done[0] += 1
        status = execute_reply.get('content', {}).get('status', '?')
        print(f"  [{done[0]}/{n_code}] cell {cell_index} {status}  "
              f"({time.time()-t0:.0f}s elapsed)", flush=True)
    nbf.write(nb, path)


client = NotebookClient(nb, timeout=timeout, kernel_name='python3',
                        resources={'metadata': {'path': os.path.dirname(path)}},
                        allow_errors=True, on_cell_executed=save)
client.execute()
nbf.write(nb, path)
print(f"DONE in {time.time()-t0:.0f}s -> {path}", flush=True)
