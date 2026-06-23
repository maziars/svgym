#!/usr/bin/env bash
# build-pyodide.sh — vendor the exact Pyodide runtime + the numpy/scipy/fonttools
# dependency closure into docs/app/pyodide/, so the in-browser optimizer loads
# entirely from our own GitHub Pages origin (no jsdelivr / PyPI at runtime). The
# service worker then caches these for instant repeat loads.
#
# Run ONCE on a machine with internet (re-run only to bump PYVER):
#     bash docs/app/build-pyodide.sh
# then commit the docs/app/pyodide/ folder and push.
set -euo pipefail
cd "$(dirname "$0")"

PYVER="0.26.2"                                   # keep in sync with svgym-worker.js PYVER
BASE="https://cdn.jsdelivr.net/pyodide/v${PYVER}/full"
DEST="pyodide"
WANT="numpy scipy fonttools"

mkdir -p "$DEST"

echo "==> core runtime (v${PYVER})"
for f in pyodide.js pyodide.asm.js pyodide.asm.wasm python_stdlib.zip pyodide-lock.json; do
  echo "    $f"
  curl -fsSL "$BASE/$f" -o "$DEST/$f"
done

echo "==> resolving dependency closure for: ${WANT}"
python3 - "$DEST" "$BASE" $WANT <<'PY'
import json, os, sys, urllib.request
dest, base = sys.argv[1], sys.argv[2]
want = [w.lower() for w in sys.argv[3:]]
lock = json.load(open(os.path.join(dest, "pyodide-lock.json")))
pkgs = lock["packages"]                          # keys are normalized lowercase names
seen, stack = set(), list(want)
while stack:
    n = stack.pop().lower()
    if n in seen:
        continue
    if n not in pkgs:
        print("    WARN: %s not found in lockfile" % n); continue
    seen.add(n)
    stack += [d.lower() for d in pkgs[n].get("depends", [])]
print("    closure: " + ", ".join(sorted(seen)))
for n in sorted(seen):
    fn = pkgs[n]["file_name"]
    out = os.path.join(dest, fn)
    if os.path.exists(out):
        print("    have  " + fn); continue
    print("    fetch " + fn)
    urllib.request.urlretrieve(base + "/" + fn, out)
PY

echo "==> done."
du -sh "$DEST"
echo "    $(ls "$DEST" | wc -l | tr -d ' ') files in $DEST/"
echo "Next: git add docs/app/pyodide && git commit -m 'Self-host Pyodide runtime' && git push"
