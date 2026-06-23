#!/usr/bin/env bash
# Copy the deterministic-pipeline Python modules into the static site so the
# in-browser optimizer can fetch them. Re-run whenever svgym/*.py changes
# (e.g. before pushing). Keeps the live tool in sync with the package.
set -e
cd "$(dirname "$0")"
mkdir -p svgym
for m in __init__.py config.py tools.py deterministic.py; do
  cp ../../svgym/"$m" "svgym/$m"
done
echo "engine modules copied into docs/app/svgym/"
