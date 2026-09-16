#!/usr/bin/env bash
set -euo pipefail

REPO="mostafatsom/TPMO"
DESCRIPTION="Two-Phase Metamodel-Driven Optimization with adaptive sampling, Extensive Constraints, XGBoost metamodeling, and FSGO search."
WEBSITE="https://doi.org/10.22068/ijoce.2025.15.2.630"

command -v git >/dev/null || { echo "git is required" >&2; exit 1; }
command -v gh >/dev/null || { echo "GitHub CLI (gh) is required" >&2; exit 1; }
gh auth status >/dev/null

if [ ! -d .git ]; then
  git init -b main
fi

git add .
if ! git diff --cached --quiet; then
  git commit -m "Initial TPMO release"
fi

if gh repo view "$REPO" >/dev/null 2>&1; then
  git remote get-url origin >/dev/null 2>&1 || git remote add origin "https://github.com/${REPO}.git"
else
  gh repo create "$REPO" --public --source=. --remote=origin --description "$DESCRIPTION" --homepage "$WEBSITE"
fi

git push -u origin main

gh repo edit "$REPO" \
  --description "$DESCRIPTION" \
  --homepage "$WEBSITE" \
  --add-topic optimization \
  --add-topic surrogate-optimization \
  --add-topic adaptive-sampling \
  --add-topic metamodel \
  --add-topic xgboost \
  --add-topic fsgo \
  --add-topic expensive-optimization \
  --add-topic engineering-optimization \
  --add-topic python

if ! git rev-parse v1.0.0 >/dev/null 2>&1; then
  git tag -a v1.0.0 -m "TPMO v1.0.0"
  git push origin v1.0.0
fi

if ! gh release view v1.0.0 --repo "$REPO" >/dev/null 2>&1; then
  gh release create v1.0.0 --repo "$REPO" --title "TPMO v1.0.0" --notes-file RELEASE_NOTES.md
fi

echo "Published: https://github.com/${REPO}"
