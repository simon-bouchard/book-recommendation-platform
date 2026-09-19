#!/bin/bash
# ops/purge_large_blobs_from_history.sh
#
# ONE-OFF MAINTENANCE SCRIPT. Not part of any automated pipeline, not meant
# to be run casually — this rewrites git history and requires a force-push.
#
# Purges the largest sources of historical repo bloat (measured 2026-09-19,
# ~1.5 GB of raw blob data across history, compressing to a 294 MB .git):
#
#   models/training/data/*.pkl   739 MB  (repeated training-run artifacts,
#                                          committed before being gitignored)
#   data/ol_subjects/*           362 MB  (raw + 5 cleaning-stage intermediates)
#   data/*.csv, data/BX-Books.csv 246 MB (core dataset CSVs, now published
#                                          separately on Kaggle)
#   models/data/*                 62 MB  (npy/pickle model artifacts)
#
# Consequences (read before running):
#   - Every commit SHA after the first commit touching any of these paths
#     changes. This repo's own clone (and any other existing clone) will
#     diverge from the rewritten remote and need to be re-cloned fresh —
#     `git pull` will not cleanly reconcile.
#   - Any open PRs or branches based on the old history become incompatible.
#     Contributors on forks need to rebase their branch onto the new history
#     (`git rebase --onto`) or cherry-pick their commits onto a fresh branch
#     off the new master — GitHub can't auto-merge across the rewrite.
#   - CI/CD deploys on push to master (see .github/workflows/ci.yml) — the
#     force-push will trigger it. Harmless on its own (deploy/cd.sh just
#     does git pull + restart the app service), but expect it to fire.
#   - GitHub may still have cached copies of the removed data independent of
#     your repo (forks, cached PR diffs) — this guarantees removal from the
#     canonical history and any fresh clone, not everywhere GitHub might have
#     touched it.
#
# Usage:
#   1. Install git-filter-repo: pip install git-filter-repo
#   2. Run this script with the repo URL as the only argument:
#        ops/purge_large_blobs_from_history.sh https://github.com/simon-bouchard/book-recommendation-platform.git
#   3. Inspect the resulting mirror clone (printed at the end) before pushing.
#   4. When ready: cd into it and run
#        git push --force --all
#        git push --force --tags
#   5. Re-clone this working repo fresh afterward — don't try to reconcile
#      the old clone with `git pull`.

set -euo pipefail

if [ -z "${1:-}" ]; then
    echo "Usage: $0 <repo-url>" >&2
    exit 1
fi

REPO_URL="$1"
WORKDIR="$(mktemp -d)/repo-mirror"

echo "Cloning mirror to $WORKDIR ..."
git clone --mirror "$REPO_URL" "$WORKDIR"
cd "$WORKDIR"

echo "Rewriting history to remove large/obsolete data paths..."
git filter-repo \
    --path models/training/data \
    --path data/ol_subjects \
    --path data/books.csv \
    --path data/authors.csv \
    --path data/users.csv \
    --path data/interactions.csv \
    --path data/books_to_subjects.csv \
    --path data/users_to_subjects.csv \
    --path data/BX-Books.csv \
    --path models/data \
    --invert-paths

echo
echo "=== Done rewriting. Mirror is at: $WORKDIR ==="
echo "Inspect it, then push with:"
echo "  cd $WORKDIR && git push --force --all && git push --force --tags"
echo "Afterward, re-clone this repo fresh rather than reconciling the old clone."
