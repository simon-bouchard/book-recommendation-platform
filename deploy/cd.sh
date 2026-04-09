#!/usr/bin/env bash
# deploy/cd.sh
#
# Called by GitHub Actions via a restricted SSH key (command= in authorized_keys).
# Pulls the latest code and restarts the main app service.
# Model servers and training pipeline are NOT touched — redeploy those manually.

set -euo pipefail

REPO_ROOT="/home/simon/bookrec"

cd "$REPO_ROOT"
git pull origin master
sudo systemctl restart bookrec.service

echo "Deploy complete."
