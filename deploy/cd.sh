#!/usr/bin/env bash
# deploy/cd.sh
#
# Called by GitHub Actions via a restricted SSH key (command= in authorized_keys).
# Pulls the latest code, rebuilds the frontend, and restarts the main app service.
# Model servers and training pipeline are NOT touched — redeploy those manually.

set -euo pipefail

REPO_ROOT="/home/simon/bookrec"

cd "$REPO_ROOT"
git pull origin master

# Rebuild frontend — load nvm so the correct Node version is available
export NVM_DIR="/home/simon/.nvm"
# shellcheck source=/dev/null
[[ -s "$NVM_DIR/nvm.sh" ]] && source "$NVM_DIR/nvm.sh" && nvm use 20

cd "$REPO_ROOT/frontend"
npm ci
npm run build
cd "$REPO_ROOT"

sudo systemctl restart bookrec.service

# Wait for the app to come up (up to 150s — covers graceful_timeout=120s + startup)
for i in $(seq 1 75); do
    if curl -sf http://localhost:8000/health/live > /dev/null 2>&1; then
        echo "Deploy complete."
        exit 0
    fi
    sleep 2
done

echo "Error: app did not become healthy after restart." >&2
exit 1
