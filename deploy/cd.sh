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

# Wait for the app to come up (up to 30s)
for i in $(seq 1 15); do
    if curl -sf http://127.0.0.1:8000/health/live > /dev/null 2>&1; then
        echo "Deploy complete."
        exit 0
    fi
    sleep 2
done

echo "Error: app did not become healthy after restart." >&2
exit 1
