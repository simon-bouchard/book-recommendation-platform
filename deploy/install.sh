#!/usr/bin/env bash
# deploy/install.sh
#
# Substitutes deploy.env values into all templates and installs the
# resulting files to their system locations.
#
# Usage:
#   cd /path/to/repo
#   cp deploy/deploy.env.example deploy/deploy.env
#   # edit deploy/deploy.env with real values
#   sudo deploy/install.sh
#
# The script must be run as root (or via sudo) so it can write to
# /etc/systemd/system/ and /etc/nginx/sites-available/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/deploy.env"

# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

if [[ $EUID -ne 0 ]]; then
    echo "Error: this script must be run as root." >&2
    exit 1
fi

if [[ ! -f "$ENV_FILE" ]]; then
    echo "Error: $ENV_FILE not found." >&2
    echo "Copy deploy/deploy.env.example to deploy/deploy.env and fill in the values." >&2
    exit 1
fi

source "$ENV_FILE"

required_vars=(USER REPO_ROOT CONDA_ENV_BIN ENV_FILE DOMAIN WEBROOT GRAFANA_HTPASSWD)
for var in "${required_vars[@]}"; do
    if [[ -z "${!var:-}" ]]; then
        echo "Error: $var is not set in deploy/deploy.env." >&2
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Substitution helper
# ---------------------------------------------------------------------------

# Substitute all {{PLACEHOLDER}} tokens in a template file and write the
# result to a destination path.
install_template() {
    local template="$1"
    local dest="$2"

    sed \
        -e "s|{{USER}}|${USER}|g" \
        -e "s|{{REPO_ROOT}}|${REPO_ROOT}|g" \
        -e "s|{{CONDA_ENV_BIN}}|${CONDA_ENV_BIN}|g" \
        -e "s|{{ENV_FILE}}|${ENV_FILE}|g" \
        -e "s|{{DOMAIN}}|${DOMAIN}|g" \
        -e "s|{{WEBROOT}}|${WEBROOT}|g" \
        -e "s|{{GRAFANA_HTPASSWD}}|${GRAFANA_HTPASSWD}|g" \
        "$template" > "$dest"

    echo "Installed: $template -> $dest"
}

# ---------------------------------------------------------------------------
# Systemd units
# ---------------------------------------------------------------------------

SYSTEMD_DIR="/etc/systemd/system"
TEMPLATE_DIR="$SCRIPT_DIR/systemd"

install_template "$TEMPLATE_DIR/bookrec.service.template"                    "$SYSTEMD_DIR/bookrec.service"
install_template "$TEMPLATE_DIR/bookrec-docker.service.template"             "$SYSTEMD_DIR/bookrec-docker.service"
install_template "$TEMPLATE_DIR/bookrec-train-automation.service.template"   "$SYSTEMD_DIR/bookrec-train-automation.service"
install_template "$TEMPLATE_DIR/bookrec-train-automation.timer.template"     "$SYSTEMD_DIR/bookrec-train-automation.timer"

systemctl daemon-reload
echo "systemd daemon reloaded."

# ---------------------------------------------------------------------------
# Nginx vhost
# ---------------------------------------------------------------------------

NGINX_AVAILABLE="/etc/nginx/sites-available"
NGINX_ENABLED="/etc/nginx/sites-enabled"
NGINX_CONF="${DOMAIN}"

install_template "$SCRIPT_DIR/nginx/recsys.conf.template" "$NGINX_AVAILABLE/$NGINX_CONF"

if [[ ! -L "$NGINX_ENABLED/$NGINX_CONF" ]]; then
    ln -s "$NGINX_AVAILABLE/$NGINX_CONF" "$NGINX_ENABLED/$NGINX_CONF"
    echo "Enabled nginx vhost: $NGINX_CONF"
fi

if nginx -t; then
    systemctl reload nginx
    echo "nginx reloaded."
else
    echo "Warning: nginx config test failed. Fix errors before reloading nginx." >&2
fi

systemctl enable bookrec-docker.service
systemctl enable bookrec.service
systemctl enable bookrec-train-automation.timer

systemctl restart bookrec-docker.service
systemctl restart bookrec.service
systemctl restart bookrec-train-automation.timer

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------

echo ""
echo "Installation complete"
echo "bookrec-docker.service, bookrec.service and bookrec-train-automation.timer enabled and restarted"
