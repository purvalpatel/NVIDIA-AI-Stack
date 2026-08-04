# GPU Server Monitoring — Exporter Setup Guide


## Table of Contents

1. [Node Exporter Setup](#1-node-exporter-setup)
2. [DCGM Exporter Overview](#2-dcgm-exporter-overview)
3. [Building the DCGM Exporter .deb Package](#3-building-the-dcgm-exporter-deb-package)
4. [Pre-Install Script — What It Does](#4-pre-install-script--what-it-does)
5. [Step-by-Step Manual Installation](#5-step-by-step-manual-installation)
6. [Post-Install: Change Password](#6-post-install-change-password)
7. [Scripts Reference](#7-scripts-reference)

---

## 1. Node Exporter Setup

### Download and Install

```bash
cd /tmp/
wget http://ftp.ubuntu.com/ubuntu/ubuntu/pool/universe/p/prometheus-node-exporter/prometheus-node-exporter_1.11.1-2_amd64.deb
sudo dpkg -i prometheus-node-exporter_1.11.1-2_amd64.deb
```

### Configure Authentication and Port

```bash
sudo mkdir -p /etc/prometheus-node-exporter && \
cat <<'EOF' | sudo tee /etc/prometheus-node-exporter/web-config.yml >/dev/null
basic_auth_users:
  prometheus: $2y$10$6a2wttZOOipUd2l7qWDBF.J7g13osPhOD9OCVrdKvJYrfBXlB7mcS
EOF

sudo sed -i 's|^ARGS=".*"|ARGS="--web.listen-address=:10100 --web.config.file=/etc/prometheus-node-exporter/web-config.yml"|' /etc/default/prometheus-node-exporter

cat /etc/default/prometheus-node-exporter /etc/prometheus-node-exporter/web-config.yml
```

### Enable and Start

```bash
sudo systemctl enable --now prometheus-node-exporter
sudo systemctl restart prometheus-node-exporter
sudo systemctl status prometheus-node-exporter
ss -tlnp | grep 10100
curl -s -u 'prometheus:Medg35us' http://localhost:10100/metrics | head -n 15
```

---

## 2. DCGM Exporter Overview

DCGM Exporter collects GPU metrics (temperature, memory, utilization, clocks, power, ECC errors) from NVIDIA's Data Center GPU Manager (DCGM) and exposes them as Prometheus metrics.

### Architecture

```
Prometheus → dcgm-exporter (:10101, TLS + basic auth)
                  ↓
            nvidia-dcgm (nv-hostengine)
                  ↓
            NVML → NVIDIA kernel module → GPU
```

### What We Built

- Custom `.deb` package with port 10101, TLS certificates, and basic auth baked in
- Systemd service file matching the `prometheus-node-exporter` style (`EnvironmentFile` + `$ARGS`)
- Pre-install script that handles prerequisites automatically

### Service File Style

The service file follows the same pattern as `prometheus-node-exporter`:

```ini
[Unit]
Description=Prometheus exporter for NVIDIA GPU metrics (DCGM)
Documentation=https://github.com/NVIDIA/dcgm-exporter
After=nvidia-dcgm.service
Requires=nvidia-dcgm.service

[Service]
Restart=on-failure
User=root
EnvironmentFile=/etc/default/dcgm-exporter
ExecStart=/usr/bin/dcgm-exporter $ARGS
ExecReload=/bin/kill -HUP $MAINPID
TimeoutStopSec=20s
SendSIGKILL=no

[Install]
WantedBy=multi-user.target
```

The `$ARGS` variable is read from `/etc/default/dcgm-exporter`:

```
ARGS="-a :10101 -f /etc/dcgm-exporter/no-profiling-counters.csv --web-config-file /etc/dcgm-exporter/web-config.yaml"
```

---

## 3. Building the DCGM Exporter .deb Package

### Prerequisites (Build Machine Only)

```bash
sudo apt-get install --yes git golang-go dpkg-dev openssl apache2-utils
```

DCGM libraries must also be installed (for Go CGo linking against `libdcgm.so`).

### Build Script (`dcgm-build.sh`)

```bash
#!/usr/bin/env bash
###############################################################################
#  build-deb.sh — Build dcgm-exporter .deb with port + auth baked in
###############################################################################
VERSION="4.5.3"
REVISION="1"
LISTEN_PORT="10101"
AUTH_USER="prometheus"
AUTH_PASS="38475ndfkghkhfg"
###############################################################################

set -euo pipefail

PKG="dcgm-exporter"
PKG_FULL="${PKG}_${VERSION}-${REVISION}_amd64"
BUILD="${PWD}/build"
ROOT="${BUILD}/${PKG_FULL}"
SRC="${BUILD}/src"

echo "══════════════════════════════════════════════════════════════"
echo "  Building ${PKG} ${VERSION}-${REVISION}"
echo "  Port: ${LISTEN_PORT}   Auth: ${AUTH_USER}/********"
echo "══════════════════════════════════════════════════════════════"

rm -rf "${BUILD}"
mkdir -p "${SRC}"

# ── 1. Clone and compile ────────────────────────────────────────────
echo "[1/6] Cloning dcgm-exporter..."
git clone --depth 1 --branch main https://github.com/NVIDIA/dcgm-exporter.git "${SRC}/dcgm-exporter"

echo "[2/6] Compiling binary..."
cd "${SRC}/dcgm-exporter"
make binary
cd "${BUILD}"

# ── 2. Generate TLS cert + bcrypt hash ──────────────────────────────
echo "[3/6] Generating TLS cert and password hash..."
mkdir -p "${BUILD}/tls"
openssl req -x509 -nodes -days 3650 \
  -newkey rsa:2048 \
  -keyout "${BUILD}/tls/server.key" \
  -out "${BUILD}/tls/server.crt" \
  -subj "/CN=dcgm-exporter" 2>/dev/null

HASH=$(htpasswd -nbBC 10 "" "${AUTH_PASS}" | tr -d ':\n')

# ── 3. Build package tree ───────────────────────────────────────────
echo "[4/6] Creating package structure..."

# Binary
mkdir -p "${ROOT}/usr/bin"
cp "${SRC}/dcgm-exporter/cmd/dcgm-exporter/dcgm-exporter" "${ROOT}/usr/bin/"
chmod 755 "${ROOT}/usr/bin/dcgm-exporter"

# Config directory
mkdir -p "${ROOT}/etc/dcgm-exporter/tls"

# Metrics CSV
cp "${SRC}/dcgm-exporter/etc/default-counters.csv" "${ROOT}/etc/dcgm-exporter/"
[ -f "${SRC}/dcgm-exporter/etc/dcp-metrics-included.csv" ] && \
  cp "${SRC}/dcgm-exporter/etc/dcp-metrics-included.csv" "${ROOT}/etc/dcgm-exporter/"

# TLS certs (baked into deb)
cp "${BUILD}/tls/server.key" "${ROOT}/etc/dcgm-exporter/tls/"
cp "${BUILD}/tls/server.crt" "${ROOT}/etc/dcgm-exporter/tls/"
chmod 600 "${ROOT}/etc/dcgm-exporter/tls/server.key"

# web-config.yaml (auth baked into deb)
cat > "${ROOT}/etc/dcgm-exporter/web-config.yaml" << EOF
tls_server_config:
  cert_file: /etc/dcgm-exporter/tls/server.crt
  key_file: /etc/dcgm-exporter/tls/server.key

basic_auth_users:
  ${AUTH_USER}: ${HASH}
EOF
chmod 600 "${ROOT}/etc/dcgm-exporter/web-config.yaml"

# Environment file
mkdir -p "${ROOT}/etc/default"
cat > "${ROOT}/etc/default/dcgm-exporter" << EOF
ARGS="-a :${LISTEN_PORT} -f /etc/dcgm-exporter/default-counters.csv --web-config-file /etc/dcgm-exporter/web-config.yaml"
EOF

# Systemd service file
mkdir -p "${ROOT}/lib/systemd/system"
cat > "${ROOT}/lib/systemd/system/dcgm-exporter.service" << 'EOF'
[Unit]
Description=Prometheus exporter for NVIDIA GPU metrics (DCGM)
Documentation=https://github.com/NVIDIA/dcgm-exporter
After=nvidia-dcgm.service
Requires=nvidia-dcgm.service

[Service]
Restart=on-failure
User=root
EnvironmentFile=/etc/default/dcgm-exporter
ExecStart=/usr/bin/dcgm-exporter $ARGS
ExecReload=/bin/kill -HUP $MAINPID
TimeoutStopSec=20s
SendSIGKILL=no

[Install]
WantedBy=multi-user.target
EOF

# ── 4. DEBIAN control files ─────────────────────────────────────────
echo "[5/6] Creating DEBIAN control files..."
mkdir -p "${ROOT}/DEBIAN"

cat > "${ROOT}/DEBIAN/control" << CTRL
Package: ${PKG}
Version: ${VERSION}-${REVISION}
Section: utils
Priority: optional
Architecture: amd64
Depends: datacenter-gpu-manager-4 (>= 4.0.0) | datacenter-gpu-manager (>= 3.0.0)
Maintainer: GPU Infrastructure Team <gpu-infra@example.com>
Description: NVIDIA DCGM Exporter for Prometheus
 GPU metrics exporter using DCGM. Pre-configured with
 port ${LISTEN_PORT}, TLS, and basic auth.
CTRL

cat > "${ROOT}/DEBIAN/conffiles" << CONF
/etc/default/dcgm-exporter
/etc/dcgm-exporter/default-counters.csv
/etc/dcgm-exporter/web-config.yaml
/etc/dcgm-exporter/tls/server.crt
/etc/dcgm-exporter/tls/server.key
CONF

cat > "${ROOT}/DEBIAN/postinst" << 'POSTINST'
#!/bin/bash
set -e
systemctl daemon-reload
echo ""
echo "  dcgm-exporter installed."
echo "  Start:   sudo systemctl enable --now dcgm-exporter"
echo "  Verify:  curl -sk -u prometheus:<password> https://localhost:10101/metrics"
echo ""
POSTINST
chmod 755 "${ROOT}/DEBIAN/postinst"

cat > "${ROOT}/DEBIAN/prerm" << 'PRERM'
#!/bin/bash
set -e
systemctl is-active --quiet dcgm-exporter 2>/dev/null && systemctl stop dcgm-exporter || true
systemctl is-enabled --quiet dcgm-exporter 2>/dev/null && systemctl disable dcgm-exporter || true
PRERM
chmod 755 "${ROOT}/DEBIAN/prerm"

cat > "${ROOT}/DEBIAN/postrm" << 'POSTRM'
#!/bin/bash
set -e
systemctl daemon-reload
if [ "$1" = "purge" ]; then
  rm -rf /etc/dcgm-exporter
  rm -f /etc/default/dcgm-exporter
fi
POSTRM
chmod 755 "${ROOT}/DEBIAN/postrm"

# ── 5. Build .deb ───────────────────────────────────────────────────
echo "[6/6] Building .deb..."
dpkg-deb --build --root-owner-group "${ROOT}"
cp "${ROOT}.deb" "${BUILD}/../"

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  DONE: $(basename ${ROOT}).deb"
echo "══════════════════════════════════════════════════════════════"
```

### What the Build Script Does

1. Clones `dcgm-exporter` from GitHub and compiles the Go binary
2. Generates a self-signed TLS certificate (valid 10 years)
3. Generates a bcrypt password hash for basic auth
4. Packages everything into a `.deb` with this structure:

| Path | Purpose |
|------|---------|
| `/usr/bin/dcgm-exporter` | Binary |
| `/lib/systemd/system/dcgm-exporter.service` | Systemd unit |
| `/etc/default/dcgm-exporter` | ARGS (port, flags) |
| `/etc/dcgm-exporter/web-config.yaml` | TLS + basic auth |
| `/etc/dcgm-exporter/tls/server.{crt,key}` | TLS certificates |
| `/etc/dcgm-exporter/default-counters.csv` | Metrics to collect |

---

## 4. Pre-Install Script — What It Does

The `pre-install.sh` script runs on each GPU server before and during dcgm-exporter installation. Here's what each step does and why:

| Step | Action | Why |
|------|--------|-----|
| 1 | Auto-detect Ubuntu version | Uses `lsb_release` to set correct NVIDIA repo (`ubuntu2204`/`ubuntu2404`) |
| 2 | Check NVIDIA driver | Verifies `nvidia-smi` works — dcgm-exporter cannot run without it |
| 3 | Detect CUDA version | Reads from `nvidia-smi` output — handles both `CUDA Version:` and `CUDA UMD Version:` (Blackwell) |
| 4 | Remove broken repos | Removes old third-party repos that block `apt-get update` |
| 5 | Install CUDA keyring | Adds NVIDIA's apt repository — does NOT install or change CUDA |
| 6 | Remove old DCGM | Purges `datacenter-gpu-manager` v3 if present |
| 7 | Install DCGM v4 | Installs `datacenter-gpu-manager-4-cuda{12,13}` and starts `nvidia-dcgm` |
| 8 | Install deb + configure | Removes old service files, installs deb, strips profiling metrics, starts service |
| 9 | Verify | Checks port 10101, tests 401 without auth, tests 200 with auth |

### Pre-Install Script (`pre-install.sh`)

```bash
#!/usr/bin/env bash
###############################################################################
#  pre-install.sh — Full dcgm-exporter installer (auto-detects everything)
#
#  Safe to run on Ubuntu 20.04, 22.04, 24.04
#  Safe to run on CUDA 12 and CUDA 13 (Blackwell)
#  Does NOT change CUDA version, driver, or toolkit
#
#  Usage: sudo bash pre-install.sh
###############################################################################
ARCH="x86_64"                                  # x86_64 | sbsa (arm64)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEB_FILE="${SCRIPT_DIR}/dcgm-exporter_4.5.3-1_amd64.deb"
AUTH_USER="prometheus"
AUTH_PASS="dfgeiorltlm"
LISTEN_PORT="10101"
###############################################################################

set -euo pipefail

echo "══════════════════════════════════════════════════════════════"
echo "  dcgm-exporter installer — $(hostname)"
echo "══════════════════════════════════════════════════════════════"

# ── 1. Auto-detect Ubuntu version ───────────────────────────────────
echo ""
echo "[1/9] Detecting OS version..."
if ! command -v lsb_release &>/dev/null; then
  echo "  ✗ lsb_release not found. Is this Ubuntu?"
  exit 1
fi

OS_ID=$(lsb_release -is)
OS_VERSION=$(lsb_release -rs)

if [ "${OS_ID}" != "Ubuntu" ]; then
  echo "  ✗ This script is for Ubuntu only. Detected: ${OS_ID}"
  exit 1
fi

DISTRO="ubuntu$(echo ${OS_VERSION} | tr -d '.')"
echo "  ✓ Detected: ${OS_ID} ${OS_VERSION} → ${DISTRO}"

# ── 2. Check NVIDIA driver ──────────────────────────────────────────
echo ""
echo "[2/9] Checking NVIDIA driver..."
if ! command -v nvidia-smi &>/dev/null; then
  echo "  ✗ nvidia-smi not found. Install NVIDIA driver first."
  exit 1
fi

nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo "  ✓ NVIDIA driver found"

# ── 3. Detect CUDA version (handles both old and Blackwell format) ──
echo ""
echo "[3/9] Detecting CUDA version..."
CUDA_VERSION=$(nvidia-smi | grep -oE "CUDA (UMD )?Version: [0-9]+" | grep -oE "[0-9]+" | head -1)

if [ -z "${CUDA_VERSION}" ]; then
  echo "  ✗ Could not detect CUDA version from nvidia-smi"
  exit 1
fi

echo "  ✓ CUDA major version: ${CUDA_VERSION}"

# ── 4. Remove broken third-party repos ──────────────────────────────
echo ""
echo "[4/9] Cleaning broken apt repos..."
rm -f /etc/apt/sources.list.d/nvidia-container*.list
rm -f /etc/apt/sources.list.d/nvidia-docker*.list
rm -f /etc/apt/sources.list.d/libnvidia-container*.list
rm -f /etc/apt/sources.list.d/helm*.list
rm -f /etc/apt/sources.list.d/influx*.list
rm -f /etc/apt/sources.list.d/ookla*.list
rm -f /etc/apt/sources.list.d/plakar*.list
rm -f /etc/apt/sources.list.d/hpe*.list
rm -f /etc/apt/sources.list.d/debian*.list
sed -i '/deb.debian.org.*unstable/d' /etc/apt/sources.list 2>/dev/null || true
echo "  ✓ Cleaned"

# ── 5. Install CUDA keyring ─────────────────────────────────────────
echo ""
echo "[5/9] Setting up NVIDIA apt repository..."
KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/${DISTRO}/${ARCH}/cuda-keyring_1.1-1_all.deb"
echo "  Repo: ${DISTRO}/${ARCH}"

wget -q -O /tmp/cuda-keyring.deb "${KEYRING_URL}"
dpkg -i /tmp/cuda-keyring.deb
rm -f /tmp/cuda-keyring.deb
apt-get update 2>/dev/null || true

echo "  ✓ NVIDIA repository configured"

# ── 6. Remove old DCGM ──────────────────────────────────────────────
echo ""
echo "[6/9] Removing old DCGM packages (if any)..."
dpkg --list datacenter-gpu-manager &>/dev/null && apt purge --yes datacenter-gpu-manager || true
dpkg --list datacenter-gpu-manager-config &>/dev/null && apt purge --yes datacenter-gpu-manager-config || true

echo "  ✓ Old packages cleaned"

# ── 7. Install DCGM host engine ─────────────────────────────────────
echo ""
echo "[7/9] Installing datacenter-gpu-manager-4-cuda${CUDA_VERSION}..."
apt-get install --yes --install-recommends datacenter-gpu-manager-4-cuda${CUDA_VERSION}

systemctl --now enable nvidia-dcgm
sleep 2

if systemctl is-active --quiet nvidia-dcgm; then
  echo "  ✓ nvidia-dcgm running"
else
  echo "  ✗ nvidia-dcgm failed to start"
  journalctl -u nvidia-dcgm --no-pager -n 10
  exit 1
fi

dcgmi discovery -l

# ── 8. Remove old service + install deb + strip profiling ────────────
echo ""
echo "[8/9] Installing dcgm-exporter deb..."

# Remove old service file if exists
if [ -f /etc/systemd/system/dcgm-exporter.service ]; then
  echo "  Removing old service file..."
  systemctl stop dcgm-exporter 2>/dev/null || true
  systemctl disable dcgm-exporter 2>/dev/null || true
  rm -f /etc/systemd/system/dcgm-exporter.service
  systemctl daemon-reload
fi

# Remove old Docker-based exporter if running
docker rm -f nvidia-dcgm-exporter 2>/dev/null || true

if [ ! -f "${DEB_FILE}" ]; then
  echo "  ✗ deb file not found: ${DEB_FILE}"
  exit 1
fi

dpkg -i "${DEB_FILE}"
systemctl daemon-reload

# Strip profiling metrics to prevent crash
echo "  Stripping profiling metrics..."
cp /etc/dcgm-exporter/default-counters.csv /etc/dcgm-exporter/no-profiling-counters.csv
sed -i '/DCGM_FI_PROF_/d' /etc/dcgm-exporter/no-profiling-counters.csv
sed -i 's|default-counters.csv|no-profiling-counters.csv|' /etc/default/dcgm-exporter

# Restart DCGM for clean state
systemctl restart nvidia-dcgm
sleep 2

# Start dcgm-exporter
systemctl enable --now dcgm-exporter

# Kill slow lshw if hanging
sleep 5
pkill lshw 2>/dev/null || true
sleep 10

if systemctl is-active --quiet dcgm-exporter; then
  echo "  ✓ dcgm-exporter running"
else
  echo "  ✗ dcgm-exporter failed to start"
  journalctl -u dcgm-exporter --no-pager -n 10
  exit 1
fi

# ── 9. Verify ────────────────────────────────────────────────────────
echo ""
echo "[9/9] Verifying..."

echo ""
echo "  Port check:"
ss -tlnp | grep ${LISTEN_PORT} && echo "  ✓ Listening on ${LISTEN_PORT}" || echo "  ✗ Not listening"

echo ""
echo "  Auth check (no creds → 401):"
HTTP_CODE=$(curl -sk -o /dev/null -w "%{http_code}" https://localhost:${LISTEN_PORT}/metrics)
if [ "${HTTP_CODE}" = "401" ]; then
  echo "  ✓ Got 401 without credentials — auth is working"
else
  echo "  ⚠ Expected 401, got ${HTTP_CODE}"
fi

echo ""
echo "  Metrics check (with creds → 200):"
HTTP_CODE=$(curl -sk -o /dev/null -w "%{http_code}" -u ${AUTH_USER}:${AUTH_PASS} https://localhost:${LISTEN_PORT}/metrics)
if [ "${HTTP_CODE}" = "200" ]; then
  echo "  ✓ Got 200 with credentials — metrics flowing"
  echo ""
  echo "  Sample output:"
  curl -sk -u ${AUTH_USER}:${AUTH_PASS} https://localhost:${LISTEN_PORT}/metrics 2>/dev/null | grep "DCGM_FI" | head -10
else
  echo "  ✗ Expected 200, got ${HTTP_CODE}"
fi

echo ""
echo "══════════════════════════════════════════════════════════════"
echo "  ✓ $(hostname) — DONE"
echo "  OS:       ${OS_ID} ${OS_VERSION}"
echo "  CUDA:     ${CUDA_VERSION}"
echo "  Port:     ${LISTEN_PORT}"
echo "  Service:  systemctl status dcgm-exporter"
echo "  Metrics:  curl -sk -u ${AUTH_USER}:<pass> https://$(hostname):${LISTEN_PORT}/metrics"
echo "  Config:   /etc/default/dcgm-exporter"
echo "══════════════════════════════════════════════════════════════"
```

---

## 5. Step-by-Step Manual Installation

If you prefer to run commands manually instead of using the script:

### Step 1 — Check NVIDIA driver

```bash
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
```

### Step 2 — Detect CUDA version

```bash
CUDA_VERSION=$(nvidia-smi | grep -oE "CUDA (UMD )?Version: [0-9]+" | grep -oE "[0-9]+" | head -1)
echo "CUDA version: ${CUDA_VERSION}"
```

### Step 3 — Remove broken third-party repos

```bash
sudo rm -f /etc/apt/sources.list.d/nvidia-container*.list
sudo rm -f /etc/apt/sources.list.d/nvidia-docker*.list
sudo rm -f /etc/apt/sources.list.d/libnvidia-container*.list
sudo rm -f /etc/apt/sources.list.d/helm*.list
sudo rm -f /etc/apt/sources.list.d/influx*.list
sudo rm -f /etc/apt/sources.list.d/ookla*.list
sudo rm -f /etc/apt/sources.list.d/plakar*.list
sudo rm -f /etc/apt/sources.list.d/hpe*.list
sudo rm -f /etc/apt/sources.list.d/debian*.list
sudo sed -i '/deb.debian.org.*unstable/d' /etc/apt/sources.list 2>/dev/null || true
```

### Step 4 — Install CUDA keyring

```bash
wget -q -O /tmp/cuda-keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i /tmp/cuda-keyring.deb
sudo rm -f /tmp/cuda-keyring.deb
sudo apt-get update 2>/dev/null || true
```

> For Ubuntu 24.04, replace `ubuntu2204` with `ubuntu2404`.

### Step 5 — Remove old DCGM

```bash
sudo dpkg --list datacenter-gpu-manager &>/dev/null && sudo apt purge --yes datacenter-gpu-manager || true
sudo dpkg --list datacenter-gpu-manager-config &>/dev/null && sudo apt purge --yes datacenter-gpu-manager-config || true
```

### Step 6 — Install DCGM host engine

```bash
sudo apt-get install --yes --install-recommends datacenter-gpu-manager-4-cuda${CUDA_VERSION}
sudo systemctl --now enable nvidia-dcgm
sleep 2
dcgmi discovery -l
```

### Step 7 — Remove old dcgm-exporter service file

```bash
sudo systemctl stop dcgm-exporter 2>/dev/null || true
sudo systemctl disable dcgm-exporter 2>/dev/null || true
sudo rm -f /etc/systemd/system/dcgm-exporter.service
sudo docker rm -f nvidia-dcgm-exporter 2>/dev/null || true
sudo systemctl daemon-reload
```

### Step 8 — Install the deb

```bash
sudo dpkg -i dcgm-exporter_4.5.3-1_amd64.deb
sudo systemctl daemon-reload
```

### Step 9 — Strip profiling metrics

```bash
sudo cp /etc/dcgm-exporter/default-counters.csv /etc/dcgm-exporter/no-profiling-counters.csv
sudo sed -i '/DCGM_FI_PROF_/d' /etc/dcgm-exporter/no-profiling-counters.csv
sudo sed -i 's|default-counters.csv|no-profiling-counters.csv|' /etc/default/dcgm-exporter
cat /etc/default/dcgm-exporter
```

### Step 10 — Restart DCGM host engine

```bash
sudo systemctl restart nvidia-dcgm
sleep 2
```

### Step 11 — Start dcgm-exporter

```bash
sudo systemctl enable --now dcgm-exporter
```

### Step 12 — Kill slow lshw if hanging

```bash
sleep 5
sudo pkill lshw 2>/dev/null || true
sleep 10
```

### Step 13 — Verify port

```bash
sudo ss -tlnp | grep 10101
```

### Step 14 — Test without auth (should get 401)

```bash
curl -sk -o /dev/null -w "HTTP %{http_code}\n" https://localhost:10101/metrics
```

### Step 15 — Test with auth (should get 200 + metrics)

```bash
curl -sk -u prometheus:eorijldfglm https://localhost:10101/metrics | head -10
```

---

## 6. Post-Install: Change Password

```bash
# 1. Generate new bcrypt hash
NEW_HASH=$(htpasswd -nbBC 10 "" "YourNewPassword" | tr -d ':\n')

# 2. Replace old hash
sudo sed -i "s|prometheus:.*|prometheus: ${NEW_HASH}|" /etc/dcgm-exporter/web-config.yaml

# 3. Restart
sudo systemctl restart dcgm-exporter

# 4. Verify
curl -sk -u prometheus:YourNewPassword https://localhost:10101/metrics | head -5
```

---

## 7. Scripts Reference

### Files to Copy to Each Server

```
dcgm-exporter/
├── dcgm-exporter_4.5.3-1_amd64.deb    # Pre-built deb package
└── pre-install.sh                       # Installer script
```

### Quick Health Check (Already Installed Servers)

```bash
sudo ss -tlnp | grep 10101 && \
curl -sk -o /dev/null -w "HTTP %{http_code}" -u prometheus:kdhgksdhfglk https://localhost:10101/metrics
```

If it returns `HTTP 200` — that server is healthy.

### Prometheus Scrape Config

```yaml
scrape_configs:
  - job_name: 'dcgm-exporter'
    scheme: https
    tls_config:
      insecure_skip_verify: true
    basic_auth:
      username: prometheus
      password: sdgerjlergj
    static_configs:
      - targets: ['gpu-node:10101']
```

## DCGM-Exporter with docker-compose

docker-compose.yaml
```
version: "3.8"

services:
  prometheus:
    image: prom/prometheus:latest
    container_name: prometheus
    hostname: prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.retention.time=90d"   # <- keep 90 days
      - "--storage.tsdb.retention.size=50GB"  # optional limit
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - ./alerts/:/etc/prometheus/alerts/
      - prometheus-data:/prometheus
    ports:
      - "9090:9090"
    restart: unless-stopped
    networks:
      - monitoring
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  dcgm-exporter:
    image: nvcr.io/nvidia/k8s/dcgm-exporter:4.4.1-4.5.2-ubuntu22.04
    container_name: dcgm-exporter
    hostname: xxxx
    ports:
      - "9401:9400"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    cap_add:
      - SYS_ADMIN
    restart: unless-stopped
    networks:
      - monitoring
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  cadvisor:
    image: gcr.io/cadvisor/cadvisor:latest
    container_name: cadvisor
    hostname: cadvisor
    ports:
      - "8080:8080"
    restart: unless-stopped
    volumes:
      - /:/rootfs:ro
      - /var/run:/var/run:ro
      - /sys:/sys:ro
      - /var/lib/docker/:/var/lib/docker:ro
      - /dev/disk/:/dev/disk:ro
    devices:
      - /dev/kmsg
    privileged: true
    networks:
      - monitoring
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"

  node-exporter:
    image: prom/node-exporter:latest
    container_name: node-exporter
    restart: unless-stopped
    ports:
      - "9100:9100"  # Exposes Node Exporter on port 9100
    networks:
      - monitoring
    volumes:
      - /proc:/host/proc:ro
      - /sys:/host/sys:ro
      - /:/host/root:ro


networks:
  monitoring:

volumes:
  prometheus-data:
```

prometheus.yaml
```
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: "gpu_exporter"
    static_configs:
      - targets:
          - "10.xx.xxx.xxx:9401"
          - "10.xx.xxx.xxx:9401"

  - job_name: "node_exporter"
    static_configs:
      - targets:
          - "10.xx.xxx.xxx:9100"
          - "10.xx.xx.xx:9100"

  - job_name: 'cadvisor'
    scrape_interval: 30s
    scrape_timeout: 30s
    static_configs:
      - targets:
          - '10.xx.xx.xx:8080'
          - '10.xx.xx.xx:8080'
```