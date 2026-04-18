#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Dev Shell (single entry point for Docker development)
#
# Usage:  bash scripts/docker/dev-shell.sh [--rebuild] [--audio] [--webcam] [--screen]
#
# Handles everything: installs Docker CE if missing, removes snap Docker,
# installs nvidia-container-toolkit, configures GPU access, stages pre-built
# libs, builds the dev image, cleans disk, starts the container, and drops
# you into a shell.
#
# Flags:
#   --rebuild   Force-rebuild the dev image (after Dockerfile changes)
#   --audio     Enable PulseAudio mic passthrough (WSLg)
#   --webcam    Enable /dev/video0 webcam passthrough
#   --screen    Launch oe_screen_capture.exe on Windows (DXGI desktop stream)
#   --clean     Clean Docker resources + show VHDX compaction instructions
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

IMAGE="omniedge-dev:latest"
COMPOSE_FILES="-f docker-compose.yaml -f docker-compose.dev.yaml"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

REBUILD=false       # --rebuild: force Docker image rebuild from scratch
ENABLE_AUDIO=false  # --audio:   pass through host audio devices to container
ENABLE_WEBCAM=false # --webcam:  map /dev/video0 into container via compose override
ENABLE_SCREEN=false # --screen:  enable Windows-side screen capture bridge
CLEAN_DISK=false    # --clean:   prune Docker caches and reclaim WSL2 VHDX space
for arg in "$@"; do
    case "$arg" in
        --rebuild) REBUILD=true ;;
        --audio)   ENABLE_AUDIO=true ;;
        --webcam)  ENABLE_WEBCAM=true ;;
        --screen)  ENABLE_SCREEN=true ;;
        --clean)   CLEAN_DISK=true ;;
        *) echo -e "${RED}Unknown flag: $arg${NC}"; exit 1 ;;
    esac
done

# Reconstruct command for restart hints
RESTART_CMD="bash scripts/docker/dev-shell.sh"
$ENABLE_AUDIO  && RESTART_CMD+=" --audio"
$ENABLE_WEBCAM && RESTART_CMD+=" --webcam"
$ENABLE_SCREEN && RESTART_CMD+=" --screen"

# =============================================================================
#  Memory monitoring — prevent WSL2 OOM crashes
# =============================================================================
# Read free RAM in MB from /proc/meminfo (Linux kernel memory stats file).
# Parses the "MemAvailable" line, extracts the kB value, converts to MB.
get_avail_mem_mb() {
    awk '/MemAvailable/ { printf "%d", $2/1024 }' /proc/meminfo 2>/dev/null || echo 0
}

# Read total installed RAM in MB from /proc/meminfo.
# Parses the "MemTotal" line — this is a fixed value, unlike MemAvailable.
get_total_mem_mb() {
    awk '/MemTotal/ { printf "%d", $2/1024 }' /proc/meminfo 2>/dev/null || echo 0
}

# Check memory and abort/warn before heavy operations
# Check if enough RAM is available before a heavy operation (build, docker up, etc).
#   $1 = operation name (for log messages, e.g. "docker build")
#   $2 = minimum MB desired (default 4096 = 4 GB)
#
# Three outcomes:
#   - Free RAM >= min_mb       → OK, just print status
#   - Free RAM < min_mb        → WARNING, try to reclaim filesystem caches
#   - Free RAM < 2 GB          → CRITICAL, prune Docker + drop caches, abort if still low
check_memory() {
    local operation="$1"
    local min_mb="${2:-4096}"
    local avail_mb total_mb used_mb
    avail_mb=$(get_avail_mem_mb)
    total_mb=$(get_total_mem_mb)
    used_mb=$(( total_mb - avail_mb ))

    echo -e "  Memory: ${avail_mb} MB free / ${total_mb} MB total (${used_mb} MB used)"

    # --- CRITICAL: under 2 GB free — WSL2 OOM-kills processes at this level ---
    if (( avail_mb < 2048 )); then
        echo -e "${RED}CRITICAL: Only ${avail_mb} MB RAM available — WSL2 will crash!${NC}"
        echo "  Freeing Docker memory..."

        # Remove stopped containers, unused images, build cache
        docker system prune -f 2>/dev/null || true
        # Flush filesystem caches back to disk, then tell kernel to free cached RAM
        sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true

        # Re-check after cleanup
        avail_mb=$(get_avail_mem_mb)
        if (( avail_mb < 2048 )); then
            echo -e "${RED}ABORT: Still only ${avail_mb} MB free after cleanup. Close other apps and retry.${NC}"
            exit 1
        fi
        echo -e "  ${GREEN}Recovered to ${avail_mb} MB free.${NC}"

    # --- WARNING: below desired threshold — try light cleanup only ---
    elif (( avail_mb < min_mb )); then
        echo -e "${YELLOW}WARNING: ${avail_mb} MB free before ${operation} (want ${min_mb} MB).${NC}"
        echo "  Dropping filesystem caches..."
        # Flush + free cached RAM (no Docker prune — not critical)
        sync && echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null 2>&1 || true
        avail_mb=$(get_avail_mem_mb)
        echo -e "  ${GREEN}Now ${avail_mb} MB free.${NC}"
    fi
}


# =============================================================================
#  Docker CE Installation
# =============================================================================
install_docker_ce() {
    echo -e "${YELLOW}Installing Docker CE...${NC}"

    # Remove snap Docker if present (breaks GPU passthrough on WSL2)
    if snap list docker &>/dev/null; then
        echo -e "${YELLOW}Removing snap Docker (incompatible with GPU passthrough)...${NC}"
        sudo snap remove --purge docker
        hash -r 2>/dev/null || true
    fi

    # Fix: Ubuntu Snap can leave stale /snap/docker/... paths in NVIDIA's
    # container runtime config. If present, GPU passthrough silently fails
    # ("docker run --gpus all" starts but container can't see the GPU).
    # This strips any leftover /snap/ references from the config file.
    local nvidia_runtime_cfg="/etc/nvidia-container-runtime/config.toml"
    if [ -f "$nvidia_runtime_cfg" ] && grep -q '/snap/' "$nvidia_runtime_cfg"; then
        echo "  Removing stale Snap references from NVIDIA container runtime config..."
        sudo sed -i 's|"/snap/[^"]*", ||g' "$nvidia_runtime_cfg"
    fi

    # Refresh package index silently (-qq = errors only)
    sudo apt-get update -qq
    sudo apt-get install -y -qq ca-certificates curl
    sudo install -m 0755 -d /etc/apt/keyrings

    # Download Docker's official GPG signing key so apt can verify packages.
    # URL is Docker's stable endpoint — same key re-served on rotation.
    # -f fail on HTTP error, -sS silent+show errors, -L follow redirects.
    sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        -o /etc/apt/keyrings/docker.asc
    # Ensure all users (including apt's _apt user) can read the key
    sudo chmod a+r /etc/apt/keyrings/docker.asc

    # Register Docker's apt repository, pinned to this distro's codename.
    # VERSION_CODENAME is read from /etc/os-release at runtime (e.g. "noble"
    # for Ubuntu 24.04), so this works on any Ubuntu version without changes.
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] \
https://download.docker.com/linux/ubuntu \
$(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list >/dev/null

    # Re-fetch now that Docker's repo is registered, then install
    sudo apt-get update -qq
    sudo apt-get install -y -qq \
        docker-ce docker-ce-cli containerd.io \
        docker-buildx-plugin docker-compose-plugin

    sudo usermod -aG docker "$USER"
    sudo systemctl enable --now docker
    echo -e "${GREEN}Docker CE installed.${NC}"
}

# =============================================================================
#  nvidia-container-toolkit
# =============================================================================
install_nvidia_toolkit() {
    echo -e "${YELLOW}Installing nvidia-container-toolkit...${NC}"

    # NVIDIA Container Toolkit APT repository setup
    # -----------------------------------------------
    # Source: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html
    # Hosted on GitHub Pages (nvidia/libnvidia-container repo).
    #
    # If these URLs break in the future:
    #   1. Check NVIDIA's install guide above for updated URLs
    #   2. Verify the GPG key URL:   curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | head -1
    #   3. Verify the repo list URL: curl -sI  https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list
    #   4. Fallback: install manually via https://github.com/NVIDIA/nvidia-container-toolkit/releases
    #
    # Last verified working: 2026-04-13

    local GPG_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
    local REPO_LIST_URL="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
    local KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"

    if ! curl -fsSL "$GPG_URL" | sudo gpg --dearmor -o "$KEYRING" 2>/dev/null; then
        echo -e "${RED}ERROR: Failed to fetch NVIDIA GPG key from $GPG_URL${NC}"
        echo "  Check https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html for updated URLs"
        return 1
    fi
    if ! curl -fsSL "$REPO_LIST_URL" \
        | sed "s#deb https://#deb [signed-by=${KEYRING}] https://#g" \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null; then
        echo -e "${RED}ERROR: Failed to fetch NVIDIA repo list from $REPO_LIST_URL${NC}"
        echo "  Check https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html for updated URLs"
        return 1
    fi
    sudo apt-get update -qq
    sudo apt-get install -y -qq nvidia-container-toolkit

    # Register NVIDIA runtime with Docker
    # -----------------------------------------------
    # nvidia-ctk adds "nvidia" to /etc/docker/daemon.json runtimes{} but does NOT
    # set it as the default. We need default-runtime = nvidia because:
    #   - docker-compose doesn't support --runtime flag per-service on WSL2
    #   - CDI device injection (--gpus) is unreliable under WSL2's GPU passthrough
    # By making nvidia the default, ALL containers get GPU access without extra flags.
    sudo nvidia-ctk runtime configure --runtime=docker

    # Set nvidia as the default Docker runtime (idempotent — skips if already set)
    local DAEMON_JSON="/etc/docker/daemon.json"
    if [ -f "$DAEMON_JSON" ] && ! grep -q '"default-runtime"' "$DAEMON_JSON"; then
        sudo python3 -c "
import json, sys

daemon_json_path = '$DAEMON_JSON'

with open(daemon_json_path) as f:
    config = json.load(f)

config['default-runtime'] = 'nvidia'

with open(daemon_json_path, 'w') as f:
    json.dump(config, f, indent=4)

print(f'Set default-runtime=nvidia in {daemon_json_path}')
"
    fi

    # WSL2 runtime fixups
    # -----------------------------------------------
    # The NVIDIA container runtime config needs two patches on WSL2:
    #
    # 1) Disable cgroups — WSL2 doesn't expose GPU cgroup controllers.
    #    Without this, containers fail at startup with:
    #      "could not create cgroup: no such file or directory"
    #    We detect WSL2 by checking for /usr/lib/wsl/lib (the WSL GPU driver shim).
    #
    # 2) Remove snap paths — Ubuntu snap installs can inject snap-packaged
    #    nvidia binaries (e.g. /snap/docker/...) into the runtime search path.
    #    These snap binaries are incompatible with the native toolkit and cause
    #    "exec format error" or version mismatches. Stripping them is safe
    #    because we installed the toolkit natively via apt above.
    local NCR_CFG="/etc/nvidia-container-runtime/config.toml"
    if [ -f "$NCR_CFG" ]; then
        # Fix 1: disable cgroups on WSL2
        if [ -d /usr/lib/wsl/lib ]; then
            sudo sed -i 's|#no-cgroups = false|no-cgroups = true|' "$NCR_CFG"
        fi

        # Fix 2: remove snap binary paths that conflict with native toolkit
        if grep -q '/snap/' "$NCR_CFG"; then
            sudo sed -i 's|"/snap/[^"]*", ||g' "$NCR_CFG"
        fi
    fi

    echo -e "${GREEN}nvidia-container-toolkit installed and configured.${NC}"
    # NOTE: caller must restart Docker to pick up daemon.json / config.toml changes
}

# =============================================================================
#  Ensure Docker + GPU are ready
# =============================================================================
ensure_docker_ready() {
    # Docker CE (not snap)
    local need_install=false
    if ! command -v docker &>/dev/null; then
        need_install=true
    elif snap list docker &>/dev/null; then
        echo -e "${YELLOW}Snap Docker detected — replacing with Docker CE.${NC}"
        need_install=true
    fi
    if $need_install; then
        install_docker_ce
    fi

    # Grant Docker access without requiring a logout/login cycle.
    # `docker info` fails when the user lacks permission to talk to the daemon.
    # `sg docker` re-execs the script in a sub-shell that has the docker group,
    # which is the standard workaround for picking up a new group mid-session.
    if ! docker info &>/dev/null; then
        if ! id -nG "$USER" | grep -qw docker; then
            sudo usermod -aG docker "$USER"
            echo -e "${YELLOW}Added $USER to docker group.${NC}"
        fi
        echo -e "${YELLOW}Re-launching script with docker group privileges...${NC}"
        exec sg docker "$0 $*"
    fi

    # nvidia-container-toolkit
    local docker_config_changed=false
    if ! dpkg -l nvidia-container-toolkit &>/dev/null 2>&1; then
        install_nvidia_toolkit
        docker_config_changed=true
    fi

    # Docker uses "runc" (CPU-only) by default. Setting "nvidia" as the default
    # runtime gives every container automatic GPU access. This is a safety net for
    # when the toolkit package is already installed but daemon.json lacks the key.
    local daemon_json="/etc/docker/daemon.json"
    if ! grep -q '"default-runtime"' "$daemon_json" 2>/dev/null; then
        echo -e "${YELLOW}Setting nvidia as default Docker runtime...${NC}"
        sudo python3 -c "
import json, pathlib
p = pathlib.Path('$daemon_json')
cfg = json.loads(p.read_text())
cfg['default-runtime'] = 'nvidia'
p.write_text(json.dumps(cfg, indent=4))
"
        docker_config_changed=true
    fi

    # Single restart point — only if we actually changed Docker config
    if $docker_config_changed; then
        sudo systemctl restart docker
        sleep 2  # dockerd needs a moment to re-initialize
    fi

    # Verify GPU access in Docker
    echo "Verifying GPU access in Docker..."
    if ! docker run --rm --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all \
            nvcr.io/nvidia/pytorch:25.12-py3 nvidia-smi; then
        echo -e "${RED}ERROR: GPU not accessible in Docker.${NC}"
        echo "  Verify your NVIDIA driver: nvidia-smi"
        exit 1
    fi
    echo -e "${GREEN}Docker + GPU ready.${NC}"
}

# =============================================================================
#  Stage pre-built libs (avoids Rust toolchain in Docker)
# =============================================================================
stage_libs() {
    local LIBS_STAGING="$PROJECT_ROOT/.docker_libs"

    # Skip if already staged
    if [ -f "$LIBS_STAGING/libtokenizers_c.a" ] && \
       [ -f "$LIBS_STAGING/libusockets.a" ]; then
        echo -e "${GREEN}Pre-built libs already staged.${NC}"
        return 0
    fi

    echo "Staging pre-built tokenizers/sentencepiece/uWebSockets libs..."
    rm -rf "$LIBS_STAGING"
    mkdir -p "$LIBS_STAGING"

    # Static libraries
    for f in libtokenizers_c.a libtokenizers_cpp.a libsentencepiece.a; do
        if [ -f "/usr/local/lib/$f" ]; then
            cp "/usr/local/lib/$f" "$LIBS_STAGING/$f"
            echo -e "  ${GREEN}OK${NC}  $f"
        else
            echo -e "  ${RED}MISSING${NC}  /usr/local/lib/$f"
            echo "  Run: cd /tmp && git clone --recurse-submodules https://github.com/mlc-ai/tokenizers-cpp.git && cd tokenizers-cpp && mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local && make -j\$(nproc) && sudo make install"
            exit 1
        fi
    done

    # Headers
    for f in tokenizers_c.h tokenizers_cpp.h; do
        if [ -f "/usr/local/include/$f" ]; then
            cp "/usr/local/include/$f" "$LIBS_STAGING/$f"
            echo -e "  ${GREEN}OK${NC}  $f"
        else
            echo -e "  ${RED}MISSING${NC}  /usr/local/include/$f"
            exit 1
        fi
    done

    # uSockets static library
    if [ -f "/usr/local/lib/libusockets.a" ]; then
        cp "/usr/local/lib/libusockets.a" "$LIBS_STAGING/libusockets.a"
        echo -e "  ${GREEN}OK${NC}  libusockets.a"
    else
        echo -e "  ${RED}MISSING${NC}  /usr/local/lib/libusockets.a — build uSockets first"
        exit 1
    fi

    # uWebSockets headers (header-only)
    mkdir -p "$LIBS_STAGING/uws_headers"
    for h in /usr/local/include/*.h; do
        base="$(basename "$h")"
        case "$base" in
            tokenizers_c.h|tokenizers_cpp.h|intypedef.h|herror.h) continue ;;
        esac
        cp "$h" "$LIBS_STAGING/uws_headers/$base"
    done
    echo -e "  ${GREEN}OK${NC}  $(ls "$LIBS_STAGING/uws_headers/" | wc -l) uWebSockets/uSockets headers"
}

# =============================================================================
#  Docker cleanup: free disk to prevent OOM
# =============================================================================
cleanup_docker() {
    echo "Cleaning unused Docker resources..."

    # Stop and remove containers not from our dev compose
    local dev_container
    dev_container=$(docker compose $COMPOSE_FILES ps -q 2>/dev/null || true)
    for cid in $(docker ps -aq 2>/dev/null); do
        [[ "$cid" == "$dev_container" ]] && continue
        echo "  Removing stale container $(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null)..."
        docker rm -f "$cid" 2>/dev/null || true
    done

    # Remove all images except our dev image and its base
    for img in $(docker images --format '{{.Repository}}:{{.Tag}}' 2>/dev/null); do
        [[ "$img" == "$IMAGE" ]] && continue
        [[ "$img" == nvidia/cuda:* ]] && continue
        [[ "$img" == nvcr.io/nvidia/* ]] && continue
        [[ "$img" == "<none>:<none>" ]] && continue
        echo "  Removing image $img..."
        docker rmi "$img" 2>/dev/null || true
    done

    # Purge dangling images and unused volumes (preserve build cache for fast rebuilds)
    docker image prune -f 2>/dev/null || true
    docker volume prune -f 2>/dev/null || true

    echo -e "  ${GREEN}Done.${NC} $(docker system df --format '{{.Type}}: {{.Size}}' 2>/dev/null | tr '\n' ', ')"
}

# =============================================================================
#  Host disk check — WSL2 VHDX bloats on Windows but never auto-shrinks
# =============================================================================
get_windows_free_gb() {
    # Returns free GB on the Windows C: drive, or empty string if not on WSL2
    if command -v powershell.exe &>/dev/null; then
        powershell.exe 'Write-Host ([math]::Floor((Get-PSDrive C).Free/1GB))' 2>/dev/null | tr -d '\r'
    fi
}

get_vhdx_size_gb() {
    # Returns the WSL2 ext4.vhdx size in GB
    if command -v powershell.exe &>/dev/null; then
        powershell.exe '
            $vhdx = Get-ChildItem "$env:LOCALAPPDATA\Packages" -Filter ext4.vhdx -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($vhdx) { Write-Host ([math]::Floor($vhdx.Length/1GB)) }
        ' 2>/dev/null | tr -d '\r'
    fi
}

check_windows_disk() {
    local win_free_gb vhdx_gb wsl_used_gb
    win_free_gb=$(get_windows_free_gb)
    [[ -z "$win_free_gb" ]] && return 0  # not WSL2, skip

    vhdx_gb=$(get_vhdx_size_gb)
    wsl_used_gb=$(( $(df --output=used / 2>/dev/null | tail -1 | tr -d ' ') / 1024 / 1024 )) 2>/dev/null || wsl_used_gb=0
    local wasted_gb=0
    [[ -n "$vhdx_gb" ]] && (( vhdx_gb > wsl_used_gb )) && wasted_gb=$(( vhdx_gb - wsl_used_gb ))

    if (( win_free_gb < 10 )); then
        echo -e "${RED}CRITICAL: Windows C: drive has only ${win_free_gb} GB free!${NC}"
        [[ -n "$vhdx_gb" ]] && echo -e "  WSL2 VHDX file: ${vhdx_gb} GB on disk (WSL uses ${wsl_used_gb} GB — ~${wasted_gb} GB reclaimable)"
        echo -e "  ${YELLOW}To reclaim space:${NC}"
        echo "    1. Run:  bash $0 --clean"
        echo "    2. Then in Windows Admin PowerShell:"
        echo "       wsl --shutdown"
        echo "       diskpart"
        echo "       select vdisk file=\"\$env:LOCALAPPDATA\\Packages\\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\\LocalState\\ext4.vhdx\""
        echo "       compact vdisk"
        echo "       detach vdisk"
        echo "       exit"
        echo -e "  ${GREEN}Why:${NC} WSL2's VHDX is grow-only — docker churn bloats the file;"
        echo "        Windows sees the file size, WSL sees ext4 used → the numbers diverge."
        echo -e "  ${GREEN}Permanent fix:${NC} add '[experimental]\\nsparseVhd=true' to %USERPROFILE%\\.wslconfig,"
        echo "        then: wsl --shutdown && wsl --manage Ubuntu --set-sparse true"
        echo "        (after that, 'sudo fstrim -av' inside WSL is all you need)."
        exit 1
    elif (( win_free_gb < 30 )); then
        echo -e "${YELLOW}WARNING: Windows C: drive has only ${win_free_gb} GB free.${NC}"
        [[ -n "$vhdx_gb" && wasted_gb -gt 10 ]] && echo -e "  VHDX bloat: ~${wasted_gb} GB reclaimable via 'wsl --shutdown && diskpart compact'"
    fi
}

# =============================================================================
#  Full disk cleanup — Docker + temp files + VHDX advisory
# =============================================================================
clean_all() {
    echo -e "${BOLD}=== Disk Cleanup ===${NC}"
    ensure_docker_ready

    # 1. Docker resources
    cleanup_docker

    # 2. Temp/stale files in project
    echo "Removing stale temp files..."
    rm -f "$PROJECT_ROOT"/docker-compose.yaml.tmp.* 2>/dev/null
    echo -e "  ${GREEN}Done.${NC}"

    # 3. Report and advise on VHDX
    local vhdx_gb wsl_used_gb win_free_gb
    vhdx_gb=$(get_vhdx_size_gb)
    win_free_gb=$(get_windows_free_gb)
    wsl_used_gb=$(( $(df --output=used / 2>/dev/null | tail -1 | tr -d ' ') / 1024 / 1024 )) 2>/dev/null || wsl_used_gb=0

    echo ""
    echo -e "${BOLD}=== Disk Status ===${NC}"
    echo "  WSL2 used:        ${wsl_used_gb} GB"
    [[ -n "$vhdx_gb" ]] && echo "  VHDX on Windows:  ${vhdx_gb} GB (bloat: ~$(( vhdx_gb - wsl_used_gb )) GB)"
    [[ -n "$win_free_gb" ]] && echo "  Windows C: free:  ${win_free_gb} GB"

    if [[ -n "$vhdx_gb" ]] && (( vhdx_gb - wsl_used_gb > 10 )); then
        echo ""
        echo -e "${BOLD}Why this happens:${NC} WSL2 stores ext4 in one VHDX file. The VHDX is"
        echo "  grow-only — docker builds grow it, but 'docker prune' inside WSL only"
        echo "  marks blocks free in ext4. Windows still sees the full file size,"
        echo "  which is why Windows reports 'no space' while 'df -h' inside WSL"
        echo "  looks fine. 'compact vdisk' releases those freed blocks back to Windows."
        echo ""
        echo -e "${YELLOW}One-time reclaim — run in Windows Admin PowerShell:${NC}"
        echo "  wsl --shutdown"
        echo "  diskpart"
        echo '  select vdisk file="%LOCALAPPDATA%\Packages\CanonicalGroupLimited.Ubuntu_79rhkp1fndgsc\LocalState\ext4.vhdx"'
        echo "  compact vdisk"
        echo "  detach vdisk"
        echo "  exit"
        echo ""
        echo -e "${GREEN}Permanent fix (WSL >= 2.0.9) — no more manual compaction:${NC}"
        echo "  1. Edit %USERPROFILE%\\.wslconfig on Windows and add:"
        echo "       [experimental]"
        echo "       sparseVhd=true"
        echo "  2. From Windows PowerShell (once):"
        echo "       wsl --shutdown"
        echo "       wsl --manage Ubuntu --set-sparse true"
        echo "  After this, freed ext4 blocks auto-release to Windows."
        echo "  Inside WSL, 'sudo fstrim -av' speeds reclaim after heavy docker churn."
    fi

    echo ""
    echo -e "${GREEN}Docker cleanup complete.${NC}"
}

# =============================================================================
#  Audio/Webcam: generate override compose snippet
# =============================================================================
generate_device_override() {
    local OVERRIDE="$PROJECT_ROOT/.docker-compose.devices.yaml"

    # Check if any device flags are set
    if ! $ENABLE_AUDIO && ! $ENABLE_WEBCAM; then
        # Remove stale override if no devices requested
        rm -f "$OVERRIDE"
        return 0
    fi

    cat > "$OVERRIDE" <<'HEADER'
# Auto-generated by dev-shell.sh — do not edit
services:
  omniedge:
HEADER

    # Volumes (PulseAudio socket)
    if $ENABLE_AUDIO; then
        cat >> "$OVERRIDE" <<'AUDIO_VOL'
    volumes:
      - /mnt/wslg/PulseServer:/mnt/wslg/PulseServer
AUDIO_VOL
    fi

    # Devices (webcam)
    if $ENABLE_WEBCAM; then
        if [ -e /dev/video0 ]; then
            cat >> "$OVERRIDE" <<'WEBCAM_DEV'
    devices:
      - /dev/video0:/dev/video0
WEBCAM_DEV
        else
            echo -e "${YELLOW}WARN: /dev/video0 not found — webcam skipped.${NC}"
            echo "  1. Attach from Windows: usbipd attach --wsl"
            echo "  2. Restart container:   $RESTART_CMD"
        fi
    fi

    # Environment (PulseAudio server)
    if $ENABLE_AUDIO; then
        cat >> "$OVERRIDE" <<'AUDIO_ENV'
    environment:
      - PULSE_SERVER=unix:/mnt/wslg/PulseServer
AUDIO_ENV
    fi

    # Add this override to compose files
    COMPOSE_FILES="$COMPOSE_FILES -f $OVERRIDE"
    echo -e "${GREEN}Devices enabled:${NC}$($ENABLE_AUDIO && echo ' audio')$($ENABLE_WEBCAM && echo ' webcam')"
}

# =============================================================================
#  Screen capture: launch oe_screen_capture.exe on Windows via WSL interop
# =============================================================================
launch_screen_capture() {
    if ! $ENABLE_SCREEN; then
        return 0
    fi

    # Common install locations (Windows paths accessible from WSL)
    local SCREEN_EXE=""
    local CANDIDATES=(
        "/mnt/c/dev/screen_capture/build/Release/oe_screen_capture.exe"
        "/mnt/c/dev/OmniEdge/tools/screen_capture/build/Release/oe_screen_capture.exe"
        "/mnt/c/Users/$USER/dev/screen_capture/build/Release/oe_screen_capture.exe"
    )
    for candidate in "${CANDIDATES[@]}"; do
        if [ -f "$candidate" ]; then
            SCREEN_EXE="$candidate"
            break
        fi
    done

    if [ -z "$SCREEN_EXE" ]; then
        echo -e "${YELLOW}WARN: oe_screen_capture.exe not found at common paths.${NC}"
        echo "  Build it from tools/screen_capture/ on Windows, then set:"
        echo "  export OE_SCREEN_CAPTURE_EXE='/mnt/c/path/to/oe_screen_capture.exe'"
        # Check env override
        if [ -n "${OE_SCREEN_CAPTURE_EXE:-}" ] && [ -f "$OE_SCREEN_CAPTURE_EXE" ]; then
            SCREEN_EXE="$OE_SCREEN_CAPTURE_EXE"
        else
            return 0
        fi
    fi

    # Check if already running
    if powershell.exe -Command "Get-Process oe_screen_capture -ErrorAction SilentlyContinue" &>/dev/null 2>&1; then
        echo -e "${GREEN}Screen capture already running.${NC}"
        return 0
    fi

    # Convert WSL path to Windows path and launch
    local WIN_PATH
    WIN_PATH=$(wslpath -w "$SCREEN_EXE" 2>/dev/null || echo "")
    if [ -n "$WIN_PATH" ]; then
        echo "Launching screen capture: $WIN_PATH"
        cmd.exe /c start "" "$WIN_PATH" &>/dev/null 2>&1 &
        sleep 1
        echo -e "${GREEN}Screen capture launched on Windows (port 5002).${NC}"
    else
        echo -e "${YELLOW}WARN: Could not convert path for Windows launch.${NC}"
        echo "  Launch manually on Windows: $SCREEN_EXE"
    fi
}

# =============================================================================
#  .env generation — auto-detect system resources for docker-compose
# =============================================================================
generate_env() {
    local env_file="$PROJECT_ROOT/.env"
    if [[ -f "$env_file" ]]; then
        return  # User already has a .env — don't overwrite
    fi

    local total_mb
    total_mb=$(get_total_mem_mb)
    local total_gb=$(( total_mb / 1024 ))

    # Container gets 75% of system RAM
    local mem_limit=$(( total_gb * 3 / 4 ))
    (( mem_limit < 4 )) && mem_limit=4  # Floor: 4 GB

    # /tmp gets ~1/6 of system RAM, minimum 1 GB
    local tmp_size=$(( total_gb / 6 ))
    (( tmp_size < 1 )) && tmp_size=1

    cat > "$env_file" <<EOF
# Auto-generated by dev-shell.sh (detected ${total_gb} GB RAM)
# Edit freely — this file won't be overwritten once it exists.
MEM_LIMIT=${mem_limit}g
TMP_SIZE=${tmp_size}g
EOF
    echo -e "${GREEN}Generated .env (MEM_LIMIT=${mem_limit}g, TMP_SIZE=${tmp_size}g) from ${total_gb} GB system RAM.${NC}"
}

# =============================================================================
#  Main
# =============================================================================
echo "============================================="
echo "  OmniEdge_AI — Dev Shell"
echo "============================================="

# ── Handle --clean: run cleanup, then continue with build if other flags set ─
if $CLEAN_DISK; then
    clean_all
    # Exit if --clean was the only intent (no --rebuild, no devices)
    if ! $REBUILD && ! $ENABLE_AUDIO && ! $ENABLE_WEBCAM && ! $ENABLE_SCREEN; then
        exit 0
    fi
    echo ""
    echo -e "${BOLD}Continuing with build...${NC}"
fi

generate_env
ensure_docker_ready
stage_libs
generate_device_override

# ── Disk preflight: check Windows host disk (the real bottleneck on WSL2) ────
check_windows_disk

# ── Disk preflight: also check WSL internal space ────────────────────────────
AVAIL_GB=$(( $(df --output=avail "$PROJECT_ROOT" 2>/dev/null | tail -1 | tr -d ' ') / 1024 / 1024 )) 2>/dev/null || AVAIL_GB=0
if (( AVAIL_GB < 10 )); then
    echo -e "${RED}ERROR: Only ${AVAIL_GB} GB WSL disk free — need at least 25 GB for image + models.${NC}"
    echo "  Run: bash $0 --clean   to free space"
    exit 1
elif (( AVAIL_GB < 25 )); then
    echo -e "${YELLOW}WARNING: Only ${AVAIL_GB} GB WSL disk free — recommend 25 GB for image + models.${NC}"
fi

# ── Build dev image if missing (or --rebuild) ────────────────────────────────
if $REBUILD && docker image inspect "$IMAGE" &>/dev/null; then
    echo -e "${YELLOW}Removing old dev image for rebuild...${NC}"
    # Stop container first so image can be removed
    docker compose $COMPOSE_FILES down 2>/dev/null || true
    docker rmi "$IMAGE" 2>/dev/null || true
fi

if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo ""
    check_memory "docker build" 8192

    # Cap Docker build memory to 80% of total to prevent WSL2 OOM
    total_mb=$(get_total_mem_mb)
    build_mem_limit="$(( total_mb * 80 / 100 ))m"

    echo "Building dev image (memory limit: ${build_mem_limit})..."
    docker build --target dev -t "$IMAGE" \
        --memory="$build_mem_limit" \
        --build-arg BUILDKIT_INLINE_CACHE=1 \
        "$PROJECT_ROOT"

    # Clean dangling images (but preserve build cache for fast rebuilds)
    docker image prune -f 2>/dev/null || true
fi

# ── Start container if not running ───────────────────────────────────────────
if ! docker compose $COMPOSE_FILES ps --status running 2>/dev/null | grep -q omniedge; then
    check_memory "container start" 4096
    docker compose $COMPOSE_FILES up -d
fi

# ── Launch screen capture on Windows (if requested) ─────────────────────────
launch_screen_capture

echo ""
echo -e "${GREEN}Dev container running.${NC}"
echo "  Inside: bash run_conversation.sh    (default profile launcher)"
echo "          bash run_security_mode.sh   (security profile)"
echo "          bash run_beautymode.sh      (beauty profile)"
echo "  Exit:   Ctrl+D or 'exit'"
echo ""

exec docker compose $COMPOSE_FILES exec omniedge bash
