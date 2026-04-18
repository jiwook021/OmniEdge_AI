#!/usr/bin/env bash
# =============================================================================
# OmniEdge_AI — Mode INI Generator
#
# Generates a mode-specific INI from config/omniedge.ini:
#   conversation | security | beauty
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -f "${SCRIPT_DIR}/../config/omniedge.ini" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [[ -f "${SCRIPT_DIR}/../../config/omniedge.ini" ]]; then
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
else
    PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
fi

BASE_INI="${PROJECT_ROOT}/config/omniedge.ini"
if [[ ! -f "${BASE_INI}" ]]; then
    echo "Base INI not found: ${BASE_INI}" >&2
    exit 1
fi

MODE="${1:-}"
OUT_INI="${2:-}"
FRONTEND_DIR="${3:-}"

if [[ -z "${MODE}" || -z "${OUT_INI}" || -z "${FRONTEND_DIR}" ]]; then
    echo "Usage: $0 <conversation|security|beauty> <out_ini_path> <frontend_dir>" >&2
    exit 1
fi

case "${MODE}" in
    conversation|security|beauty) ;;
    *)
        echo "Unsupported mode: ${MODE}" >&2
        exit 1
        ;;
esac

mkdir -p "$(dirname "${OUT_INI}")"
cp "${BASE_INI}" "${OUT_INI}"

set_ini_value() {
    local section="$1" key="$2" value="$3" file="$4"
    local tmp
    tmp="$(mktemp)"
    awk -v section="$section" -v key="$key" -v value="$value" '
        BEGIN {
            in_sec = 0;
            section_found = 0;
            replaced = 0;
        }
        function section_name(line, s) {
            s = line;
            gsub(/^[ \t]*\[/, "", s);
            gsub(/\][ \t]*$/, "", s);
            return s;
        }
        {
            if ($0 ~ /^[ \t]*\[[^]]+\][ \t]*$/) {
                if (in_sec && !replaced) {
                    print key " = " value;
                    replaced = 1;
                }
                in_sec = (section_name($0) == section);
                if (in_sec) section_found = 1;
                print $0;
                next;
            }

            if (in_sec) {
                line = $0;
                sub(/^[ \t]+/, "", line);
                if (line ~ ("^" key "[ \t]*=")) {
                    print key " = " value;
                    replaced = 1;
                    next;
                }
            }

            print $0;
        }
        END {
            if (!section_found) {
                print "";
                print "[" section "]";
                print key " = " value;
            } else if (in_sec && !replaced) {
                print key " = " value;
            }
        }
    ' "$file" > "$tmp"
    mv "$tmp" "$file"
}

set_module_toggles() {
    local enabled_csv="$1"
    local enabled_set=",${enabled_csv},"
    local module
    local all_modules=(
        video_ingest audio_ingest screen_ingest websocket_bridge background_blur
        security_camera security_vlm conversation_model tts audio_denoise beauty
        face_recognition face_filter sam2 video_denoise
    )
    for module in "${all_modules[@]}"; do
        if [[ "${enabled_set}" == *",${module},"* ]]; then
            set_ini_value "modules" "${module}" "1" "${OUT_INI}"
        else
            set_ini_value "modules" "${module}" "0" "${OUT_INI}"
        fi
    done
}

set_boot_mode() {
    local key="$1"
    local modules_csv="$2"
    set_ini_value "boot_modes" "${key}" "${modules_csv}" "${OUT_INI}"
}

# Global overrides for all generated mode files.
set_ini_value "websocket_bridge" "frontend_dir" "${FRONTEND_DIR}" "${OUT_INI}"
set_ini_value "websocket_bridge" "ws_port" "9001" "${OUT_INI}"

case "${MODE}" in
    conversation)
        set_ini_value "daemon" "default_mode" "conversation" "${OUT_INI}"
        set_ini_value "daemon" "launch_order" "video_ingest,audio_ingest,screen_ingest,conversation_model" "${OUT_INI}"
        set_boot_mode "conversation"      "video_ingest,audio_ingest,screen_ingest,conversation_model"
        set_boot_mode "simple_llm"        "video_ingest,audio_ingest,screen_ingest,conversation_model"
        set_boot_mode "full_conversation" "video_ingest,audio_ingest,screen_ingest,conversation_model"
        set_module_toggles "video_ingest,audio_ingest,screen_ingest,websocket_bridge,background_blur,conversation_model,tts,audio_denoise"
        ;;
    security)
        set_ini_value "daemon" "default_mode" "security" "${OUT_INI}"
        set_ini_value "daemon" "launch_order" "video_ingest,audio_ingest,security_camera,security_vlm" "${OUT_INI}"
        set_boot_mode "security" "video_ingest,audio_ingest,security_camera,security_vlm"
        set_module_toggles "video_ingest,audio_ingest,websocket_bridge,security_camera,security_vlm"
        ;;
    beauty)
        set_ini_value "daemon" "default_mode" "beauty" "${OUT_INI}"
        set_ini_value "daemon" "launch_order" "video_ingest,audio_ingest,beauty" "${OUT_INI}"
        set_boot_mode "beauty" "video_ingest,audio_ingest,beauty"
        set_module_toggles "video_ingest,audio_ingest,websocket_bridge,beauty"
        ;;
esac

echo "${OUT_INI}"
