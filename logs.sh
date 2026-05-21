#!/usr/bin/env bash
# logs.sh — quick log viewer for the weapon-detection pipeline
# Usage:  ./logs.sh              (show table)
#         ./logs.sh <number>     (tail that log)
#         ./logs.sh pbs          (tail HPC PBS output live via SSH)

set -euo pipefail

BOLD=$'\e[1m'; CYAN=$'\e[36m'; GREEN=$'\e[32m'; YELLOW=$'\e[33m'; DIM=$'\e[2m'; RESET=$'\e[0m'

# ── log registry ──────────────────────────────────────────────────────────────
declare -a NAMES PATHS
add() { NAMES+=("$1"); PATHS+=("$2"); }

add "PBS step9 stdout"          "logs/pbs_step9.out"
add "PBS step9 stderr"          "logs/pbs_step9.err"
add "train_all log"             "train_run.log"
add "part4 log"                 "part4_run.log"
add "alerts (JSONL)"            "logs/alerts/alerts.jsonl"
add "detector weights"          "logs/detector/best.pt"
add "classifier weights"        "logs/classifier/best.pt"
add "student weights"           "logs/student/weights/best.pt"
add "student last.pt"           "logs/student/weights/last.pt"
add "fp_correction weights"     "logs/fp_correction/detector_ft_best.pt"
add "HPC fp_correction weights" "runs/detect/logs/fp_correction/weights/best.pt"
add "student results CSV"       "runs/detect/logs/student/results.csv"
add "detector results CSV"      "runs/detect/train/results.csv"

# ── helpers ───────────────────────────────────────────────────────────────────
human_size() {
    local f="$1"
    [[ -f "$f" ]] || { echo "—"; return; }
    local b; b=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
    if   (( b >= 1073741824 )); then printf "%.1f GB" "$(echo "scale=1; $b/1073741824" | bc)"
    elif (( b >= 1048576 ));    then printf "%.1f MB" "$(echo "scale=1; $b/1048576"    | bc)"
    elif (( b >= 1024 ));       then printf "%.0f KB" "$(echo "scale=0; $b/1024"       | bc)"
    else echo "${b} B"; fi
}

mod_time() {
    local f="$1"
    [[ -f "$f" ]] || { echo ""; return; }
    stat -c '%y' "$f" 2>/dev/null | cut -c1-16 \
        || stat -f '%Sm' -t '%Y-%m-%d %H:%M' "$f" 2>/dev/null
}

# ── PBS tail via SSH ──────────────────────────────────────────────────────────
if [[ "${1:-}" == "pbs" ]]; then
    echo "${CYAN}Tailing HPC PBS log (Ctrl-C to stop)…${RESET}"
    ssh -t btech10170.23@172.16.220.100 "tail -f ~/Weapon_detection/logs/pbs_step9.out"
    exit 0
fi

# ── table ─────────────────────────────────────────────────────────────────────
echo ""
echo "${BOLD}${CYAN}  #   Log file                       Size       Modified          Path${RESET}"
echo "  ─────────────────────────────────────────────────────────────────────────────"

for i in "${!NAMES[@]}"; do
    num=$(( i + 1 ))
    name="${NAMES[$i]}"
    path="${PATHS[$i]}"

    if [[ -f "$path" ]]; then
        size=$(human_size "$path")
        mtime=$(mod_time "$path")
        printf "  ${GREEN}%-3s${RESET} %-32s ${YELLOW}%-10s${RESET} %-17s  ${DIM}%s${RESET}\n" \
            "$num" "$name" "$size" "$mtime" "$path"
    else
        printf "  ${DIM}%-3s %-32s %-10s %-17s  %s${RESET}\n" \
            "$num" "$name" "missing" "" "$path"
    fi
done

echo ""
echo "  ${DIM}./logs.sh <number>   tail -f that log${RESET}"
echo "  ${DIM}./logs.sh pbs        live SSH tail of HPC PBS output${RESET}"
echo ""

# ── optional: tail a numbered log ─────────────────────────────────────────────
if [[ -n "${1:-}" ]] && [[ "${1}" =~ ^[0-9]+$ ]]; then
    idx=$(( $1 - 1 ))
    if (( idx < 0 || idx >= ${#PATHS[@]} )); then
        echo "No log #${1}"; exit 1
    fi
    target="${PATHS[$idx]}"
    if [[ ! -f "$target" ]]; then
        echo "File not found: $target"; exit 1
    fi
    echo "${CYAN}── tailing ${target} (Ctrl-C to stop) ──────────────────${RESET}"
    tail -f "$target"
fi
