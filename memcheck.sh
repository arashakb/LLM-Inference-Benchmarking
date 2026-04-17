#!/usr/bin/env bash
# memcheck.sh — Quick memory overview for Mac Mini LLM inference sessions
# Usage: ./memcheck.sh [--top N]

set -eo pipefail

TOP_N=20
if [[ "${1:-}" == "--top" ]]; then TOP_N="${2:-20}"; elif [[ -n "${1:-}" ]]; then TOP_N="$1"; fi

# Colors
BOLD='\033[1m'
DIM='\033[2m'
CYAN='\033[36m'
YELLOW='\033[33m'
GREEN='\033[32m'
RED='\033[31m'
RESET='\033[0m'

# --- System Summary ---
echo -e "${BOLD}${CYAN}=== Memory Overview ===${RESET}"
echo ""

# Total RAM
total_bytes=$(sysctl -n hw.memsize)
total_gb=$(echo "scale=1; $total_bytes / 1073741824" | bc)

# Parse memory_pressure output
mp_output=$(memory_pressure 2>/dev/null)
free_pct=$(echo "$mp_output" | grep "System-wide memory free percentage" | awk '{print $NF}')

# Parse top for PhysMem line
phys=$(top -l 1 -s 0 -n 0 2>/dev/null | grep "PhysMem")
used=$(echo "$phys" | sed -E 's/.*PhysMem: ([0-9]+[MG]) used.*/\1/')
wired=$(echo "$phys" | sed -E 's/.*\(([0-9]+[MG]) wired.*/\1/')
compressor=$(echo "$phys" | sed -E 's/.*[, ]([0-9]+[MG]) compressor.*/\1/')
unused=$(echo "$phys" | sed -E 's/.*\) *, *([0-9]+[MG]) unused.*/\1/')

# Swap
swap_in=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $3}' || echo "n/a")
swap_used=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' || echo "n/a")

# Color based on free %
pct_num=${free_pct/\%/}
if (( pct_num >= 50 )); then
    pct_color=$GREEN
elif (( pct_num >= 25 )); then
    pct_color=$YELLOW
else
    pct_color=$RED
fi

printf "  Total RAM:    %s GB\n" "$total_gb"
printf "  Used:         %s\n" "$used"
printf "  Free:         %s\n" "$unused"
printf "  Wired:        %s\n" "$wired"
printf "  Compressor:   %s\n" "$compressor"
printf "  Swap used:    %s\n" "$swap_used"
echo -e "  Free %%:       ${pct_color}${free_pct}${RESET}"
echo ""

# --- Load ---
load=$(sysctl -n vm.loadavg | tr -d '{}' | awk '{$1=$1; print}')
procs=$(ps -e | wc -l | tr -d ' ')
echo -e "${BOLD}${CYAN}=== System Load ===${RESET}"
printf "  Load avg:     %s\n" "$load"
printf "  Processes:    %s\n" "$procs"
echo ""

# --- Top Processes by RSS ---
echo -e "${BOLD}${CYAN}=== Top ${TOP_N} Processes by Memory ===${RESET}"
echo ""
printf "  ${DIM}%-8s %10s  %s${RESET}\n" "PID" "RSS" "COMMAND"
printf "  ${DIM}%-8s %10s  %s${RESET}\n" "---" "---" "-------"

ps -eo pid,rss,comm -r | awk -v n="$TOP_N" '
NR > 1 && NR <= n+1 && $2 > 0 {
    mb = $2 / 1024
    cmd = $3
    # shorten common paths
    gsub(/\/Applications\//, "", cmd)
    gsub(/\.app\/Contents\/MacOS\//, " ", cmd)
    gsub(/\.app\/Contents\/Frameworks\//, " fwk/", cmd)
    gsub(/\/System\/Library\//, "sys/", cmd)
    gsub(/\/Users\/bruski\//, "~/", cmd)
    gsub(/\/usr\//, "", cmd)
    if (mb >= 1024)
        printf "  %-8s %8.1f GB  %s\n", $1, mb/1024, cmd
    else
        printf "  %-8s %8.1f MB  %s\n", $1, mb, cmd
    }
'
echo ""

# --- Grouped App Totals ---
echo -e "${BOLD}${CYAN}=== Grouped App Totals (>50 MB) ===${RESET}"
echo ""
printf "  ${DIM}%10s  %s${RESET}\n" "TOTAL" "APP GROUP"
printf "  ${DIM}%10s  %s${RESET}\n" "-----" "---------"

ps -eo rss,comm | awk '
NR > 1 && $1 > 0 {
    cmd = $2
    if (cmd ~ /[Cc]hrome/)       group = "Google Chrome"
    else if (cmd ~ /[Cc]ursor/)  group = "Cursor (IDE)"
    else if (cmd ~ /claude/)     group = "Claude Code"
    else if (cmd ~ /[Dd]ocker/)  group = "Docker"
    else if (cmd ~ /[Oo]bsidian/) group = "Obsidian"
    else if (cmd ~ /[Tt]odoist/) group = "Todoist"
    else if (cmd ~ /[Tt]elegram/) group = "Telegram"
    else if (cmd ~ /OneDrive/)   group = "OneDrive"
    else if (cmd ~ /[Oo]utlook/) group = "Outlook"
    else if (cmd ~ /GitKraken/)  group = "GitKraken"
    else if (cmd ~ /Typeless/)   group = "Typeless"
    else if (cmd ~ /Tailscale/)  group = "Tailscale"
    else if (cmd ~ /openclaw/)   group = "OpenClaw"
    else if (cmd ~ /iTerm/)      group = "iTerm2"
    else if (cmd ~ /Raycast/)    group = "Raycast"
    else if (cmd ~ /Codex/)      group = "Codex"
    else if (cmd ~ /TeamViewer/) group = "TeamViewer"
    else if (cmd ~ /Snipaste/)   group = "Snipaste"
    else if (cmd ~ /python/)     group = "Python"
    else if (cmd ~ /node/)       group = "Node.js (other)"
    else group = ""

    if (group != "") totals[group] += $1
}
END {
    n = 0
    for (g in totals) {
        mb = totals[g] / 1024
        if (mb >= 50) {
            arr[n] = mb
            names[n] = g
            n++
        }
    }
    # simple sort descending
    for (i = 0; i < n; i++)
        for (j = i+1; j < n; j++)
            if (arr[j] > arr[i]) { t=arr[i]; arr[i]=arr[j]; arr[j]=t; tn=names[i]; names[i]=names[j]; names[j]=tn }
    for (i = 0; i < n; i++) {
        mb = arr[i] + 0
        name = names[i]
        if (mb >= 1024)
            printf "  %8.1f GB  %s\n", mb/1024, name
        else
            printf "  %8.0f MB  %s\n", mb, name
    }
}
'
echo ""
