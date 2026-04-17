#!/usr/bin/env bash
# memcheck.sh — Quick memory overview for Mac Mini LLM inference sessions
# Usage: ./memcheck.sh [--top N]

set -eo pipefail

TOP_N=5
if [[ "${1:-}" == "--top" ]]; then TOP_N="${2:-5}"; elif [[ -n "${1:-}" ]]; then TOP_N="$1"; fi

# --- System Summary ---
echo "=== Memory Overview ==="
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
swap_used=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' || echo "n/a")

printf "  Total RAM:    %s GB\n" "$total_gb"
printf "  Used:         %s\n" "$used"
printf "  Free:         %s\n" "$unused"
printf "  Wired:        %s\n" "$wired"
printf "  Compressor:   %s\n" "$compressor"
printf "  Swap used:    %s\n" "$swap_used"
printf "  Free %%:       %s\n" "$free_pct"
echo ""

# --- Load ---
load=$(sysctl -n vm.loadavg | tr -d '{}' | awk '{$1=$1; print}')
procs=$(ps -e | wc -l | tr -d ' ')
echo "=== System Load ==="
printf "  Load avg:     %s\n" "$load"
printf "  Processes:    %s\n" "$procs"
echo ""

# --- GPU ---
echo "=== GPU (Apple Silicon) ==="
echo ""

perf=$(ioreg -r -c IOAccelerator | grep '"PerformanceStatistics"' | head -1)
if [[ -n "$perf" ]]; then
    alloc=$(echo "$perf"  | grep -oE '"Alloc system memory"=[0-9]+'   | grep -oE '[0-9]+$')
    device=$(echo "$perf" | grep -oE '"Device Utilization %"=[0-9]+'  | grep -oE '[0-9]+$')
    render=$(echo "$perf" | grep -oE '"Renderer Utilization %"=[0-9]+' | grep -oE '[0-9]+$')
    tiler=$(echo "$perf"  | grep -oE '"Tiler Utilization %"=[0-9]+'   | grep -oE '[0-9]+$')
    [[ -n "$alloc"  ]] && printf "  Metal alloc:  %.0f MB\n" "$(echo "scale=2; $alloc/1048576" | bc)"
    [[ -n "$device" ]] && printf "  GPU active:   %s%%\n" "$device"
    [[ -n "$render" ]] && printf "  Renderer:     %s%%\n" "$render"
    [[ -n "$tiler"  ]] && printf "  Tiler:        %s%%\n" "$tiler"
else
    printf "  Metal stats:  unavailable\n"
fi

if [[ $EUID -eq 0 ]]; then
    pm=$(powermetrics --samplers gpu_power -n1 -i500 2>/dev/null)
    freq=$(echo "$pm"  | grep -i "GPU frequency" | grep -oE '[0-9]+ MHz' | head -1)
    power=$(echo "$pm" | grep -i "GPU Power"     | grep -oE '[0-9.]+ mW'  | head -1)
    [[ -n "$freq"  ]] && printf "  GPU freq:     %s\n" "$freq"
    [[ -n "$power" ]] && printf "  GPU power:    %s\n" "$power"
else
    printf "  Freq/power: run as sudo for GPU wattage\n"
fi
echo ""

# --- Top Processes by RSS ---
echo "=== Top ${TOP_N} Processes by Memory ==="
echo ""
printf "  %-8s %10s  %s\n" "PID" "RSS" "COMMAND"
printf "  %-8s %10s  %s\n" "---" "---" "-------"

ps -eo pid,rss,comm -r | awk -v n="$TOP_N" '
NR > 1 && NR <= n+1 && $2 > 0 {
    mb = $2 / 1024
    cmd = $3
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
echo "=== Grouped App Totals (>50 MB) ==="
echo ""
printf "  %10s  %s\n" "TOTAL" "APP GROUP"
printf "  %10s  %s\n" "-----" "---------"

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
