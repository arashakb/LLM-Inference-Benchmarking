#!/usr/bin/env bash
# memcheck.sh — Quick memory overview for LLM inference sessions.
# Supports macOS (Apple Silicon / Metal) and Linux (NVIDIA CUDA).
# Usage: ./memcheck.sh [--top N]

set -eo pipefail

TOP_N=5
if [[ "${1:-}" == "--top" ]]; then TOP_N="${2:-5}"; elif [[ -n "${1:-}" ]]; then TOP_N="$1"; fi

OS=$(uname -s)

# --- System Summary ---
echo "=== Memory Overview ==="
echo ""

if [[ "$OS" == "Darwin" ]]; then
    total_bytes=$(sysctl -n hw.memsize)
    total_gb=$(echo "scale=1; $total_bytes / 1073741824" | bc)

    mp_output=$(memory_pressure 2>/dev/null)
    free_pct=$(echo "$mp_output" | grep "System-wide memory free percentage" | awk '{print $NF}')

    phys=$(top -l 1 -s 0 -n 0 2>/dev/null | grep "PhysMem")
    used=$(echo "$phys" | sed -E 's/.*PhysMem: ([0-9]+[MG]) used.*/\1/')
    wired=$(echo "$phys" | sed -E 's/.*\(([0-9]+[MG]) wired.*/\1/')
    compressor=$(echo "$phys" | sed -E 's/.*[, ]([0-9]+[MG]) compressor.*/\1/')
    unused=$(echo "$phys" | sed -E 's/.*\) *, *([0-9]+[MG]) unused.*/\1/')

    swap_used=$(sysctl -n vm.swapusage 2>/dev/null | awk '{print $6}' || echo "n/a")

    printf "  Total RAM:    %s GB\n" "$total_gb"
    printf "  Used:         %s\n" "$used"
    printf "  Free:         %s\n" "$unused"
    printf "  Wired:        %s\n" "$wired"
    printf "  Compressor:   %s\n" "$compressor"
    printf "  Swap used:    %s\n" "$swap_used"
    printf "  Free %%:       %s\n" "$free_pct"
elif [[ "$OS" == "Linux" ]]; then
    mem_total_kb=$(awk '/^MemTotal:/ {print $2}' /proc/meminfo)
    mem_avail_kb=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    mem_free_kb=$(awk '/^MemFree:/ {print $2}' /proc/meminfo)
    buffers_kb=$(awk '/^Buffers:/ {print $2}' /proc/meminfo)
    cached_kb=$(awk '/^Cached:/ {print $2}' /proc/meminfo)
    swap_total_kb=$(awk '/^SwapTotal:/ {print $2}' /proc/meminfo)
    swap_free_kb=$(awk '/^SwapFree:/ {print $2}' /proc/meminfo)

    kb_to_gb() { awk -v v="$1" 'BEGIN {printf "%.1f", v/1048576}'; }
    total_gb=$(kb_to_gb "$mem_total_kb")
    used_kb=$((mem_total_kb - mem_avail_kb))
    used_gb=$(kb_to_gb "$used_kb")
    avail_gb=$(kb_to_gb "$mem_avail_kb")
    free_gb=$(kb_to_gb "$mem_free_kb")
    cached_gb=$(kb_to_gb $((buffers_kb + cached_kb)))
    swap_used_kb=$((swap_total_kb - swap_free_kb))
    swap_used_mb=$(awk -v v="$swap_used_kb" 'BEGIN {printf "%.1f", v/1024}')
    free_pct=$(awk -v a="$mem_avail_kb" -v t="$mem_total_kb" 'BEGIN {printf "%.1f", a*100/t}')

    printf "  Total RAM:    %s GB\n" "$total_gb"
    printf "  Used:         %s GB\n" "$used_gb"
    printf "  Available:    %s GB\n" "$avail_gb"
    printf "  Free:         %s GB\n" "$free_gb"
    printf "  Cached/Buf:   %s GB\n" "$cached_gb"
    printf "  Swap used:    %s MB\n" "$swap_used_mb"
    printf "  Free %%:       %s\n" "$free_pct"
else
    printf "  Unsupported OS: %s\n" "$OS"
fi
echo ""

# --- Load ---
echo "=== System Load ==="
if [[ "$OS" == "Darwin" ]]; then
    load=$(sysctl -n vm.loadavg | tr -d '{}' | awk '{$1=$1; print}')
    procs=$(ps -e | wc -l | tr -d ' ')
elif [[ "$OS" == "Linux" ]]; then
    load=$(cut -d' ' -f1-3 /proc/loadavg)
    procs=$(ps -e --no-headers 2>/dev/null | wc -l | tr -d ' ')
fi
printf "  Load avg:     %s\n" "$load"
printf "  Processes:    %s\n" "$procs"
echo ""

# --- GPU ---
if [[ "$OS" == "Darwin" ]]; then
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
elif [[ "$OS" == "Linux" ]]; then
    echo "=== GPU (NVIDIA CUDA) ==="
    echo ""

    NVSMI=""
    for p in nvidia-smi /usr/lib/wsl/lib/nvidia-smi /usr/bin/nvidia-smi; do
        if command -v "$p" &>/dev/null; then NVSMI="$p"; break; fi
    done

    if [[ -n "$NVSMI" ]]; then
        gpu_rows=$("$NVSMI" \
            --query-gpu=driver_version,index,name,memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw \
            --format=csv,noheader,nounits 2>/dev/null)

        if [[ -n "$gpu_rows" ]]; then
            driver_ver=$(echo "$gpu_rows" | head -1 | awk -F',' '{print $1}' | xargs)
            cuda_ver=$("$NVSMI" 2>/dev/null | grep -oE 'CUDA Version: [0-9.]+' | head -1 | awk '{print $NF}')
            [[ -n "$driver_ver" ]] && printf "  Driver:       %s\n" "$driver_ver"
            [[ -n "$cuda_ver"   ]] && printf "  CUDA:         %s\n" "$cuda_ver"
            echo ""

            echo "$gpu_rows" | while IFS=',' read -r drv idx name mtot mused mfree ugpu umem temp power; do
                idx=$(echo "$idx" | xargs)
                name=$(echo "$name" | xargs)
                mtot=$(echo "$mtot" | xargs)
                mused=$(echo "$mused" | xargs)
                mfree=$(echo "$mfree" | xargs)
                ugpu=$(echo "$ugpu" | xargs)
                umem=$(echo "$umem" | xargs)
                temp=$(echo "$temp" | xargs)
                power=$(echo "$power" | xargs)
                printf "  GPU %s: %s\n" "$idx" "$name"
                printf "    VRAM used:    %s / %s MB (free %s MB)\n" "$mused" "$mtot" "$mfree"
                printf "    GPU active:   %s%%\n" "$ugpu"
                printf "    Mem bw:       %s%%\n" "$umem"
                printf "    Temp:         %s C\n" "$temp"
                printf "    Power:        %s W\n" "$power"
            done

            apps=$("$NVSMI" --query-compute-apps=pid,process_name,used_memory \
                              --format=csv,noheader,nounits 2>/dev/null)
            if [[ -n "$apps" ]]; then
                echo ""
                printf "  GPU processes:\n"
                printf "    %-8s %10s  %s\n" "PID" "VRAM" "PROCESS"
                echo "$apps" | while IFS=',' read -r pid pname pmem; do
                    pid=$(echo "$pid" | xargs)
                    pname=$(echo "$pname" | xargs)
                    pmem=$(echo "$pmem" | xargs)
                    [[ -z "$pid" ]] && continue
                    printf "    %-8s %7s MB  %s\n" "$pid" "$pmem" "$pname"
                done
            fi
        else
            printf "  nvidia-smi:   query returned no GPUs\n"
        fi
    else
        printf "  nvidia-smi:   not found (no NVIDIA driver?)\n"
    fi
fi
echo ""

# --- Top Processes by RSS ---
echo "=== Top ${TOP_N} Processes by Memory ==="
echo ""
printf "  %-8s %10s  %s\n" "PID" "RSS" "COMMAND"
printf "  %-8s %10s  %s\n" "---" "---" "-------"

if [[ "$OS" == "Darwin" ]]; then
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
elif [[ "$OS" == "Linux" ]]; then
    ps -eo pid,rss,comm --sort=-rss --no-headers | awk -v n="$TOP_N" '
    NR <= n && $2 > 0 {
        mb = $2 / 1024
        cmd = $3
        if (mb >= 1024)
            printf "  %-8s %8.1f GB  %s\n", $1, mb/1024, cmd
        else
            printf "  %-8s %8.1f MB  %s\n", $1, mb, cmd
    }'
fi
echo ""

# --- Grouped App Totals ---
echo "=== Grouped App Totals (>50 MB) ==="
echo ""
printf "  %10s  %s\n" "TOTAL" "APP GROUP"
printf "  %10s  %s\n" "-----" "---------"

if [[ "$OS" == "Darwin" ]]; then
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
elif [[ "$OS" == "Linux" ]]; then
    ps -eo rss,comm --no-headers | awk '
    $1 > 0 {
        cmd = $2
        if (cmd ~ /^python/)                              group = "Python"
        else if (cmd ~ /ollama/)                          group = "Ollama"
        else if (cmd ~ /vllm/)                            group = "vLLM"
        else if (cmd ~ /llama-server|llama\.cpp|llama_cpp/) group = "llama.cpp"
        else if (cmd ~ /text-generation|tgi/)             group = "TGI"
        else if (cmd ~ /[Dd]ocker|containerd|runc/)       group = "Docker/containerd"
        else if (cmd ~ /^node$/)                          group = "Node.js"
        else if (cmd ~ /code|cursor/)                     group = "IDE"
        else if (cmd ~ /claude/)                          group = "Claude Code"
        else if (cmd ~ /jupyter|ipykernel/)               group = "Jupyter"
        else if (cmd ~ /systemd/)                         group = "systemd"
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
fi
echo ""
