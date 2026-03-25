# Jetson Log Agent

Autonomous log investigation agent for NVIDIA Jetson Orin Nano. Searches hardware and inference pipeline logs, diagnoses root causes using a RAG-powered field manual, and escalates critical issues via email — all running locally on a 4B parameter LLM.

## Architecture

Three LangGraph ReAct agents connected to a local llama.cpp server (Nemotron-3 Nano 4B):

```
+------------------------------------------+
|           Main Agent (Router)            |
|                                          |
|  search_logs           consult_manual    |
|  send_email            reboot_device     |
+--------+-----------------------+---------+
         |                       |
+--------v-----------+ +---------v-----------------+
| Log Search         | | Manual Consultant         |
| Sub-Agent          | | Sub-Agent                 |
|                    | |                           |
| Tool: shell        | | Tool: search_manual       |
| (grep, awk, date)  | | (FAISS + embeddings)      |
+--------+-----------+ +---------+-----------------+
         |                       |
    demo_logs/             field_manual.md
 app.log thermal.log      (12 incident sections)
   dmesg.log
```

**Main agent** follows an investigation procedure loaded from the field manual. It delegates log searching and manual lookups to specialized sub-agents, then decides whether to email ops or reboot.

**Log search sub-agent** knows the three log formats (ISO timestamps, syslog, dmesg seconds-since-boot) and the correct shell commands to filter each. Given a time window, it runs the queries and returns raw error lines.

**Manual consultant sub-agent** searches a FAISS-indexed field manual using dense embeddings (BAAI/bge-small-en-v1.5 via fastembed). Returns severity, root causes, and recommended actions for each error.

## Synthetic Log Scenario

`gen_logs.py` generates 24 hours of realistic logs anchored to the current time. An incident is placed ~2 hours ago:

1. A third inference pipeline (camera-03) starts, pushing GPU utilization to 95%
2. Temperature rises rapidly — fan ramps to 100%
3. GPU thermal throttle triggers at 70.5°C — clocks drop 918 → 624 MHz
4. Inference deadline misses cascade across all three pipelines
5. Deep throttle at 73°C — clocks drop further to 420 MHz
6. Inference queue fills (32/32) — pipeline stalls
7. Load shedding: camera-03 suspended, TensorRT engine unloaded
8. Recovery over 30 seconds — clocks restore, pipelines stabilize
9. Camera-03 restarts after 60 seconds

Three log files:

| File | Timestamp format | Levels |
|------|-----------------|--------|
| `app.log` | ISO (`2026-03-24T17:00:00.000Z`) | `level=ERROR/WARN/INFO` |
| `thermal.log` | Syslog (`Mar 24 17:00:00`) | `ERROR/WARN/INFO` |
| `dmesg.log` | Seconds since boot (`[82810.000000]`) | `CRITICAL/WARNING` |

## Field Manual

`field_manual.md` is the single source of truth for agent behavior. It contains:

- **Investigation Procedure** — the workflow the agent follows (search logs → look up errors → escalate → summarize)
- **12 incident sections** — each with severity, log signatures, root causes, and recommended actions

| Section | Severity |
|---------|----------|
| Thermal Throttle — GPU Over-Temperature | Critical |
| Thermal Throttle — Sustained / Deep Throttle | Critical |
| Rapid Temperature Increase | High |
| Power Budget Exceeded | High |
| Inference Deadline Miss | High |
| Inference Latency Degradation | Medium |
| Frame Drops — Inference Backpressure | High |
| Inference Queue Full — Pipeline Stall | Critical |
| Pipeline Degraded Mode | High |
| Pipeline Suspended — Load Shedding | Critical |
| GPU Memory Warning | Medium |
| RTSP Stream Hiccup | Low |

Updating the manual changes how the agent investigates — no code changes needed. The FAISS index is rebuilt automatically when the manual changes.

## Prerequisites

**Hardware:**
- NVIDIA Jetson Orin Nano (8GB)
- SSD storage (model, swap, project files)

**Software on Jetson:**
- JetPack 6.x (Ubuntu 22.04, Python 3.10+)
- [llama.cpp](https://github.com/ggerganov/llama.cpp) built at `/opt/llama.cpp/build/bin/`
- [bubblewrap](https://github.com/containers/bubblewrap) — `sudo apt install bubblewrap`
- [socat](http://www.dest-unreach.org/socat/) — `sudo apt install socat` (network bridge for sandbox)
- [uv](https://github.com/astral-sh/uv) — installed automatically by `make install`

**Dev machine (for remote deployment):**
- SSH access to Jetson
- rsync

## Setup

### 1. Install system dependencies on Jetson

```bash
sudo apt install bubblewrap socat
```

Build llama.cpp (if not already at `/opt/llama.cpp/build/bin/`):
```bash
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build -j$(nproc)
```

### 2. Deploy and set up the agent

From the dev machine:
```bash
git clone https://github.com/andrei-ace/jetson-nano-log-agent.git
cd jetson-nano-log-agent

# Edit REMOTE in Makefile if your Jetson hostname differs
make setup
```

This SSHs into the Jetson and:
1. Checks for bubblewrap, socat, and llama.cpp (fails early if missing)
2. Creates 8GB swap on SSD (prevents OOM with LLM + embeddings)
3. Creates Python venv with uv
4. Installs dependencies (langchain-openai, langgraph, fastembed, faiss-cpu)
5. Downloads Nemotron-3 Nano 4B GGUF (~1.9 GB)

### Manual setup on Jetson

```bash
cd /ssd/jetson-log-agent
make swap             # 8G swapfile on SSD (idempotent, persistent)
make install          # venv + deps (checks for bubblewrap, socat, llama.cpp)
make download-model   # Nemotron-3 Nano 4B GGUF
```

## Running

**Terminal 1** — start the LLM server:

```bash
make server
```

Starts llama-server on port 8080 with all layers on GPU and 16K context.

**Terminal 2** — launch the agent:

```bash
./launch.sh
```

This generates fresh logs, builds the FAISS index if needed, and launches the agent inside a bubblewrap sandbox.

```
Jetson Log Agent ready. Type your question (or 'quit' to exit).

You: any errors in the past 2 hours?
```

### Example session

```
You: any errors past 2 hours?

  [logs] last 2 hours
    [logs] $ date -u '+%Y-%m-%dT%H:%M'
    [logs] 2026-03-24T18:27
    [logs] $ date -u -d '2 hours ago' '+%Y-%m-%dT%H:%M'
    [logs] 2026-03-24T16:27
    [logs] $ awk -v cutoff=2026-03-24T16:27 '$1 >= cutoff' app.log | grep -E 'level=ERROR|level=WARN'
    [logs] 2026-03-24T16:27:14Z level=ERROR component=thermal msg="GPU thermal throttle activated" ...
    [logs] ...
    [logs] $ grep -E 'ERROR|WARN' thermal.log
    [logs] Mar 24 16:10:54 ERROR: GPU thermal throttle triggered: gpu=70.5C ...
    [logs] ...
    [logs] $ grep -E 'CRITICAL|WARNING' dmesg.log
    [logs] [82810.000000] tegra-thermal: CRITICAL: zone1(gpu) exceeded trip point 70000mC
    [logs] ...
  [manual] GPU thermal throttle triggered
    [manual] search: GPU thermal throttle triggered
    [manual] ## Thermal Throttle — GPU Over-Temperature
    [manual] **Severity:** Critical
    [manual] ...
  [email] To: ops-team@company.com | CRITICAL: GPU Thermal Throttle on Jetson-07

  [EMAIL SENT] To: ops-team@company.com | Subject: CRITICAL: GPU Thermal Throttle on Jetson-07

Critical GPU thermal throttle detected ~2h ago on jetson-07:
- GPU reached 70.5°C, clocks reduced 918→624→420 MHz
- Inference deadline misses across all pipelines
- Camera-03 suspended via load shedding, recovered after 60s

Recommended: inspect enclosure airflow, verify fan profile,
lower TensorRT batch size. Email sent to ops.
```

The agent:
1. Delegates log searching to the log sub-agent (cyan `[logs]` prefix)
2. Looks up each error in the field manual via the manual sub-agent (green `[manual]` prefix)
3. Sends an email with error details and recommended actions (yellow `[email]`)
4. Summarizes findings with evidence and root cause chain

### Viewing action log

Emails and reboot commands are persisted to `output/actions.log`:

```bash
cat output/actions.log
```

## Sandbox

The agent runs inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox:

| Mount | Mode | Purpose |
|-------|------|---------|
| /usr, /bin, /lib, /etc, /sys | Read-only | System libraries and config |
| .venv, run_agent.py, demo_logs | Read-only | Agent code and log files |
| kb_index/ | Read-write | FAISS index + embedding model cache |
| output/ | Read-write | Action log (emails, reboots) |
| /tmp | tmpfs | Ephemeral scratch space |

Process isolation: PID, IPC, UTS namespaces. Clean environment. Network is fully isolated (`--unshare-net`) — a socat bridge forwards only `127.0.0.1:8080` to the llama-server via a Unix socket. No other network access.

## Project Structure

```
├── run_agent.py        # Three-agent system (main router + 2 sub-agents)
├── build_index.py      # Offline FAISS index builder
├── gen_logs.py         # Synthetic log generator (24h + thermal incident)
├── field_manual.md     # Incident runbook (12 sections, RAG source of truth)
├── launch.sh           # Log gen + index build + bwrap sandbox launcher
├── Makefile            # Deploy, install, server, swap targets
├── pyproject.toml      # Python dependencies
├── demo_logs/          # Generated: app.log, thermal.log, dmesg.log
├── kb_index/           # Built: index.faiss, chunks.json, model_cache/
└── output/             # Persisted: actions.log (emails, reboots)
```

## Makefile Targets

| Target | Run from | Description |
|--------|----------|-------------|
| `make deploy` | Dev machine | Rsync project to Jetson |
| `make setup` | Dev machine | Deploy + swap + install + download model |
| `make swap` | Jetson | Create 8G SSD swapfile (idempotent) |
| `make install` | Jetson | Create venv, install Python deps |
| `make download-model` | Jetson | Download Nemotron-3 Nano 4B GGUF (~1.9 GB) |
| `make hf-login` | Jetson | Authenticate with HuggingFace (faster downloads) |
| `make server` | Jetson | Start llama-server on port 8080 |
| `make gen-logs` | Jetson | Regenerate logs with current timestamps |
| `make build-index` | Jetson | Rebuild FAISS index from field manual |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_DIR` | `/workspace/demo_logs` | Log files directory |
| `KB_DIR` | `/workspace/kb_index` | FAISS index directory |
| `ACTION_LOG` | `/workspace/output/actions.log` | Email/reboot action log |
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server endpoint |
| `OPENAI_API_KEY` | `not-needed` | API key (any string for local) |
| `OPENAI_MODEL` | `local-model` | Model name passed to ChatOpenAI |

All set automatically by `launch.sh` inside the sandbox.
