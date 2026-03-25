# Jetson Log Agent

Autonomous log investigation agent for NVIDIA Jetson Orin Nano. Searches hardware and inference pipeline logs, diagnoses root causes using a RAG-powered field manual, and escalates critical issues via email — all running locally on a 4B parameter LLM.

## Why Run Agents on Edge Devices

A GPU thermal throttle at 2 AM doesn't wait for a cloud API to respond. When a Jetson running inference pipelines on a factory floor hits a thermal cascade, the device needs to shed load in seconds — not minutes. An agent running locally on the device can read the logs, diagnose the root cause, and act immediately.

**No network dependency.** Edge devices operate in environments where connectivity is unreliable, metered, or restricted by policy. A surveillance system in a warehouse, a drone, or an offshore platform can't rely on cloud inference to reason about its own failures. The agent works identically whether the network is up or down.

**Sensitive data stays on device.** Hardware logs contain thermal profiles, power rail voltages, inference pipeline configurations, and camera stream URLs — operational telemetry that reveals the physical topology and security posture of the deployment. Sending this to a cloud API for analysis is a data exfiltration risk. A local agent processes everything in a sandbox and only emits structured alerts.

**Cost at scale.** A fleet of hundreds of Jetson devices, each generating logs every 5 seconds, would rack up significant API costs if every investigation required cloud LLM calls. A local 4B model running on hardware that's already deployed and powered has zero marginal cost per query.

**Closed-loop autonomy.** The real value isn't just reading logs — it's acting on them. This agent follows a field manual to triage severity, looks up recommended actions, emails the ops team, and can reboot devices as a last resort. That closed loop only works reliably if it runs where the hardware is, without depending on external services that might be the reason things are failing in the first place.

The trade-off is capability: a 4B model on a Jetson won't match a frontier model's reasoning. But for structured tasks with a well-defined procedure, a clear set of tools, and a domain-specific knowledge base — it's enough. The agent doesn't need to be creative. It needs to run `grep`, read the manual, and send an email.

## Why Nemotron-3 Nano 4B

The Jetson Orin Nano has 8GB of unified memory shared between CPU and GPU. Running a language model locally on this budget means every megabyte of KV cache matters — a standard transformer with full attention would exhaust memory after a few thousand tokens of context, leaving nothing for the inference pipelines the device is actually meant to run.

Nemotron-3 Nano 4B is a hybrid architecture that interleaves transformer attention layers with Mamba2 (selective state space) layers. Mamba2 layers process sequences in constant memory — they maintain a fixed-size hidden state rather than caching every past token's key-value pairs. Only the attention layers need a KV cache, and there are far fewer of them in the hybrid design.

The practical effect: Nemotron-3 Nano can run with a 16K context window on the Orin Nano while consuming a fraction of the KV cache memory that a pure transformer of the same size would need. This leaves enough headroom for the embedding model (fastembed/bge-small-en-v1.5), the FAISS index, and the Python agent runtime — all sharing that same 8GB.

The Q4_K_M quantization (4-bit, ~2.8 GB on disk) further reduces the memory footprint while preserving enough quality for the structured reasoning this agent needs: parsing log timestamps, following a step-by-step procedure, and formatting email reports.

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
| (grep, awk, date)  | | (FAISS + re-ranker)       |
+--------+-----------+ +---------+-----------------+
         |                       |
    demo_logs/                docs/
 app.log thermal.log     (4 docs, 30+ sections)
   dmesg.log
```

**Main agent** follows an investigation procedure loaded from the field manual. It delegates log searching and manual lookups to specialized sub-agents, then decides whether to email ops or reboot.

**Log search sub-agent** knows the three log formats (ISO timestamps, syslog, dmesg seconds-since-boot) and the correct shell commands to filter each. Given a time window, it runs the queries and returns raw error lines.

**Manual consultant sub-agent** searches the knowledge base using two-stage retrieval: FAISS with dense embeddings (BAAI/bge-small-en-v1.5) for broad recall, then a cross-encoder re-ranker (Xenova/ms-marco-MiniLM-L-6-v2, 80MB) for precision. Returns severity, root causes, and recommended actions.

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

The `docs/` directory is the knowledge base — all Markdown files are chunked by `##` headings, embedded, and indexed with FAISS. A cross-encoder re-ranker (Xenova/ms-marco-MiniLM-L-6-v2) improves retrieval precision at query time.

**`field_manual.md`** is the primary source. It contains:

- **Investigation Procedure** — the workflow the agent follows (search logs → look up errors → escalate → summarize)
- **14 incident sections** — each with severity, log signatures, root causes, and recommended actions

**Additional docs:**
- **`hardware_spec.md`** — Orin Nano specs, thermal envelope, memory budget, power rails, NVPMODEL profiles
- **`deployment_guide.md`** — provisioning, systemd services, pipeline config, monitoring, decommissioning
- **`known_issues.md`** — 6 documented platform quirks with workarounds (nvgpu driver hang, TensorRT memory leak, CUDA thrashing, etc.)

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

Updating any doc in `docs/` changes what the agent knows — no code changes needed. The FAISS index is rebuilt automatically when any doc changes.

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

### 2. Set up the agent

`make setup` auto-detects whether you're on the Jetson or a dev machine:

**On the Jetson directly:**
```bash
git clone https://github.com/andrei-ace/jetson-nano-log-agent.git
cd jetson-nano-log-agent
make setup
```

This creates swap, installs deps, and downloads the model — all locally.

**From a dev machine (deploys via SSH):**
```bash
git clone https://github.com/andrei-ace/jetson-nano-log-agent.git
cd jetson-nano-log-agent

# Edit the top of Makefile to match your setup:
#   REMOTE       := user@jetson-hostname
#   REMOTE_DIR   := /ssd/jetson-log-agent
#   LLAMA_SERVER := /path/to/llama-server
make setup
```

This rsyncs the code to the Jetson, then runs `make setup` there via SSH.

Either way, setup:
1. Checks for bubblewrap, socat, and llama.cpp (fails early if missing)
2. Creates 8GB swap on SSD (prevents OOM with LLM + embeddings)
3. Creates Python venv with uv
4. Installs dependencies (langchain-openai, langgraph, fastembed, faiss-cpu)
5. Downloads Nemotron-3 Nano 4B GGUF (~2.8 GB)
6. Builds FAISS index from `docs/` (downloads embedding model + re-ranker on first run, ~130 MB total)

### Manual setup (step by step)

```bash
cd /ssd/jetson-log-agent
make swap             # 8G swapfile on SSD (idempotent, persistent)
make install          # venv + deps (checks for bubblewrap, socat, llama.cpp)
make download-models  # All models: LLM GGUF + ONNX embedder + re-ranker
make build-index      # FAISS index from docs/
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

This generates fresh logs, builds the FAISS index if needed (downloads embedding model + re-ranker on first run), and launches the agent inside a bubblewrap sandbox.

```
Jetson Log Agent ready. Type your question (or 'quit' to exit).

You: any errors in the past 2 hours?
```

### Example session

```
You: check last hour

  [logs] last hour
    [logs] $ ISO_CUT=$(date -u -d '1 hour ago' '+%Y-%m-%dT%H:%M'); ...
    [logs] ISO=2026-03-25T15:50 SYS=15:50:27 DMESG=86395.015000
    [logs] $ awk -v c=2026-03-25T15:50 '$1>=c && /level=ERROR/' app.log
    [logs] 2026-03-25T16:19:50 level=ERROR ... msg="GPU thermal throttle activated"
    [logs] 2026-03-25T16:19:51 level=ERROR ... msg="Inference deadline missed" ...
    [logs] ...
    [logs] $ awk -v c=15:50:27 '$3>=c && /ERROR|WARN/' thermal.log
    [logs] Mar 25 16:19:45 ... WARN: Rapid temperature increase detected
    [logs] Mar 25 16:19:50 ... ERROR: GPU thermal throttle triggered: gpu=70.5C
    [logs] Mar 25 16:19:53 ... ERROR: GPU thermal throttle deepened: gpu=73.0C
    [logs] ...
    [logs] $ awk -F'[][]' -v c=86395.015000 '$2+0>=c && /CRITICAL|WARNING/' dmesg.log
    [logs] [88210.000000] tegra-thermal: CRITICAL: zone1(gpu) exceeded trip point 70000mC
    [logs] [88213.000000] tegra-thermal: CRITICAL: zone1(gpu) still above trip, deepening throttle
    [logs] [88213.000000] nvgpu: gpu0: WARNING: severe clock reduction

  [manual] thermal throttle GPU Jetson
    [manual] ## Thermal Throttle — GPU Over-Temperature
    [manual] **Severity:** Critical
    [manual] **Recommended actions:**
    [manual] 1. Physically inspect enclosure—remove dust from fan blades
    [manual] 2. Verify ambient temperature; if >35°C, improve room cooling
    [manual] 3. Reduce TensorRT max batch size from 4 to 2
    [manual] ...

  [email] To: ops-team@company.com | CRITICAL: GPU thermal throttle on jetson-07

  [EMAIL SENT] To: ops-team@company.com | Subject: CRITICAL: GPU thermal throttle on jetson-07

**Root Cause Chain:** Rapid temperature rise (+8.3°C/5s) → approaching throttle threshold
(63.5°C) → thermal throttle triggered (70.5°C) → deepened (73.0°C) → inference deadline
misses → pipeline stall → camera-03 suspended via load shedding.

**Evidence:**
- app.log: GPU thermal throttle, 12x inference deadline missed, pipeline stall, frame drops
- thermal.log: Rapid temperature increase, fan at maximum, cooling capacity limited
- dmesg.log: CRITICAL zone1(gpu) trip point exceeded, severe clock reduction

**Recommended Actions (from field manual):**
1. Physically inspect enclosure—remove dust from fan blades, ensure airflow unobstructed
2. Verify ambient temperature; if >35°C, improve room cooling or relocate unit
3. Reduce TensorRT max batch size from 4 to 2 (~40% GPU time reduction)
4. Reapply thermal paste (Noctua NT-H1) on GPU heatsink contact
5. Confirm fan profile set to `cool` via `sudo jetson_clocks --show`

Email sent to ops-team@company.com. No reboot required — thermal cascade self-recovered.
```

The agent:
1. Delegates log searching to the log sub-agent (cyan `[logs]` prefix) — computes cutoffs for all three log formats, searches each file with time filtering, deduplicates
2. Groups related errors (thermal throttle + deadline misses = one incident) and consults the field manual via the manual sub-agent (green `[manual]` prefix) — two-stage FAISS + cross-encoder re-ranking
3. Sends an email with error details and recommended actions from the manual (yellow `[email]`)
4. Summarizes with root cause chain, evidence from all three log files, and specific remediation steps
5. Checks reboot policy — not needed since the thermal cascade self-recovered

### Viewing action log

Emails and reboot commands are persisted to `output/actions.log`:

```
$ cat output/actions.log
============================================================
Date: 2026-03-25T17:19:05Z
From: jetson-log-agent@jetson-07
To: ops-team@company.com
Subject: CRITICAL: GPU Thermal Throttle and Pipeline Stall on Jetson Orin Nano
============================================================
CRITICAL ISSUE DETECTED: GPU thermal throttle has been triggered and
deepened, with GPU temperature reaching 73.0°C. This is causing pipeline
stall (camera-03 inference queue full) and RTSP stream disconnection.

Key evidence:
- app.log: GPU thermal throttle activated/deepened, pipeline stall,
  RTSP stream disconnected, power budget exceeded, frame drops
- thermal.log: Rapid temperature increase (+8.3°C/5s), throttle
  triggered (70.5°C), deepened (73.0°C)
- dmesg.log: CRITICAL: zone1(gpu) exceeded trip point 70000 mC

Recommended actions (from manual):
1. Physically inspect enclosure—remove dust, ensure airflow unobstructed.
2. Verify ambient temperature; if >35°C, improve room cooling.
3. Reduce TensorRT max batch size from 4 to 2 (~40% GPU time reduction).
4. Reapply thermal paste (Noctua NT-H1) on GPU heatsink contact.
5. Run `sudo jetson_clocks --show` to confirm fan profile is `cool`.

This is a Critical severity issue requiring immediate attention.
```

## Sandbox

The agent runs inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox:

| Mount | Mode | Purpose |
|-------|------|---------|
| /usr, /bin, /lib, /etc, /sys | Read-only | System libraries and config |
| .venv, run_agent.py, demo_logs | Read-only | Agent code and log files |
| kb_index/ | Read-only | FAISS index + embedding/re-ranker model cache |
| output/ | Read-write | Action log (emails, reboots) |
| /tmp | tmpfs | Ephemeral scratch space |

Process isolation: PID, IPC, UTS namespaces. Clean environment. Network is fully isolated (`--unshare-net`) — a socat bridge forwards only `127.0.0.1:8080` to the llama-server via a Unix socket. No other network access.

## Project Structure

```
├── run_agent.py        # Three-agent system (main router + 2 sub-agents)
├── build_index.py      # FAISS index builder + model downloader
├── gen_logs.py         # Synthetic log generator (24h, 2 incidents)
├── docs/               # Knowledge base (4 markdown docs, chunked for RAG)
│   ├── field_manual.md       # Incident runbook (14 sections)
│   ├── hardware_spec.md      # Orin Nano specs and limits
│   ├── deployment_guide.md   # Provisioning and operations
│   └── known_issues.md       # Platform quirks and workarounds
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
| `make download-models` | Jetson | Download all models (LLM GGUF + ONNX embedder + re-ranker) |
| `make hf-login` | Jetson | Authenticate with HuggingFace (faster downloads) |
| `make server` | Jetson | Start llama-server on port 8080 |
| `make gen-logs` | Jetson | Regenerate logs with current timestamps |
| `make build-index` | Jetson | Rebuild FAISS index from docs/ |

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
