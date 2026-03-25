# Jetson Log Agent

Autonomous log investigation agent for NVIDIA Jetson Orin Nano. Searches hardware and inference pipeline logs, diagnoses root causes using a RAG-powered field manual, and escalates critical issues via email — all running locally on a 4B parameter LLM.

## Why Run Agents on Edge Devices

Imagine a dozen Jetson Orin Nanos deployed on an offshore platform, running YOLOv8 inference on camera feeds 24/7 to monitor equipment, safety perimeters, and flare stacks. The nearest technician is a helicopter ride away, and satellite uplink costs dollars per megabyte.

The deployment model is simple: inference nodes only run the camera pipelines, while a separate monitoring Jetson collects their logs and runs the agent. When something goes wrong, it diagnoses the problem locally and sends a structured email over satellite — a diagnosis, root cause, and specific fix. Not megabytes of raw telemetry.

**No network dependency.** Edge devices operate in environments where connectivity is unreliable, metered, or restricted by policy. The agent works identically whether the network is up or down.

**Sensitive data stays on device.** Hardware logs contain thermal profiles, power rail voltages, inference pipeline configurations, and camera stream URLs — operational telemetry that reveals the physical topology and security posture of the deployment. A local agent processes everything in a sandbox and only emits structured alerts.

**Cost at scale.** A local 4B model running on hardware that's already deployed and powered has zero marginal cost per query.

**Closed-loop autonomy.** The agent follows a field manual to triage severity, looks up recommended actions, emails the ops team, and can reboot devices as a last resort. That closed loop only works reliably if it runs where the hardware is, without depending on external services that might be the reason things are failing in the first place.

For this class of operational task — following procedures, running shell commands, formatting structured reports — procedural reliability matters more than open-ended reasoning ability.

## Why Nemotron-3 Nano 4B

The monitoring Jetson has 8 GB of unified memory. The agent stack needs an LLM, an embedding model, a cross-encoder re-ranker, a FAISS index, and a Python runtime — all within that budget.

Nemotron-3 Nano 4B is a hybrid architecture with 42 layers, but only 4 of them are transformer attention layers (at positions 12, 17, 24, and 32). The remaining 38 are Mamba2 selective state space layers. The Mamba2 layers maintain a fixed-size state regardless of sequence length, so their memory use does not grow with context.

The measured memory allocation from llama.cpp:

| Component | Size | Notes |
|-----------|------|-------|
| KV cache (4 attention layers @ 16K) | 256 MiB | K: 128 MiB, V: 128 MiB |
| Mamba2 recurrent state (38 layers) | 324 MiB | Fixed — does not grow with context |
| Compute buffers | ~318 MiB | GPU + host |

A pure transformer with 42 attention layers at the same dimensions would need roughly 2.6 GB of KV cache alone with a 16K context window. That would leave no room for the model weights, let alone an embedding model and re-ranker.

In Q4_K_M quantization, the model file is about 2.8 GB on disk. In practice, that was enough to run a 16K context window together with the embedding model (BAAI/bge-small-en-v1.5, ~50 MB), re-ranker (Xenova/ms-marco-MiniLM-L-6-v2, ~80 MB), FAISS index, and Python runtime within the Orin Nano's 8 GB memory budget.

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

**Log search sub-agent** knows the three log formats (ISO timestamps, syslog, dmesg seconds-since-boot) and the correct shell commands to filter each. Given a time window, it computes cutoffs, searches all files, and deduplicates repeated errors with counts and time ranges.

**Manual consultant sub-agent** searches the knowledge base using two-stage retrieval: FAISS with dense embeddings (BAAI/bge-small-en-v1.5) for broad recall, then a cross-encoder re-ranker (Xenova/ms-marco-MiniLM-L-6-v2, 80 MB) for precision. Returns severity, root causes, and recommended actions.

## Synthetic Log Scenario

`gen_logs.py` generates 24 hours of realistic logs anchored to the current time, with a boot sequence, two incidents, and background noise:

**Boot sequence** — kernel init, GPU detection, NVMe identification, thermal daemon start, TensorRT engine loading, pipeline connections.

**Incident 1 (~45 min ago):** GPU memory spike to 90%, CUDA unified memory thrashing with 340ms latency spike, self-resolved by TensorRT memory pool compaction.

**Incident 2 (~30 min ago):** Thermal throttle cascade:
1. Third pipeline (camera-03) starts, GPU utilization hits 95%
2. Temperature rises rapidly — fan ramps to 100%
3. GPU thermal throttle at 70.5°C — clocks drop 918 → 624 MHz
4. Inference deadline misses cascade across all three pipelines
5. Deep throttle at 73°C — clocks further to 420 MHz
6. Inference queue fills (32/32) — pipeline stalls
7. Load shedding: camera-03 suspended, TensorRT engine unloaded
8. Recovery over 30 seconds — clocks restore, pipelines stabilize
9. Camera-03 restarts after 60 seconds

**Background noise** — scattered NVMe I/O errors, RTSP stream hiccups, GPU memory 85% warnings, NTP syncs.

Three log files:

| File | Timestamp format | Levels |
|------|-----------------|--------|
| `app.log` | ISO (`2026-03-24T17:00:00.000Z`) | `level=ERROR/WARN/INFO` |
| `thermal.log` | Syslog (`Mar 24 17:00:00`) | `ERROR/WARN/INFO` |
| `dmesg.log` | Seconds since boot (`[82810.000000]`) | `CRITICAL/WARNING` |

## Knowledge Base

The `docs/` directory is the knowledge base — all Markdown files are chunked by `##` headings, embedded, and indexed with FAISS. A cross-encoder re-ranker improves retrieval precision at query time.

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
| OOM Killer — Process Terminated | Critical |
| NVMe I/O Error — Storage Fault | High |
| RTSP Stream Hiccup | Low |

Updating any doc in `docs/` changes what the agent knows — no code changes needed. The FAISS index is rebuilt automatically when any doc changes.

## Performance

Measured from llama.cpp server logs during a real investigation:

| Metric | Value |
|--------|-------|
| Model | Nemotron-3 Nano 4B Q4_K_M |
| Context window | 16,384 tokens |
| Generation speed | ~15.5 tokens/sec (consistent across all calls) |
| Prompt eval speed | 135–400 tokens/sec (varies with cache hits) |
| Log search shell calls | 5–8 seconds each |
| Log search summarization | ~2.5 minutes (2,423 tokens generated) |
| Manual lookup | ~16 seconds |
| Main agent final response | ~48 seconds (722 tokens generated) |
| Full investigation wall time | ~6 minutes end to end |

## Prerequisites

**Hardware:**
- NVIDIA Jetson Orin Nano (8 GB)
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

This creates swap, installs deps, downloads all models, and builds the index — all locally.

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
2. Creates 8 GB swap on SSD (prevents OOM with LLM + embeddings)
3. Creates Python venv with uv
4. Installs dependencies (langchain-openai, langgraph, fastembed, faiss-cpu)
5. Downloads all models: Nemotron-3 Nano 4B GGUF (~2.8 GB) + ONNX embedder + re-ranker (~130 MB)
6. Builds FAISS index from `docs/`

### Manual setup (step by step)

```bash
cd /ssd/jetson-log-agent
make swap             # 8 GB swapfile on SSD (idempotent, persistent)
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

This generates fresh logs, rebuilds the FAISS index if any doc changed, and launches the agent inside a bubblewrap sandbox.

```
Jetson Log Agent ready. Type your question (or 'quit' to exit).

You: check last hour
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
3. Sends a structured email with error details and recommended actions from the manual (yellow `[email]`)
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

Since the agent runs shell commands against log files, sandboxing is a first-class requirement. The agent runs inside a [bubblewrap](https://github.com/containers/bubblewrap) sandbox:

| Mount | Mode | Purpose |
|-------|------|---------|
| /usr, /bin, /lib, /etc, /sys/devices/system/cpu | Read-only | System libraries, config, CPU info |
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
| `make setup` | Either | Auto-detects Jetson; runs locally or deploys via SSH |
| `make deploy` | Dev machine | Rsync project to Jetson |
| `make swap` | Jetson | Create 8 GB SSD swapfile (idempotent) |
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
