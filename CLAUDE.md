# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LangChain ReAct agent for investigating hardware and inference pipeline logs on a Jetson Orin Nano. Three-agent architecture: main router + log search sub-agent + manual consultant sub-agent. Connects to a local `llama-server` (`/opt/llama.cpp/build/bin`) running Nemotron-3 Nano 4B via `langchain-openai`. Agent process sandboxed with bubblewrap. Python managed with `uv`.

## Build & Run

```bash
# From dev machine
make deploy             # Rsync to Jetson
make setup              # Deploy + install + download model

# On Jetson (/ssd/jetson-log-agent)
make install            # Create venv, install deps
make download-model     # Download Nemotron-3 Nano 4B GGUF
make server             # Start llama-server (terminal 1)
make gen-logs           # Regenerate logs with current timestamps
make build-index        # Rebuild FAISS index from field_manual.md
./launch.sh             # Gen logs + build index (if needed) + run agent in bwrap sandbox
```

## Architecture

`run_agent.py` — three-agent architecture:
- **Main agent** (router): `search_logs`, `consult_manual`, `send_email`, `reboot_device` tools
- **Log search sub-agent**: has `shell` tool, knows all three log formats and timestamp filtering
- **Manual sub-agent**: has `search_manual` (RAG via FAISS + fastembed), reads full chunks, returns compact severity + actions
- All agents use `create_react_agent` from `langgraph.prebuilt` with `ChatOpenAI` pointing at local llama-server
- Interactive CLI loop with streaming output, `<think>` blocks shown dimmed

`docs/` — knowledge base source documents:
- `field_manual.md` — incident runbook (investigation procedure, 14 error sections, severity guide)
- `hardware_spec.md` — Orin Nano specs, thermal envelope, memory budget, power rails
- `deployment_guide.md` — provisioning, systemd services, pipeline config, monitoring
- `known_issues.md` — platform quirks with workarounds (nvgpu hangs, TensorRT leaks, CUDA thrashing)

`build_index.py` — chunks all `docs/*.md` by `##` headings, embeds with fastembed (BAAI/bge-small-en-v1.5), downloads cross-encoder re-ranker (Xenova/ms-marco-MiniLM-L-6-v2, 80MB), writes `kb_index/`. Auto-runs on first `./launch.sh` or when any doc changes.

`gen_logs.py` — generates 24h of synthetic logs anchored to current time:
- Boot sequence (kernel init, GPU detection, pipeline startup)
- Memory spike incident at ~45min ago (CUDA unified memory thrashing, self-resolved)
- Thermal throttle cascade at ~30min ago (full incident: throttle → deadline misses → pipeline suspension → recovery)
- Scattered NVMe I/O errors, RTSP hiccups, GPU memory warnings throughout

`launch.sh` — regenerates logs, sets up socat network bridge (sandbox can only reach llama-server), then execs into bwrap sandbox with read-only mounts, network isolation, and clean env. Requires `socat` and `bubblewrap`.

`demo_logs/` — generated files:
- `app.log` — inference pipeline supervisor (TensorRT, latency, fps, GPU metrics)
- `thermal.log` — tegra thermal daemon (temperatures, fan, DVFS throttle)
- `dmesg.log` — kernel ring buffer (nvgpu driver, power rails, thermal trip points)
