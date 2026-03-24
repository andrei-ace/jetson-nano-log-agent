# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LangChain ReAct agent for investigating hardware and inference pipeline logs on a Jetson Orin Nano. Two-agent architecture: main investigator + manual consultant sub-agent. Connects to a local `llama-server` (`/opt/llama.cpp/build/bin`) running Nemotron-3 Nano 4B via `langchain-openai`. Agent process sandboxed with bubblewrap. Python managed with `uv`.

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

`field_manual.md` — Markdown incident runbook, source for RAG knowledge base.

`build_index.py` — offline script: chunks `field_manual.md` by `##` headings, embeds with fastembed (BAAI/bge-small-en-v1.5), writes `kb_index/index.faiss` + `kb_index/chunks.json`. Run once or when field manual changes; `launch.sh` auto-rebuilds if stale.

`gen_logs.py` — generates 24h of synthetic logs anchored to current time:
- Incident (thermal throttle cascade) placed ~2h ago
- Normal 5s ticks + occasional misc events fill the rest

`launch.sh` — regenerates logs, then execs into bwrap sandbox with read-only mounts and clean env.

`demo_logs/` — generated files:
- `app.log` — inference pipeline supervisor (TensorRT, latency, fps, GPU metrics)
- `thermal.log` — tegra thermal daemon (temperatures, fan, DVFS throttle)
- `dmesg.log` — kernel ring buffer (nvgpu driver, power rails, thermal trip points)
