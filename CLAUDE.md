# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

LangChain ReAct agent for investigating hardware and inference pipeline logs on a Jetson Orin Nano. Single agent with a `shell` tool, connects to a local `llama-server` (`/opt/llama.cpp/build/bin`) running Nemotron-3 Nano 4B via `langchain-openai`. Agent process sandboxed with bubblewrap. Python managed with `uv`.

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
make run                # Gen logs + run agent (terminal 2)
./launch.sh             # Gen logs + run agent in bwrap sandbox
```

## Architecture

`run_agent.py` — single agent with one `shell` tool:
- `shell` executes allowlisted read-only commands (grep, awk, sed, head, tail, date, etc.)
- Command validation: allowlist + dangerous pattern regex + timeout + output cap
- `create_react_agent` from `langgraph.prebuilt` with `ChatOpenAI` pointing at local llama-server
- Interactive CLI loop with streaming output, `<think>` blocks shown dimmed

`gen_logs.py` — generates 24h of synthetic logs anchored to current time:
- Incident (thermal throttle cascade) placed ~2h ago
- Normal 5s ticks + occasional misc events fill the rest

`launch.sh` — regenerates logs, then execs into bwrap sandbox with read-only mounts and clean env.

`demo_logs/` — generated files:
- `app.log` — inference pipeline supervisor (TensorRT, latency, fps, GPU metrics)
- `thermal.log` — tegra thermal daemon (temperatures, fan, DVFS throttle)
- `dmesg.log` — kernel ring buffer (nvgpu driver, power rails, thermal trip points)
