# Jetson Log Investigation Agent

A ReAct agent that uses shell commands to investigate hardware and inference
pipeline logs, sandboxed with bubblewrap. Designed for Jetson Orin Nano
running Nemotron-3 Nano 4B via `llama-server`.

## How it works

Single agent with one tool (`shell`) that runs read-only commands (grep, awk,
sed, head, tail, date, etc.) against synthetic log files. The agent figures out
what commands to run based on your question.

Logs are generated fresh each run with timestamps anchored to now. The incident
(thermal throttle cascade from adding a 3rd camera pipeline) is placed ~2 hours
ago.

## Log files

- **app.log** — Inference pipeline supervisor: TensorRT, batch latency/fps,
  GPU utilization, memory, power, frame drops, deadline misses
- **thermal.log** — Tegra thermal daemon: thermal zones, fan PWM/RPM, DVFS
  throttle events
- **dmesg.log** — Kernel ring buffer: nvgpu driver, tegra-pmc power rails,
  thermal trip points

## Setup (Jetson Orin Nano, Ubuntu 22.04 / JetPack 6.x)

```bash
sudo apt install bubblewrap ripgrep
```

`llama-server` expected at `/opt/llama.cpp/build/bin/llama-server`.

### From dev machine

```bash
make deploy     # rsync to Jetson
make setup      # deploy + install + download model
```

### On the Jetson

```bash
cd /ssd/jetson-log-agent
make server     # terminal 1: starts llama-server (auto-downloads model)
./launch.sh     # terminal 2: gen logs + run agent in sandbox
```

Or without sandbox: `make run`

### Environment variables

| Variable | Default | Description |
|---|---|---|
| `OPENAI_BASE_URL` | `http://127.0.0.1:8080/v1` | llama-server endpoint |
| `OPENAI_API_KEY` | `not-needed` | API key (any string for local) |
| `OPENAI_MODEL` | `local-model` | Model name (informational) |

## Example questions

```
any errors in past 3 hours?
what caused pipeline camera-03 to be suspended?
trace session sess-a1b2c3
what was the GPU temperature timeline during the incident?
how many inference deadline misses occurred?
did the fan reach maximum speed?
```

## Sandbox

Bubblewrap provides: read-only filesystem, PID/IPC/UTS namespace isolation,
clean environment, minimal /dev. Network is NOT isolated (agent needs
localhost to reach llama-server); shell tool blocks networking commands.
