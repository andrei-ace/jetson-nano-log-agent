# Jetson Orin Nano — Known Issues and Workarounds

Documented quirks, bugs, and workarounds for the Jetson Orin Nano platform running JetPack 6.x with llama.cpp and TensorRT inference.

## JI-001: nvgpu Driver Hang After Deep Throttle Recovery

**Status:** Open (JetPack 6.0, 6.1)

After a deep thermal throttle event (clocks reduced to 420 MHz), the GPU occasionally fails to restore clocks to 918 MHz even after temperatures return to normal. The `nvgpu` driver reports `gpcclk` stuck at 624 MHz and `tegrastats` shows reduced GPU utilization despite no thermal pressure.

**Symptoms:**
- `tegrastats` shows `GR3D_FREQ 624MHz` when `Temp GPU@50.0C` (well below throttle threshold)
- Inference latency remains elevated (60-80 ms instead of 25-35 ms)
- `dmesg` shows no errors — the driver thinks clocks are restored

**Workaround:**
```bash
# Force clock reset
sudo jetson_clocks
sleep 2
sudo jetson_clocks --restore
```

If that doesn't work, restart the nvgpu driver:
```bash
sudo rmmod nvgpu && sudo modprobe nvgpu
```

This kills all GPU contexts — all TensorRT engines and llama-server must be restarted.

**Root cause:** Race condition in the `tegra-thermal` DVFS handler when two throttle levels are released in rapid succession (420→624→918). The second release event is sometimes dropped.

## JI-002: TensorRT Engine Memory Not Freed After Pipeline Suspension

**Status:** Confirmed (TensorRT 8.6, 10.x)

When a pipeline is suspended via load shedding, the TensorRT engine is unloaded but the CUDA memory allocation is not always released back to the pool. `tegrastats` shows the memory as still allocated even though no engine is using it.

**Symptoms:**
- After suspending camera-03, expected memory drop of ~210 MB does not appear
- `nvidia-smi` (or `tegrastats` RAM field) shows no change
- Restarting the pipeline allocates *additional* 210 MB instead of reusing the stale allocation

**Workaround:**
Restart the inference supervisor process to force a clean CUDA context teardown:
```bash
systemctl restart jetson-inference-supervisor
```

**Note:** This interrupts all active pipelines for ~2 seconds during restart.

**Root cause:** TensorRT's `IRuntime::deserializeCudaEngine()` allocates from a sub-pool. When the engine is destroyed, the sub-pool is freed, but the parent pool retains the pages. This is by design for performance (avoids repeated `cudaMalloc`) but is problematic when memory is scarce.

## JI-003: RTSP Stream Reconnect Causes Frame Duplication

**Status:** Low priority

When an RTSP stream disconnects and reconnects, the first 2-3 frames after reconnection are duplicates of the last pre-disconnect frame. This is because the RTSP client's jitter buffer replays stale packets.

**Impact:** Minimal — the inference pipeline processes duplicate frames but the results are discarded by the downstream dedup filter. However, this inflates the `frames_processed` counter, making throughput metrics slightly inaccurate.

**Workaround:** No fix needed. If accurate frame counting is required, add a frame hash check:
```python
if frame_hash == prev_frame_hash:
    metrics.duplicate_frames += 1
    continue
```

## JI-004: llama-server Prompt Cache Grows Unbounded

**Status:** Monitoring

The llama-server's prompt cache (`--cache-type-k f16`) grows proportionally to the number of unique prompt prefixes seen. In a multi-turn conversation, each turn creates a new cache entry. After ~50 turns, the cache can consume 200+ MB of RAM.

**Symptoms:**
- llama-server's RSS grows over time
- Eventually triggers OOM killer if swap is insufficient
- Server logs show `prompt_save: saving prompt with length XXXX, total state size = XXX MiB`

**Workaround:**
- Restart llama-server periodically (every 100 conversations) via the health check cron job
- Use `--cache-type-k q4_0` instead of `f16` to reduce cache memory by 4x at the cost of slight quality degradation
- Set `--slots 1` to limit concurrent conversations (reduces cache diversity)

## JI-005: Fan PWM Stuck at Boot Value After nvfancontrol Restart

**Status:** Fixed in JetPack 6.2

If `nvfancontrol` is restarted while the fan is at high duty (e.g., 90%), the daemon reads the current duty as the "default" and refuses to ramp down below it.

**Workaround (JetPack 6.0, 6.1):**
```bash
# Manually reset fan to auto before restarting the daemon
echo 0 | sudo tee /sys/devices/pwm-fan/hwmon/hwmon*/automatic
echo 1 | sudo tee /sys/devices/pwm-fan/hwmon/hwmon*/automatic
sudo systemctl restart nvfancontrol
```

## JI-006: CUDA Unified Memory Thrashing Under High GPU Memory Pressure

**Status:** By design (JetPack 6.x)

When GPU memory usage exceeds ~85% on the Orin Nano (which uses unified memory), the CUDA driver begins migrating pages between CPU and GPU address spaces. This manifests as sudden latency spikes (200-500 ms) on inference that are not correlated with thermal throttle.

**Symptoms:**
- Intermittent latency spikes with no thermal events
- `tegrastats` shows `GR3D_FREQ` at full clock and `Temp GPU` normal
- Spikes correlate with `RAM` field approaching 7+ GB

**Workaround:**
- Keep total memory usage below 6.5 GB (81%) to avoid the migration threshold
- Reduce LLM context window (`--ctx-size 8192` instead of 16384) to free ~200 MB
- Use `CUDA_MANAGED_FORCE_DEVICE_ALLOC=1` to pin managed allocations on GPU (reduces thrashing but may trigger OOM earlier)
