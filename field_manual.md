# Jetson Orin Nano — Incident Field Manual

Operational runbook for the Jetson Orin Nano inference cluster.

## Investigation Procedure

This procedure applies only when asked to investigate, check, or take action on logs. For other questions, answer directly without running the full procedure.

Follow these steps in order when investigating logs:

1. **Search logs** — call `search_logs` with the requested time window. It handles timestamp parsing for all three log formats.
2. **Look up every error** — for each distinct error found, call `consult_manual` with the log message to get root causes and recommended actions from this manual.
3. **Escalate critical issues** — after looking up remediation, call `send_email` to ops-team@company.com for any Critical or High severity issues. Include the error details AND the recommended actions from the lookup in the email body.
4. **Summarize** — present findings with evidence, root cause chain, and recommended actions.

**Reboot policy:**
- Call `reboot_device` only as a last resort — after thermal throttle persists through multiple recovery cycles, or if the scheduler is stuck and pipelines cannot restart.
- Always email ops BEFORE rebooting.
- Never reboot for transient issues (single deadline miss, RTSP hiccup, brief thermal spike).

**Severity guide:**
- **Critical** — thermal throttle, pipeline suspended, queue full. Email immediately.
- **High** — deadline miss, frame drops, power budget, degraded mode. Email if sustained.
- **Medium** — latency degradation, GPU memory warning. Monitor, no email needed.
- **Low** — RTSP hiccup. No email unless frequent.

## Thermal Throttle — GPU Over-Temperature

**Severity:** Critical

GPU temperature exceeded the 70°C trip point, triggering DVFS (Dynamic Voltage and Frequency Scaling) to reduce clock speeds from 918 MHz to 624 MHz. Inference throughput drops immediately.

**Log signatures:**
- thermal.log: `ERROR: GPU thermal throttle triggered: gpu=XX.XC exceeds threshold=70.0C`
- app.log: `level=ERROR component=thermal msg="GPU thermal throttle activated"`
- dmesg.log: `tegra-thermal: CRITICAL: zone1(gpu) exceeded trip point 70000mC`

**Root causes:**
- Too many concurrent inference pipelines for current cooling capacity
- Blocked or degraded enclosure airflow
- Ambient temperature above operating spec (>35°C)
- Dried or missing thermal paste on heatsink

**Recommended actions:**
1. Physically inspect the enclosure — remove dust from fan blades, verify intake/exhaust are unobstructed
2. Check ambient temperature with `cat /sys/class/thermal/thermal_zone*/temp` — if ambient >35°C, improve room cooling or relocate unit
3. Lower TensorRT max batch size from 4 to 2 in the pipeline config to cut per-batch GPU time by ~40%
4. If recurring, reapply thermal paste (Noctua NT-H1 or similar, rated for embedded) on the GPU heatsink contact
5. Run `sudo jetson_clocks --show` to verify fan profile is set to `cool` not `quiet`

## Thermal Throttle — Sustained / Deep Throttle

**Severity:** Critical

Temperature remains above threshold after initial DVFS action. Clocks are reduced further (624 MHz → 420 MHz), causing severe inference throughput degradation.

**Log signatures:**
- thermal.log: `ERROR: GPU thermal throttle deepened: gpu=XX.XC still above threshold`
- thermal.log: `WARN: Fan at maximum speed, cooling capacity limited`
- dmesg.log: `nvgpu: gpu0: WARNING: severe clock reduction, inference throughput will be degraded`

**Root causes:**
- Initial throttle insufficient — workload still generating too much heat at 624 MHz
- Cooling system at maximum capacity (fan 100%, 5000 RPM)
- Likely too many pipelines active for the thermal envelope

**Recommended actions:**
1. Immediately suspend the lowest-priority pipeline via the scheduler API — do not wait for auto-shedding
2. Switch to NVPMODEL 15W profile (`sudo nvpmodel -m 1`) which caps clocks more gracefully than emergency DVFS
3. Run `tegrastats` and log 60 seconds to confirm temperatures are actually dropping after load shed
4. Schedule a 30-minute thermal soak test during maintenance window to find the sustainable pipeline count
5. If fan RPM at 100% duty is below 4500 (spec 5000), file a hardware RMA — the fan bearing may be failing

## Rapid Temperature Increase

**Severity:** High

Temperature is rising faster than 5°C per 5-second interval, indicating workload spike or cooling degradation.

**Log signatures:**
- thermal.log: `WARN: Rapid temperature increase detected: gpu delta=+X.XC/5s (threshold=5.0C/5s)`

**Root causes:**
- New pipeline just started, adding GPU load suddenly
- Fan failure or degraded airflow (fan speed not increasing proportionally)
- Burst of high-complexity inference frames

**Recommended actions:**
1. Alert on-call — this is a leading indicator; throttle typically follows within 5-10 seconds
2. If a new pipeline just launched, add a 10-second stagger delay in the scheduler config between pipeline starts
3. Correlate with fan duty in thermal.log — PWM should be ramping proportionally; if fan duty is flat while temp climbs, the tegra-thermal daemon may be hung (`systemctl restart nvfancontrol`)
4. Check `dmesg | grep -i fan` for fan controller errors — a stuck fan PWM signal causes rapid uncontrolled heating

## Power Budget Exceeded

**Severity:** High

Total GPU power draw exceeded the 15W power budget, triggering DVFS power limiting.

**Log signatures:**
- app.log: `level=WARN component=power msg="Power budget exceeded" current_draw=XX.XW budget=15.0W`
- dmesg.log: `tegra-pmc: power rail VDD_GPU: current=XXXXXmW, budget=15000mW, OVER BUDGET`

**Root causes:**
- Directly correlated with thermal throttle — high utilization drives both power and temperature
- Too many concurrent inference workloads
- GPU clock and utilization both at maximum

**Recommended actions:**
1. Check current power mode: `sudo nvpmodel -q` — if running MAXN (30W), switch to 15W mode (`sudo nvpmodel -m 1`)
2. Verify the power supply is rated for sustained load — Jetson Orin Nano requires 5V/4A USB-C or 9-20V barrel jack; underpowered supplies cause voltage sag under load
3. Run `sudo tegrastats --interval 1000` and watch the `VDD_GPU_SOC` and `VDD_CPU_CV` rails — identify which rail is over budget
4. If power events correlate with thermal throttle, the thermal issue is primary — fix cooling first, power will follow

## Inference Deadline Miss

**Severity:** High

Inference batch latency exceeded the 100 ms deadline. Downstream consumers (display, recording, alerting) will see stale or missing frames.

**Log signatures:**
- app.log: `level=ERROR component=inference msg="Inference deadline missed" latency_ms=XXX deadline_ms=100`

**Root causes:**
- GPU thermal throttle reduced clocks, increasing latency
- Inference queue backlog — too many batches queued
- TensorRT engine contention between pipelines
- Occasional: large/complex frames taking longer to process

**Recommended actions:**
1. Cross-reference thermal.log — if clocks are reduced (624/420 MHz), this is a thermal symptom, not an inference issue; fix the throttle first
2. If no throttle present, profile the TensorRT engine: `trtexec --loadEngine=/models/yolov8n.engine --batch=4` to get baseline latency; if baseline >80ms, the model needs optimization (FP16, pruning, or smaller input resolution)
3. Reduce input resolution from 640x640 to 416x416 on the affected pipeline — this typically halves inference time
4. If only one pipeline is affected, check for frame-size outliers (e.g., 4K RTSP stream not being downscaled before inference)
5. Tune `deadline_ms` per-pipeline: camera-03 (non-critical) can tolerate 200ms; camera-01 (security) should stay at 100ms

## Inference Latency Degradation

**Severity:** Medium

Inference latency is increasing above the expected 40 ms maximum but hasn't yet missed the 100 ms deadline. This is an early warning.

**Log signatures:**
- app.log: `level=WARN component=inference msg="Inference latency increasing" latency_ms=XX expected_max_ms=40`

**Root causes:**
- Temperature rising, GPU approaching throttle threshold
- GPU memory pressure causing cache thrashing
- New pipeline just loaded, competing for GPU resources

**Recommended actions:**
1. No immediate action — this is an early warning. Set a 60-second watch: `watch -n5 'grep "latency_ms" app.log | tail -5'`
2. Check temperature trend: `grep "Zone readings" thermal.log | tail -10` — if gpu_temp_c is climbing >2°C per reading, a throttle is imminent
3. If the latency increase coincides with a new pipeline starting, it will stabilize in 10-15 seconds as TensorRT warms up
4. If sustained >60 seconds without throttle, check for GPU memory pressure: high memory usage causes TensorRT to spill to system memory, increasing latency

## Frame Drops — Inference Backpressure

**Severity:** High

Frames are being dropped because the inference pipeline cannot keep up. Downstream consumers receive gaps in the video stream.

**Log signatures:**
- app.log: `level=WARN component=inference msg="Frame drop detected" dropped_frames=XX reason="inference backpressure"`

**Root causes:**
- Inference latency exceeds frame production rate
- Usually follows deadline misses — the pipeline is already behind
- Queue depth at or near maximum

**Recommended actions:**
1. Check the `dropped_frames` count trend — if escalating (8 → 24 → ...), the system is in a death spiral and load shedding is needed now
2. Configure frame skipping on the RTSP decoder: set `framerate=15` instead of 30 to halve the incoming frame rate while maintaining coverage
3. If drops are isolated to one pipeline (e.g., camera-03), check its stream resolution — a 4K stream produces 4x more data than 1080p
4. Verify the inference queue depth: `grep "queue_depth" app.log | tail -5` — if approaching 32/32, the drops are from queue saturation, not individual frame latency

## Inference Queue Full — Pipeline Stall

**Severity:** Critical

The inference job queue has reached its maximum capacity (32/32). No new frames can be submitted until existing jobs complete. This causes a complete pipeline stall.

**Log signatures:**
- app.log: `level=ERROR component=inference msg="Pipeline stall: camera-XX inference queue full" queue_depth=32 max_queue=32`

**Root causes:**
- Sustained low throughput from thermal throttle or deep throttle
- All 32 queue slots filled because jobs complete slower than frames arrive
- Usually a cascading effect: throttle → slow inference → queue fills → stall

**Recommended actions:**
1. Emergency: the scheduler should auto-shed within seconds, but if it doesn't, manually suspend a pipeline: `curl -X POST localhost:8081/api/pipelines/camera-03/suspend`
2. Do NOT increase `max_queue` above 32 — it will just delay the stall while consuming more GPU memory for buffered frames
3. After recovery, add a queue high-water-mark alert at 24/32 (75%) so you get warning before hitting the hard limit
4. Consider switching from synchronous to asynchronous inference: let the RTSP decoder drop frames at the source rather than queuing them all

## Pipeline Degraded Mode

**Severity:** High

A pipeline has entered degraded mode after sustained deadline misses (4+ consecutive). The scheduler is considering load shedding.

**Log signatures:**
- app.log: `level=ERROR component=scheduler msg="Pipeline camera-XX entering degraded mode" reason="sustained deadline misses (4 consecutive)"`

**Root causes:**
- Sustained inference deadline misses on this pipeline
- GPU cannot service all active pipelines at required throughput
- Usually the last step before pipeline suspension

**Recommended actions:**
1. You have seconds before the scheduler suspends this pipeline — if it's high-priority (e.g., camera-01 security feed), immediately suspend a lower-priority pipeline to free GPU headroom
2. Check the pipeline priority config in `/etc/jetson-agent/pipelines.yaml` — ensure critical feeds have `priority: 1` so they survive load shedding
3. Grep for deadline misses on other pipelines — if all pipelines are degraded, the GPU is globally overloaded and shedding one won't be enough
4. Post-recovery: add a `max_consecutive_misses: 2` alert threshold (before the scheduler's 4-miss trigger) for earlier warning

## Pipeline Suspended — Load Shedding

**Severity:** Critical

The scheduler has suspended a pipeline to shed load. The TensorRT engine is unloaded and the RTSP stream is disconnected. This pipeline is completely offline.

**Log signatures:**
- app.log: `level=WARN component=scheduler msg="Shedding load: suspending pipeline camera-XX" reason="thermal_throttle + deadline_miss"`
- app.log: `level=ERROR component=stream msg="RTSP stream disconnected" reason="pipeline suspended"`
- app.log: `level=INFO component=inference msg="TensorRT engine unloaded for pipeline camera-XX"`

**Root causes:**
- Combination of thermal throttle and sustained deadline misses
- The scheduler determined this pipeline should be sacrificed to save others
- Load shedding is a last-resort automated recovery mechanism

**Recommended actions:**
1. This is the scheduler's last-resort protection — do NOT immediately restart the pipeline, it will re-trigger the throttle cascade
2. Wait for clocks to restore to 918 MHz (check `grep "clocks restored" thermal.log`) before allowing restart — typically 30-60 seconds
3. After auto-restart, verify the pipeline is healthy: `grep "restored to normal mode" app.log` — if this doesn't appear within 90 seconds, the scheduler may be stuck; restart it: `systemctl restart jetson-inference-scheduler`
4. Log the incident duration and cause in the ops runbook — if pipeline suspensions happen >2x per day, the system is under-provisioned for the current workload
5. Consider splitting workloads across two Jetson units if the site requires 3+ concurrent streams reliably

## GPU Memory Warning

**Severity:** Medium

GPU memory usage has reached 85% of total capacity (6.8 GB / 8.0 GB). Further allocations may fail or cause swapping.

**Log signatures:**
- app.log: `level=WARN component=metrics "GPU memory_used=6.8/8.0GB (85%) -- approaching limit"`

**Root causes:**
- Multiple TensorRT engines loaded simultaneously
- Memory fragmentation after repeated engine load/unload cycles
- Model or batch size too large for available memory

**Recommended actions:**
1. Count loaded TensorRT engines: `grep "TensorRT engine loaded" app.log | grep -v "unloaded" | wc -l` — each uses ~210 MB; 3 engines = 630 MB + framework overhead
2. If only 2 pipelines are active but memory is 85%, check for a stale engine from a previously suspended pipeline that wasn't properly unloaded: `grep "unloaded" app.log | tail -5`
3. Enable TensorRT engine caching (`trt_engine_cache=true`) to avoid re-allocating memory on pipeline restarts
4. Monitor with `tegrastats` — watch the `GR3D` (GPU) and `RAM` fields; if RAM keeps growing between readings with no new pipelines, there's a leak in the inference wrapper
5. If running yolov8s (22MB model), switch to yolov8n (6MB) — the engine memory footprint drops from ~350 MB to ~210 MB

## RTSP Stream Hiccup

**Severity:** Low

An RTSP camera stream experienced a brief interruption and reconnected. Frames may have been lost during the reconnect window.

**Log signatures:**
- app.log: `level=WARN component=stream msg="RTSP stream hiccup" reconnect_ms=XXX`

**Root causes:**
- Network instability between Jetson and IP camera
- Camera firmware issues or periodic key-frame generation stalls
- Switch/router congestion on the camera VLAN

**Recommended actions:**
1. If `reconnect_ms` < 500 and frequency < 2/hour, this is normal for IP cameras — no action needed
2. If `reconnect_ms` > 2000, the camera may be rebooting or overheating — check camera's web UI for uptime and internal temp
3. For frequent hiccups (>5/hour), run `ping -c 100 192.168.1.50` and check for packet loss — any loss >1% indicates a network issue (bad cable, switch port flapping, PoE budget exceeded)
4. Check the PoE switch power budget: if the switch is at capacity, cameras will brown-out under load; verify with `show power inline` on the switch
5. Set RTSP transport to TCP (`rtsp_transport=tcp` in pipeline config) instead of UDP — TCP retransmits dropped packets automatically, at the cost of slightly higher latency
