# Jetson Orin Nano — Hardware Reference

Specifications and operational limits for the NVIDIA Jetson Orin Nano 8GB (P3767-0005) deployed in the inference cluster.

## Compute

- **SoC:** NVIDIA Orin (Ampere GPU + Arm Cortex-A78AE CPU)
- **GPU:** 1024 CUDA cores, 32 Tensor cores, max 918 MHz (MAXN), thermal cap varies with DVFS
- **CPU:** 6-core Arm Cortex-A78AE, max 1.5 GHz
- **Memory:** 8 GB LPDDR5, 68 GB/s bandwidth, shared between CPU and GPU (unified memory)
- **DLA:** 2x NVDLA v2.0 engines (not used in current pipeline — TensorRT targets GPU only)

## Thermal Envelope

- **TDP:** 7W (default) / 15W (MAXN mode)
- **GPU thermal trip point:** 70°C — triggers DVFS clock reduction
- **GPU thermal hysteresis:** 5°C — throttle releases when temp drops below 65°C
- **Fan specs:** PWM-controlled, 5000 RPM maximum, target temp 65°C
- **Ambient operating range:** 0°C to 35°C (derate above 35°C)

The thermal cascade sequence under overload:
1. GPU reaches 70°C → clocks drop 918 → 624 MHz (first throttle)
2. If still above 70°C after 3s → clocks drop 624 → 420 MHz (deep throttle)
3. Fan duty ramps linearly: 40% at 45°C, 100% at 70°C
4. Recovery: clocks restore in reverse order once temp drops below hysteresis band

## Power Rails

- **VDD_GPU_SOC:** GPU + SoC power, budget 15W in MAXN mode
- **VDD_CPU_CV:** CPU cluster power
- **VDD_IN:** Total board input, 5V/4A USB-C or 9-20V DC barrel jack
- **PoE note:** If powered via PoE HAT, ensure the PoE switch can sustain 20W per port under load — 802.3af (15.4W) is insufficient for MAXN

Monitor power rails in real time:
```
sudo tegrastats --interval 1000
```

## NVPMODEL Power Profiles

| Mode | Name | GPU Max Clock | CPU Max Clock | TDP |
|------|------|--------------|--------------|-----|
| 0 | MAXN | 918 MHz | 1.5 GHz | 15W |
| 1 | 15W | 624 MHz | 1.4 GHz | 15W |
| 2 | 7W | 510 MHz | 1.0 GHz | 7W |

Check current mode: `sudo nvpmodel -q`
Switch mode: `sudo nvpmodel -m 1`

Mode 0 (MAXN) is required for 3-pipeline inference at 30 fps. Mode 1 sustains 2 pipelines. Mode 2 is single-pipeline only.

## Storage

- **eMMC:** 16 GB (JetPack OS, boot partition — do not use for models or logs)
- **NVMe SSD:** Samsung 970 EVO Plus 250GB at `/ssd` (model weights, swap, logs, project files)
  - Sequential read: 3500 MB/s
  - Sequential write: 1500 MB/s
  - Endurance: 150 TBW (at 40 GB/day write rate = ~10 years)
  - Thermal throttle: 70°C (add thermal pad if sandwiched near SoC)

SMART health check:
```
sudo nvme smart-log /dev/nvme0n1
```

Key fields: `percentage_used` (drive wear, >90% = replace), `media_errors` (non-zero = failing), `temperature` (>70°C = thermal pad needed).

## Memory Budget (8 GB shared)

Typical allocation at full load (3 pipelines + LLM agent):

| Component | Memory | Notes |
|-----------|--------|-------|
| JetPack OS + services | ~800 MB | systemd, NetworkManager, nvfancontrol |
| llama-server (Nemotron-3 Nano 4B Q4_K_M) | ~3.2 GB | Model weights + 16K KV cache |
| TensorRT engine x3 | ~630 MB | 210 MB per yolov8n pipeline |
| CUDA runtime + driver | ~400 MB | Shared across all GPU contexts |
| Python agent + fastembed | ~350 MB | Agent, ONNX Runtime, embedding model |
| FAISS index + chunks | ~5 MB | Negligible |
| **Total** | **~5.4 GB** | |
| **Remaining for swap/buffers** | **~2.6 GB** | OS file cache, tmpfs, headroom |

When the 4th process (e.g., a debug shell, tegrastats, or a leaked TensorRT context) pushes usage above ~7.5 GB, the OOM killer activates.

## Network

- **Ethernet:** Gigabit (RTL8168), IP `192.168.1.7` (static, VLAN 10)
- **Camera VLAN:** 192.168.1.0/24, ports 50-59 reserved for RTSP cameras
- **Management VLAN:** 192.168.2.0/24, SSH access
- **DNS:** local only (`jetson-07.local` via mDNS)

## GPU Context Limits

The Orin Nano supports up to 16 concurrent GPU contexts. Each TensorRT engine pipeline allocates one context. The llama-server allocates one context per active inference slot.

If context allocation fails:
```
nvgpu: gpu0: failed to allocate new context: out of contexts
```

This usually means a suspended pipeline's context wasn't properly released. Check with:
```
cat /sys/kernel/debug/nvgpu/gpu.0/allocations
```
