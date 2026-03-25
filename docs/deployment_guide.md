# Jetson Orin Nano — Deployment Guide

Standard operating procedures for deploying, updating, and maintaining Jetson Orin Nano inference nodes in the field.

## Site Requirements

- **Power:** 20W sustained per unit. Use a UPS rated for at least 2x the node count — a 10-node rack needs a 400W UPS minimum. PoE is supported but requires 802.3at (25.5W) switches, not 802.3af.
- **Cooling:** Minimum 50 CFM airflow per unit. Do not stack units without 1U spacing. Intake temperature must stay below 35°C.
- **Network:** Gigabit Ethernet, dedicated VLAN for camera traffic (RTSP). Management VLAN for SSH. QoS must prioritize RTSP over bulk traffic.
- **Storage:** 250 GB NVMe minimum. Industrial-grade preferred for 24/7 operation (Samsung PM9A3 or Kioxia XG8).

## Initial Provisioning

### 1. Flash JetPack

Flash JetPack 6.x using NVIDIA SDK Manager from a Linux host. Use the default partition layout. After first boot:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install bubblewrap socat
```

### 2. Build llama.cpp

```bash
git clone https://github.com/ggerganov/llama.cpp /opt/llama.cpp
cd /opt/llama.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build -j$(nproc)
```

Verify: `/opt/llama.cpp/build/bin/llama-server --version`

### 3. Configure NVMe

If the SSD is not yet mounted:

```bash
sudo mkfs.ext4 /dev/nvme0n1p1
sudo mkdir /ssd
echo '/dev/nvme0n1p1 /ssd ext4 defaults,noatime 0 2' | sudo tee -a /etc/fstab
sudo mount /ssd
```

### 4. Deploy the agent

From the dev machine:
```bash
make setup   # deploys code, creates swap, installs deps, downloads model
```

### 5. Configure systemd services

Create `/etc/systemd/system/llama-server.service`:
```ini
[Unit]
Description=llama.cpp inference server
After=network.target

[Service]
Type=simple
User=inference
ExecStart=/opt/llama.cpp/build/bin/llama-server \
    --model /ssd/jetson-log-agent/models/NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf \
    --port 8080 --n-gpu-layers -1 --ctx-size 16384 --reasoning-format none
Restart=on-failure
RestartSec=5
OOMScoreAdjust=-500

[Install]
WantedBy=multi-user.target
```

The `OOMScoreAdjust=-500` makes the OOM killer prefer other processes over llama-server.

Enable: `sudo systemctl enable --now llama-server`

## Pipeline Configuration

Pipeline definitions live in `/etc/jetson-agent/pipelines.yaml`:

```yaml
pipelines:
  camera-01:
    stream: rtsp://192.168.1.50/cam1
    model: /models/yolov8n.engine
    priority: 1        # highest — survives load shedding
    deadline_ms: 100
    max_batch: 4

  camera-02:
    stream: rtsp://192.168.1.51/cam2
    model: /models/yolov8n.engine
    priority: 2
    deadline_ms: 100
    max_batch: 4

  camera-03:
    stream: rtsp://192.168.1.52/cam3
    model: /models/yolov8n.engine
    priority: 3        # lowest — first to be shed
    deadline_ms: 200   # relaxed deadline for non-critical feed
    max_batch: 2       # reduced batch to limit GPU pressure
```

Priority 1 pipelines are never shed. Priority 3 pipelines are shed first during thermal throttle.

## Updating the Agent

From the dev machine:
```bash
make deploy   # rsync code changes to Jetson
```

If `field_manual.md` or any doc in `docs/` changed, the FAISS index rebuilds automatically on next `./launch.sh`.

If Python dependencies changed, SSH into the Jetson and run `make install`.

## Monitoring

### Real-time telemetry

```bash
sudo tegrastats --interval 1000
```

Fields to watch:
- `RAM`: total memory usage — alert at 7 GB
- `GR3D_FREQ`: GPU clock — drops below 918 MHz indicate throttling
- `Temp PLL@XX.XC CPU@XX.XC GPU@XX.XC`: thermal zones — alert at 65°C
- `VDD_GPU_SOC XXXX/XXXX`: current/average power in mW

### Log rotation

Logs at `/ssd/jetson-log-agent/demo_logs/` are regenerated on each agent launch. In production, configure logrotate:

```
/var/log/jetson-inference/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
}
```

### Health checks

Add to crontab (`crontab -e`):
```
*/5 * * * * curl -sf http://127.0.0.1:8080/health || systemctl restart llama-server
```

## Decommissioning

Before removing a unit from the field:

1. Suspend all pipelines: `curl -X POST localhost:8081/api/pipelines/all/suspend`
2. Stop services: `sudo systemctl stop llama-server`
3. Securely wipe the SSD: `sudo blkdiscard /dev/nvme0n1` (instant on NVMe, erases all data)
4. Remove from fleet inventory
