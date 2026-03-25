"""Generate realistic synthetic Jetson Orin Nano logs anchored to current time."""

import os
import random
from datetime import datetime, timedelta, timezone

OUT_DIR = os.environ.get("LOG_DIR", os.path.join(os.path.dirname(__file__), "demo_logs"))

# ---------------------------------------------------------------------------
# Time helpers
# ---------------------------------------------------------------------------

NOW = datetime.now(timezone.utc)
T_START = NOW - timedelta(hours=24)
T_MEMORY_SPIKE = NOW - timedelta(minutes=45)  # memory pressure builds
T_INCIDENT = NOW - timedelta(minutes=30)       # thermal cascade follows
BOOT_TIME = T_START - timedelta(seconds=3600)


def iso(t: datetime) -> str:
    return t.strftime("%Y-%m-%dT%H:%M:%S.") + f"{t.microsecond // 1000:03d}Z"


def syslog(t: datetime) -> str:
    return t.strftime("%b %d %H:%M:%S")


def dmesg_ts(t: datetime) -> str:
    secs = (t - BOOT_TIME).total_seconds()
    return f"[{secs:.6f}]"


# ---------------------------------------------------------------------------
# Normal operation generators
# ---------------------------------------------------------------------------

PIPELINES = ["camera-01", "camera-02"]
SKUS = ["yolov8n"]


def normal_app_tick(t: datetime, lines: list):
    """Generate ~5s of normal operation app log lines."""
    lines.append(f"{iso(t)} level=INFO node=jetson-07 component=supervisor "
                 f'msg="System monitor tick" uptime_s={int((t - BOOT_TIME).total_seconds())}')

    tt = t + timedelta(milliseconds=2)
    cpu = random.uniform(38, 48)
    gpu = cpu - random.uniform(1, 3)
    board = cpu - random.uniform(4, 7)
    lines.append(f"{iso(tt)} level=INFO node=jetson-07 component=thermal "
                 f'msg="Zone readings" cpu_temp_c={cpu:.1f} gpu_temp_c={gpu:.1f} board_temp_c={board:.1f}')

    for i, pipe in enumerate(PIPELINES):
        tt2 = t + timedelta(milliseconds=100 + i * 100)
        lat = random.randint(25, 35)
        fps = round(4000 / lat, 1)
        bid = random.randint(10000, 99999)
        lines.append(f"{iso(tt2)} level=INFO node=jetson-07 component=inference "
                     f'msg="Batch completed" model=yolov8n pipeline={pipe} '
                     f"batch_id=B-{bid} frames=4 latency_ms={lat} fps={fps}")

    tt3 = t + timedelta(seconds=2)
    gpu_util = random.randint(55, 68)
    mem = round(random.uniform(2.8, 3.2), 1)
    power = round(random.uniform(8.5, 10.0), 1)
    lines.append(f"{iso(tt3)} level=INFO node=jetson-07 component=metrics "
                 f'"GPU utilization={gpu_util}% memory_used={mem}/8.0GB '
                 f'power_draw={power}W clocks_gpu=918MHz clocks_mem=1600MHz"')


def normal_thermal_tick(t: datetime, lines: list):
    cpu = random.uniform(38, 48)
    gpu = cpu - random.uniform(1, 3)
    board = cpu - random.uniform(4, 7)
    aux = board - random.uniform(2, 4)
    fan = random.randint(40, 55)
    rpm = 2400 + fan * 30
    lines.append(f"{syslog(t)} jetson-07 tegra-thermal [1201] INFO: "
                 f"Reading thermal zones: cpu={cpu:.1f}C gpu={gpu:.1f}C "
                 f"board={board:.1f}C aux={aux:.1f}C")
    lines.append(f"{syslog(t)} jetson-07 tegra-thermal [1201] INFO: "
                 f"Fan PWM duty={fan}% rpm={rpm} target_temp=65.0C")


def normal_dmesg_tick(t: datetime, lines: list):
    lines.append(f"{dmesg_ts(t)} tegra-pmc: System power state: active, "
                 f"rail=VDD_GPU voltage=900mV")
    tt = t + timedelta(milliseconds=5)
    lines.append(f"{dmesg_ts(tt)} nvgpu: gpu0: clocks: gpcclk=918MHz, "
                 f"max=1300MHz, thermal_cap=918MHz")
    gpu = random.uniform(36, 46)
    tt2 = t + timedelta(milliseconds=15)
    lines.append(f"{dmesg_ts(tt2)} tegra-thermal: zone1(gpu): "
                 f"temp={int(gpu * 1000)}mC, trip=70000mC, hyst=5000mC")


# ---------------------------------------------------------------------------
# Incident generators (the thermal throttle cascade)
# ---------------------------------------------------------------------------


def generate_incident(t0: datetime, app: list, thermal: list, dmesg: list):
    """Generate the full incident: 3rd pipeline → thermal throttle → recovery."""
    t = t0

    # -- Pipeline camera-03 starts --
    app.append(f"{iso(t)} level=INFO node=jetson-07 component=scheduler "
               f'msg="Pipeline camera-03 starting" model=yolov8n '
               f"stream=rtsp://192.168.1.50/cam3 session=sess-a1b2c3")
    app.append(f"{iso(t + timedelta(milliseconds=20))} level=INFO node=jetson-07 "
               f'component=inference msg="Loading TensorRT engine for pipeline camera-03" '
               f"session=sess-a1b2c3 engine=/models/yolov8n.engine")

    t += timedelta(seconds=1)
    app.append(f"{iso(t)} level=INFO node=jetson-07 component=inference "
               f'msg="TensorRT engine loaded" session=sess-a1b2c3 '
               f"load_time_ms=580 device_memory_mb=210")
    dmesg.append(f"{dmesg_ts(t)} nvgpu: gpu0: new context allocated: "
                 f"pid=4890 comm=trt_inference size=214958080B")
    dmesg.append(f"{dmesg_ts(t)} nvgpu: gpu0: memory: "
                 f"used=3670016KB/8388608KB (43.7%) -- allocation +524288KB")

    t += timedelta(milliseconds=500)
    app.append(f"{iso(t)} level=INFO node=jetson-07 component=inference "
               f'msg="Batch completed" model=yolov8n pipeline=camera-03 '
               f"batch_id=B-90103 frames=4 latency_ms=30 fps=133.3 session=sess-a1b2c3")

    # -- Temperature rising (t0+3s to t0+8s) --
    for dt, cpu, gpu_t, fan, gpu_util, power in [
        (3, 51.0, 49.5, 65, 78, 11.4),
        (5, 58.5, 56.0, 80, 91, 13.8),
        (7, 65.0, 63.5, 95, 95, 14.9),
    ]:
        tt = t0 + timedelta(seconds=dt)
        board = cpu - 9
        aux = board - 3

        thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] INFO: "
                       f"Reading thermal zones: cpu={cpu:.1f}C gpu={gpu_t:.1f}C "
                       f"board={board:.1f}C aux={aux:.1f}C")
        if gpu_t > 55:
            thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] WARN: "
                           f"Rapid temperature increase detected: gpu delta=+{random.uniform(5, 9):.1f}C/5s "
                           f"(threshold=5.0C/5s)")
        thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] INFO: "
                       f"Fan PWM duty={fan}% rpm={2400 + fan * 30} target_temp=65.0C")

        app.append(f"{iso(tt)} level=INFO node=jetson-07 component=thermal "
                   f'msg="Zone readings" cpu_temp_c={cpu} gpu_temp_c={gpu_t} board_temp_c={board:.1f}')

        dmesg.append(f"{dmesg_ts(tt)} tegra-thermal: zone1(gpu): "
                     f"temp={int(gpu_t * 1000)}mC, trip=70000mC, hyst=5000mC")

        app.append(f"{iso(tt + timedelta(milliseconds=200))} level=INFO node=jetson-07 "
                   f'component=metrics "GPU utilization={gpu_util}% memory_used=3.6/8.0GB '
                   f'power_draw={power}W clocks_gpu=918MHz clocks_mem=1600MHz"')

        if gpu_t > 55:
            for i, pipe in enumerate(["camera-01", "camera-02", "camera-03"]):
                lat = random.randint(40, 70) if pipe != "camera-03" else random.randint(50, 90)
                fps = round(4000 / lat, 1)
                extra = " session=sess-a1b2c3" if pipe == "camera-03" else ""
                app.append(f"{iso(tt + timedelta(milliseconds=300 + i * 100))} level=WARN "
                           f"node=jetson-07 component=inference "
                           f'msg="Inference latency increasing" pipeline={pipe} '
                           f"batch_id=B-{random.randint(90120, 90199)} frames=4 "
                           f"latency_ms={lat} fps={fps}{extra} expected_max_ms=40")

    if gpu_t >= 63:
        tt7 = t0 + timedelta(seconds=7)
        thermal.append(f"{syslog(tt7)} jetson-07 tegra-thermal [1201] WARN: "
                       f"Approaching throttle threshold: gpu=63.5C threshold=70.0C (6.5C headroom)")

    # -- Throttle at t0+10s --
    t_throttle = t0 + timedelta(seconds=10)

    thermal.append(f"{syslog(t_throttle)} jetson-07 tegra-thermal [1201] ERROR: "
                   f"GPU thermal throttle triggered: gpu=70.5C exceeds threshold=70.0C")
    thermal.append(f"{syslog(t_throttle)} jetson-07 tegra-thermal [1201] INFO: "
                   f"DVFS action: gpu clocks 918MHz -> 624MHz")
    thermal.append(f"{syslog(t_throttle)} jetson-07 tegra-thermal [1201] INFO: "
                   f"Fan PWM duty=100% rpm=5000 target_temp=65.0C")

    app.append(f"{iso(t_throttle)} level=ERROR node=jetson-07 component=thermal "
               f'msg="GPU thermal throttle activated" gpu_temp_c=70.5 '
               f'throttle_threshold_c=70.0 action="clocks reduced to 624MHz"')
    app.append(f"{iso(t_throttle + timedelta(milliseconds=10))} level=WARN node=jetson-07 "
               f'component=power msg="Power budget exceeded" current_draw=15.2W '
               f'budget=15.0W action="DVFS limiting engaged"')

    dmesg.append(f"{dmesg_ts(t_throttle)} tegra-thermal: CRITICAL: "
                 f"zone1(gpu) exceeded trip point 70000mC")
    dmesg.append(f"{dmesg_ts(t_throttle)} tegra-thermal: cooling action: "
                 f"setting gpu_max_freq=624000000Hz (was 918000000Hz)")
    dmesg.append(f"{dmesg_ts(t_throttle)} nvgpu: gpu0: thermal cap applied: "
                 f"gpcclk 918MHz -> 624MHz")
    dmesg.append(f"{dmesg_ts(t_throttle)} nvgpu: gpu0: pending work backlog: "
                 f"32 jobs queued, avg completion 142ms (normal: 30ms)")
    dmesg.append(f"{dmesg_ts(t_throttle)} tegra-pmc: power rail VDD_GPU: "
                 f"current=15200mW, budget=15000mW, OVER BUDGET")

    # -- Deadline misses t0+11s --
    t_miss = t0 + timedelta(seconds=11)
    for i, (pipe, lat) in enumerate([("camera-03", 142), ("camera-01", 118),
                                      ("camera-02", 125), ("camera-03", 165)]):
        tt = t_miss + timedelta(milliseconds=i * 300)
        fps = round(4000 / lat, 1)
        extra = " session=sess-a1b2c3" if pipe == "camera-03" else ""
        app.append(f"{iso(tt)} level=ERROR node=jetson-07 component=inference "
                   f'msg="Inference deadline missed" pipeline={pipe} '
                   f"batch_id=B-{90130 + i} frames=4 latency_ms={lat} "
                   f"fps={fps}{extra} deadline_ms=100")

    app.append(f"{iso(t_miss + timedelta(milliseconds=500))} level=WARN node=jetson-07 "
               f'component=inference msg="Frame drop detected" pipeline=camera-03 '
               f"session=sess-a1b2c3 dropped_frames=8 reason=\"inference backpressure\"")

    # -- Deeper throttle t0+13s --
    t_deep = t0 + timedelta(seconds=13)
    thermal.append(f"{syslog(t_deep)} jetson-07 tegra-thermal [1201] ERROR: "
                   f"GPU thermal throttle deepened: gpu=73.0C still above threshold")
    thermal.append(f"{syslog(t_deep)} jetson-07 tegra-thermal [1201] INFO: "
                   f"DVFS action: gpu clocks 624MHz -> 420MHz")
    thermal.append(f"{syslog(t_deep)} jetson-07 tegra-thermal [1201] WARN: "
                   f"Fan at maximum speed, cooling capacity limited -- check enclosure airflow")

    app.append(f"{iso(t_deep)} level=ERROR node=jetson-07 component=thermal "
               f'msg="GPU thermal throttle deepened" gpu_temp_c=73.0 '
               f'action="clocks reduced to 420MHz"')

    dmesg.append(f"{dmesg_ts(t_deep)} tegra-thermal: CRITICAL: "
                 f"zone1(gpu) still above trip, deepening throttle")
    dmesg.append(f"{dmesg_ts(t_deep)} tegra-thermal: cooling action: "
                 f"setting gpu_max_freq=420000000Hz (was 624000000Hz)")
    dmesg.append(f"{dmesg_ts(t_deep)} nvgpu: gpu0: thermal cap applied: "
                 f"gpcclk 624MHz -> 420MHz")
    dmesg.append(f"{dmesg_ts(t_deep)} nvgpu: gpu0: WARNING: severe clock reduction, "
                 f"inference throughput will be degraded")

    # -- More deadline misses, pipeline stall t0+14-15s --
    for dt_s, lat1, lat2, lat3 in [(14, 190, 195, 210), (15, 185, 190, 205)]:
        tt = t0 + timedelta(seconds=dt_s)
        for pipe, lat in [("camera-01", lat1), ("camera-02", lat2), ("camera-03", lat3)]:
            fps = round(4000 / lat, 1)
            extra = " session=sess-a1b2c3" if pipe == "camera-03" else ""
            app.append(f"{iso(tt)} level=ERROR node=jetson-07 component=inference "
                       f'msg="Inference deadline missed" pipeline={pipe} '
                       f"batch_id=B-{random.randint(90135, 90199)} frames=4 "
                       f"latency_ms={lat} fps={fps}{extra} deadline_ms=100")

    t_stall = t0 + timedelta(seconds=15)
    app.append(f"{iso(t_stall)} level=ERROR node=jetson-07 component=inference "
               f'msg="Pipeline stall: camera-03 inference queue full" '
               f"session=sess-a1b2c3 queue_depth=32 max_queue=32")
    app.append(f"{iso(t_stall + timedelta(milliseconds=10))} level=ERROR node=jetson-07 "
               f'component=scheduler msg="Pipeline camera-03 entering degraded mode" '
               f'session=sess-a1b2c3 reason="sustained deadline misses (4 consecutive)"')
    app.append(f"{iso(t_stall + timedelta(milliseconds=100))} level=WARN node=jetson-07 "
               f'component=inference msg="Frame drop detected" pipeline=camera-03 '
               f"session=sess-a1b2c3 dropped_frames=24 reason=\"inference backpressure\"")

    # -- Load shedding: suspend camera-03 at t0+17s --
    t_shed = t0 + timedelta(seconds=17)
    app.append(f"{iso(t_shed)} level=WARN node=jetson-07 component=scheduler "
               f'msg="Shedding load: suspending pipeline camera-03" '
               f'session=sess-a1b2c3 reason="thermal_throttle + deadline_miss"')
    app.append(f"{iso(t_shed + timedelta(milliseconds=10))} level=INFO node=jetson-07 "
               f'component=inference msg="TensorRT engine unloaded for pipeline camera-03" '
               f"session=sess-a1b2c3 freed_memory_mb=210")
    app.append(f"{iso(t_shed + timedelta(milliseconds=20))} level=ERROR node=jetson-07 "
               f'component=stream msg="RTSP stream disconnected" pipeline=camera-03 '
               f'session=sess-a1b2c3 stream=rtsp://192.168.1.50/cam3 reason="pipeline suspended"')

    dmesg.append(f"{dmesg_ts(t_shed)} nvgpu: gpu0: context released: "
                 f"pid=4890 comm=trt_inference freed=214958080B")
    dmesg.append(f"{dmesg_ts(t_shed)} nvgpu: gpu0: memory: "
                 f"used=3145728KB/8388608KB (37.5%)")

    # -- Recovery t0+20s to t0+35s --
    for dt_s, gpu_t, clk, fan in [
        (20, 68.5, 624, 100), (25, 60.5, 624, 90),
        (30, 53.0, 918, 70), (35, 46.0, 918, 55),
    ]:
        tt = t0 + timedelta(seconds=dt_s)
        cpu = gpu_t + random.uniform(1, 3)
        board = gpu_t - random.uniform(8, 12)
        aux = board - random.uniform(2, 4)

        thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] INFO: "
                       f"Reading thermal zones: cpu={cpu:.1f}C gpu={gpu_t:.1f}C "
                       f"board={board:.1f}C aux={aux:.1f}C")

        if dt_s == 25:
            thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] INFO: "
                           f"DVFS action: gpu clocks 420MHz -> 624MHz (temperature recovering)")
            dmesg.append(f"{dmesg_ts(tt)} tegra-thermal: cooling action: "
                         f"setting gpu_max_freq=624000000Hz (recovering)")
            dmesg.append(f"{dmesg_ts(tt)} nvgpu: gpu0: thermal cap relaxed: "
                         f"gpcclk 420MHz -> 624MHz")
        elif dt_s == 30:
            thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] INFO: "
                           f"GPU thermal throttle released: gpu={gpu_t:.1f}C well below threshold=70.0C")
            thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] INFO: "
                           f"DVFS action: gpu clocks 624MHz -> 918MHz (fully restored)")
            dmesg.append(f"{dmesg_ts(tt)} tegra-thermal: zone1(gpu) well below "
                         f"hysteresis band, releasing throttle")
            dmesg.append(f"{dmesg_ts(tt)} tegra-thermal: cooling action: "
                         f"setting gpu_max_freq=918000000Hz (fully restored)")
            dmesg.append(f"{dmesg_ts(tt)} nvgpu: gpu0: thermal cap removed: "
                         f"gpcclk restored to 918MHz")
            dmesg.append(f"{dmesg_ts(tt)} nvgpu: gpu0: pending work backlog cleared: "
                         f"0 jobs queued")

            app.append(f"{iso(tt)} level=INFO node=jetson-07 component=thermal "
                       f'msg="GPU thermal throttle released" gpu_temp_c={gpu_t} '
                       f'action="clocks restored to 918MHz"')

        thermal.append(f"{syslog(tt)} jetson-07 tegra-thermal [1201] INFO: "
                       f"Fan PWM duty={fan}% rpm={2400 + fan * 30} target_temp=65.0C")

        app.append(f"{iso(tt)} level=INFO node=jetson-07 component=metrics "
                   f'"GPU utilization={random.randint(58, 70)}% memory_used=3.4/8.0GB '
                   f'power_draw={random.uniform(9.0, 10.5):.1f}W '
                   f'clocks_gpu={clk}MHz clocks_mem=1600MHz"')

        for i, pipe in enumerate(PIPELINES):
            lat = random.randint(28, 38) if clk == 918 else random.randint(40, 55)
            fps = round(4000 / lat, 1)
            app.append(f"{iso(tt + timedelta(milliseconds=100 + i * 100))} level=INFO "
                       f"node=jetson-07 component=inference "
                       f'msg="Batch completed" model=yolov8n pipeline={pipe} '
                       f"batch_id=B-{random.randint(90200, 90299)} frames=4 "
                       f"latency_ms={lat} fps={fps}")

    # -- Restart camera-03 at t0+60s --
    t_restart = t0 + timedelta(seconds=60)
    app.append(f"{iso(t_restart)} level=INFO node=jetson-07 component=scheduler "
               f'msg="Attempting pipeline camera-03 restart" session=sess-a1b2c3')
    app.append(f"{iso(t_restart + timedelta(milliseconds=500))} level=INFO node=jetson-07 "
               f'component=inference msg="TensorRT engine loaded" session=sess-a1b2c3 '
               f"load_time_ms=600 device_memory_mb=210")
    app.append(f"{iso(t_restart + timedelta(seconds=1))} level=INFO node=jetson-07 "
               f'component=stream msg="RTSP stream reconnected" pipeline=camera-03 '
               f"session=sess-a1b2c3 stream=rtsp://192.168.1.50/cam3")
    app.append(f"{iso(t_restart + timedelta(seconds=1, milliseconds=500))} level=INFO "
               f"node=jetson-07 component=inference "
               f'msg="Batch completed" model=yolov8n pipeline=camera-03 '
               f"batch_id=B-90300 frames=4 latency_ms=30 fps=133.3 session=sess-a1b2c3")
    app.append(f"{iso(t_restart + timedelta(seconds=1, milliseconds=510))} level=INFO "
               f"node=jetson-07 component=scheduler "
               f'msg="Pipeline camera-03 restored to normal mode" session=sess-a1b2c3')

    dmesg.append(f"{dmesg_ts(t_restart)} nvgpu: gpu0: new context allocated: "
                 f"pid=4890 comm=trt_inference size=214958080B")
    dmesg.append(f"{dmesg_ts(t_restart)} nvgpu: gpu0: memory: "
                 f"used=3670016KB/8388608KB (43.7%)")


# ---------------------------------------------------------------------------
# Sprinkle some other events over 24h
# ---------------------------------------------------------------------------


def add_misc_events(t: datetime, app: list, thermal: list, dmesg: list):
    """Occasional non-incident events: payment declines, brief warnings, etc."""
    r = random.random()
    if r < 0.02:
        # Brief GPU memory warning
        app.append(f"{iso(t)} level=WARN node=jetson-07 component=metrics "
                   f'"GPU memory_used=6.8/8.0GB (85%) -- approaching limit"')
    elif r < 0.04:
        # USB camera reconnect
        cam = random.choice(["camera-01", "camera-02"])
        app.append(f"{iso(t)} level=WARN node=jetson-07 component=stream "
                   f'msg="RTSP stream hiccup" pipeline={cam} '
                   f"reconnect_ms={random.randint(200, 1500)}")
    elif r < 0.045:
        # NTP sync
        dmesg.append(f"{dmesg_ts(t)} clocksource: Switched to clocksource arch_sys_counter")
    elif r < 0.048:
        # NVMe I/O warning (transient read error, corrected by retry)
        sector = random.randint(100000, 9999999)
        dmesg.append(f"{dmesg_ts(t)} nvme nvme0: I/O Cmd(0x02) error, "
                     f"sc 0x002 sct 0x0, cdw10=0x{sector:08x}, cdw12=0x0007ff00")
        dmesg.append(f"{dmesg_ts(t)} nvme nvme0: I/O error, dev nvme0n1, "
                     f"sector {sector} op 0x0:(READ) flags 0x0 phys_seg 1 prio class 0")


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------


def generate_boot_sequence(t0: datetime, app: list, thermal: list, dmesg: list):
    """Generate startup logs: kernel init, model loading, pipeline starts."""
    t = t0

    dmesg.append(f"{dmesg_ts(t)} tegra-pmc: Jetson Orin Nano (P3767-0005) power on, "
                 f"rail=VDD_IN voltage=5000mV")
    dmesg.append(f"{dmesg_ts(t)} nvgpu: gpu0: NVIDIA GA10B (Ampere), "
                 f"1024 CUDA cores, 8192MB unified memory")
    dmesg.append(f"{dmesg_ts(t)} nvgpu: gpu0: firmware loaded, gpcclk=918MHz max=1300MHz")
    dmesg.append(f"{dmesg_ts(t)} nvme nvme0: Samsung 970 EVO Plus 250GB, "
                 f"firmware 2B2QEXM7, LBA ns:0 nr_sectors:488397168")
    dmesg.append(f"{dmesg_ts(t)} tegra-thermal: registered zones: "
                 f"cpu, gpu, board, aux | trip=70000mC hyst=5000mC")

    t += timedelta(seconds=2)
    thermal.append(f"{syslog(t)} jetson-07 tegra-thermal [1201] INFO: "
                   f"Daemon started, polling interval=5000ms")
    thermal.append(f"{syslog(t)} jetson-07 tegra-thermal [1201] INFO: "
                   f"Reading thermal zones: cpu=32.5C gpu=31.0C board=28.5C aux=26.0C")
    thermal.append(f"{syslog(t)} jetson-07 tegra-thermal [1201] INFO: "
                   f"Fan PWM duty=35% rpm=3450 target_temp=65.0C")

    t += timedelta(seconds=3)
    app.append(f"{iso(t)} level=INFO node=jetson-07 component=supervisor "
               f'msg="Inference supervisor starting" version=2.4.1 pid=1823')
    app.append(f"{iso(t + timedelta(milliseconds=50))} level=INFO node=jetson-07 "
               f'component=supervisor msg="Loading pipeline config" '
               f"path=/etc/jetson-agent/pipelines.yaml pipelines=2")

    t += timedelta(seconds=1)
    for i, pipe in enumerate(["camera-01", "camera-02"]):
        tt = t + timedelta(seconds=i * 3)
        stream = f"rtsp://192.168.1.{50 + i}/cam{i + 1}"
        app.append(f"{iso(tt)} level=INFO node=jetson-07 component=scheduler "
                   f'msg="Pipeline {pipe} starting" model=yolov8n stream={stream}')
        app.append(f"{iso(tt + timedelta(milliseconds=20))} level=INFO node=jetson-07 "
                   f'component=inference msg="Loading TensorRT engine for pipeline {pipe}" '
                   f"engine=/models/yolov8n.engine")
        app.append(f"{iso(tt + timedelta(milliseconds=600))} level=INFO node=jetson-07 "
                   f'component=inference msg="TensorRT engine loaded" '
                   f"pipeline={pipe} load_time_ms={550 + random.randint(0, 80)} device_memory_mb=210")
        app.append(f"{iso(tt + timedelta(seconds=1))} level=INFO node=jetson-07 "
                   f'component=stream msg="RTSP stream connected" '
                   f"pipeline={pipe} stream={stream}")

        dmesg.append(f"{dmesg_ts(tt)} nvgpu: gpu0: new context allocated: "
                     f"pid={1830 + i} comm=trt_inference size=214958080B")

    t += timedelta(seconds=8)
    app.append(f"{iso(t)} level=INFO node=jetson-07 component=supervisor "
               f'msg="All pipelines online" active=2 standby=0')
    dmesg.append(f"{dmesg_ts(t)} nvgpu: gpu0: memory: "
                 f"used=2621440KB/8388608KB (31.3%)")


def generate_memory_spike(t0: datetime, app: list, thermal: list, dmesg: list):
    """Minor incident: GPU memory spikes to 90%, self-resolves after GC."""
    t = t0

    app.append(f"{iso(t)} level=WARN node=jetson-07 component=metrics "
               f'"GPU memory_used=7.2/8.0GB (90%) -- approaching limit"')
    dmesg.append(f"{dmesg_ts(t)} nvgpu: gpu0: memory: "
                 f"used=7340032KB/8388608KB (87.5%) -- high watermark")
    app.append(f"{iso(t + timedelta(seconds=1))} level=WARN node=jetson-07 "
               f'component=inference msg="CUDA unified memory migration detected" '
               f"pages_migrated=1847 direction=gpu_to_cpu latency_spike_ms=340")
    app.append(f"{iso(t + timedelta(seconds=1, milliseconds=200))} level=WARN "
               f"node=jetson-07 component=inference "
               f'msg="Inference latency increasing" pipeline=camera-02 '
               f"batch_id=B-{random.randint(50000, 59999)} frames=4 "
               f"latency_ms=285 fps=14.0 expected_max_ms=40")

    app.append(f"{iso(t + timedelta(seconds=5))} level=INFO node=jetson-07 "
               f'component=inference msg="TensorRT memory pool compacted" '
               f"freed_mb=380 trigger=high_watermark")
    dmesg.append(f"{dmesg_ts(t + timedelta(seconds=5))} nvgpu: gpu0: memory: "
                 f"used=5242880KB/8388608KB (62.5%) -- after compaction")

    app.append(f"{iso(t + timedelta(seconds=8))} level=INFO node=jetson-07 "
               f'component=metrics "GPU memory_used=5.0/8.0GB (62%) -- recovered"')


def generate():
    app_lines = []
    thermal_lines = []
    dmesg_lines = []

    # Boot sequence at T_START
    generate_boot_sequence(T_START, app_lines, thermal_lines, dmesg_lines)

    t = T_START + timedelta(seconds=20)  # skip past boot
    incident_done = False
    memory_spike_done = False

    while t < NOW:
        # Normal ticks every 5s
        normal_app_tick(t, app_lines)
        normal_thermal_tick(t, thermal_lines)
        normal_dmesg_tick(t, dmesg_lines)
        add_misc_events(t, app_lines, thermal_lines, dmesg_lines)

        # Memory spike (minor, self-resolving)
        if not memory_spike_done and t >= T_MEMORY_SPIKE:
            generate_memory_spike(t, app_lines, thermal_lines, dmesg_lines)
            memory_spike_done = True
            t += timedelta(seconds=15)
            continue

        # Thermal throttle cascade (major incident)
        if not incident_done and t >= T_INCIDENT:
            generate_incident(t, app_lines, thermal_lines, dmesg_lines)
            incident_done = True
            t += timedelta(seconds=65)  # skip past incident + recovery
            continue

        t += timedelta(seconds=5)

    os.makedirs(OUT_DIR, exist_ok=True)
    for name, lines in [("app.log", app_lines), ("thermal.log", thermal_lines),
                         ("dmesg.log", dmesg_lines)]:
        path = os.path.join(OUT_DIR, name)
        with open(path, "w") as f:
            f.write("\n".join(lines) + "\n")
        print(f"Wrote {len(lines):>5} lines to {path}")


if __name__ == "__main__":
    generate()
