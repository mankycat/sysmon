# sysmon.py
import argparse, sqlite3, time, os, math
from datetime import datetime, timezone
import psutil
import matplotlib.pyplot as plt

# --- Optional: GPU via NVML ---
try:
    import pynvml
    _NVML_OK = True
except Exception:
    _NVML_OK = False

DB_SCHEMA_HOST = """
CREATE TABLE IF NOT EXISTS host_metrics(
  ts REAL NOT NULL,                 -- epoch seconds (UTC)
  cpu_percent REAL,
  mem_percent REAL,
  cpu_temp_c REAL,
  rapl_watts REAL
);
CREATE INDEX IF NOT EXISTS idx_host_ts ON host_metrics(ts);
"""

DB_SCHEMA_GPU = """
CREATE TABLE IF NOT EXISTS gpu_metrics(
  ts REAL NOT NULL,                 -- epoch seconds (UTC)
  gpu_index INTEGER,
  name TEXT,
  util_percent REAL,
  mem_used_gb REAL,
  mem_total_gb REAL,
  power_w REAL,
  temp_c REAL
);
CREATE INDEX IF NOT EXISTS idx_gpu_ts ON gpu_metrics(ts);
"""

DB_SCHEMA_DISK = """
CREATE TABLE IF NOT EXISTS disk_metrics(
  ts REAL NOT NULL,
  device TEXT NOT NULL,            -- e.g. sda, nvme0n1, or __total__
  read_bytes INTEGER,
  write_bytes INTEGER,
  read_count INTEGER,
  write_count INTEGER,
  busy_ms INTEGER
);
CREATE INDEX IF NOT EXISTS idx_disk_ts ON disk_metrics(ts);
"""

def utc_now_epoch():
    return datetime.now(timezone.utc).timestamp()

def init_db(path):
    conn = sqlite3.connect(path)
    with conn:
        conn.executescript(DB_SCHEMA_HOST)
        conn.executescript(DB_SCHEMA_GPU)
        conn.executescript(DB_SCHEMA_DISK)
    return conn

# ----------- RAPL (CPU封包功耗) -----------
def rapl_paths():
    base = "/sys/class/powercap/intel-rapl:0"
    efile = os.path.join(base, "energy_uj")
    return efile if os.path.exists(efile) else None

class RaplMeter:
    def __init__(self):
        self.path = rapl_paths()
        self.last_energy = None
        self.last_time = None
    def read_watts(self):
        # 返回「自上次呼叫到現在」的平均瓦數；第一次呼叫回傳 None
        if not self.path: return None
        try:
            now = time.time()
            with open(self.path, "r") as f:
                uj = int(f.read().strip())
            if self.last_energy is None:
                self.last_energy, self.last_time = uj, now
                return None
            duj = uj - self.last_energy
            dt = max(1e-6, now - self.last_time)
            self.last_energy, self.last_time = uj, now
            # 微焦耳 -> 焦耳，再除以秒 = 瓦
            return (duj / 1_000_000.0) / dt
        except Exception:
            return None

# ----------- GPU (NVML) -----------
def nvml_init():
    if not _NVML_OK: return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False

def nvml_shutdown():
    if _NVML_OK:
        try: pynvml.nvmlShutdown()
        except Exception: pass

def poll_gpus():
    """Return list of dict per GPU"""
    out = []
    if not _NVML_OK: return out
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore")
            util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem_used_gb = mem.used / (1024**3)
            mem_total_gb = mem.total / (1024**3)
            # power may raise on unsupported devices
            try:
                power_w = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
            except Exception:
                power_w = None
            try:
                temp_c = pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU)
            except Exception:
                temp_c = None
            out.append({
                "gpu_index": i,
                "name": name,
                "util": float(util) if util is not None else None,
                "mem_used_gb": mem_used_gb,
                "mem_total_gb": mem_total_gb,
                "power_w": power_w,
                "temp_c": float(temp_c) if temp_c is not None else None,
            })
    except Exception:
        pass
    return out

# ----------- Host sensors -----------
def host_temps_c():
    """Pick a representative CPU temp (max of cores) if available."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps: return None
        candidates = []
        for label, entries in temps.items():
            for e in entries:
                if e.current is not None:
                    # prefer CPU/coretemp packages
                    score = 2 if ("core" in label.lower() or "cpu" in label.lower()) else 1
                    candidates.append((score, e.current))
        if not candidates: return None
        # max of highest-score group
        max_score = max(s for s,_ in candidates)
        vals = [v for s,v in candidates if s==max_score]
        return max(vals) if vals else None
    except Exception:
        return None

import itertools
def poll_disks():
    # per-disk counters + total
    try:
        per = psutil.disk_io_counters(perdisk=True, nowrap=True) or {}
        total = psutil.disk_io_counters(perdisk=False, nowrap=True)
    except Exception:
        return {}, None
    # busy_time 欄位不同平台可能無，容錯為 0
    def busy_ms(x): return getattr(x, "busy_time", 0)
    per_out = {
        dev: dict(read_bytes=v.read_bytes, write_bytes=v.write_bytes,
                  read_count=v.read_count, write_count=v.write_count,
                  busy_ms=busy_ms(v))
        for dev, v in per.items()
    }
    tot_out = dict(read_bytes=total.read_bytes, write_bytes=total.write_bytes,
                   read_count=total.read_count, write_count=total.write_count,
                   busy_ms=busy_ms(total))
    return per_out, tot_out


# ----------- Collector loop -----------
def collect_loop(db_path, interval):
    conn = init_db(db_path)
    rapl = RaplMeter()
    have_nvml = nvml_init()
    print(f"[collector] writing to {db_path} every {interval}s; NVML={have_nvml}, RAPL={'on' if rapl.path else 'off'}")
    try:
        # prime cpu_percent first call
        psutil.cpu_percent(interval=None)
        while True:
            ts = utc_now_epoch()
            cpu_p = psutil.cpu_percent(interval=None)
            mem_p = psutil.virtual_memory().percent
            cpu_tc = host_temps_c()
            rapl_w = rapl.read_watts()
            per, tot = poll_disks()

            with conn:
                conn.execute(
                    "INSERT INTO host_metrics(ts,cpu_percent,mem_percent,cpu_temp_c,rapl_watts) VALUES(?,?,?,?,?)",
                    (ts, cpu_p, mem_p, cpu_tc, rapl_w)
                )
                for g in poll_gpus():
                    conn.execute(
                        "INSERT INTO gpu_metrics(ts,gpu_index,name,util_percent,mem_used_gb,mem_total_gb,power_w,temp_c) VALUES(?,?,?,?,?,?,?,?)",
                        (ts, g["gpu_index"], g["name"], g["util"], g["mem_used_gb"], g["mem_total_gb"], g["power_w"], g["temp_c"])
                    )
        
                # per device
                for dev, v in per.items():
                    conn.execute(
                    "INSERT INTO disk_metrics(ts,device,read_bytes,write_bytes,read_count,write_count,busy_ms) VALUES(?,?,?,?,?,?,?)",
                    (ts, dev, v["read_bytes"], v["write_bytes"], v["read_count"], v["write_count"], v["busy_ms"])
                    )
                # total row
                if tot:
                    conn.execute(
                    "INSERT INTO disk_metrics(ts,device,read_bytes,write_bytes,read_count,write_count,busy_ms) VALUES(?,?,?,?,?,?,?)",
                    (ts, "__total__", tot["read_bytes"], tot["write_bytes"], tot["read_count"], tot["write_count"], tot["busy_ms"])
                    )

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n[collector] stopped by user.")
    finally:
        conn.close()
        nvml_shutdown()

# ----------- Plotting -----------
def parse_time(s):
    # Accept "YYYY-mm-dd HH:MM" or epoch seconds
    s = s.strip()
    if s.isdigit(): return float(s)
    dt = datetime.strptime(s, "%Y-%m-%d %H:%M")
    return dt.replace(tzinfo=timezone.utc).timestamp()

def query_df(conn, sql, args=()):
    cur = conn.execute(sql, args)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    return cols, rows

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def ts_to_local_str(ts):
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")

def plot_series(x, ys, labels, title, ylab, out_png):
    plt.figure(figsize=(10,4.5))
    for y,lab in zip(ys, labels):
        plt.plot(x, y, label=lab, linewidth=1.5)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(ylab)
    if len(labels)>1: plt.legend(loc="best")
    # x ticks thinning
    if len(x) > 12:
        step = max(1, len(x)//12)
        xt = x[::step]; xtl = [ts_to_local_str(t) for t in xt]
        plt.xticks(xt, xtl, rotation=30, ha="right")
    else:
        plt.xticks(x, [ts_to_local_str(t) for t in x], rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def deriv_per_sec(ts, vals):
    # 以相鄰差分 / Δt，回傳與 ts[1:] 對齊的序列
    out_ts, out_v = [], []
    for (t0,v0),(t1,v1) in zip(zip(ts,vals), zip(ts[1:], vals[1:])):
        dt = max(1e-6, t1 - t0)
        out_ts.append(t1)
        out_v.append((v1 - v0) / dt)
    return out_ts, out_v

def safe_nan(arr):
    import math
    return [ (x if x is not None else math.nan) for x in arr ]

def plot_range(db_path, t_from, t_to, outdir):
    ensure_dir(outdir)
    conn = sqlite3.connect(db_path)

    # Host charts
    cols, rows = query_df(conn,
        "SELECT ts,cpu_percent,mem_percent,cpu_temp_c,rapl_watts FROM host_metrics WHERE ts BETWEEN ? AND ? ORDER BY ts",
        (t_from, t_to)
    )
    if rows:
        ts = [r[0] for r in rows]
        cpu = [r[1] for r in rows]
        mem = [r[2] for r in rows]
        ctemp = [r[3] for r in rows]
        rapl = [r[4] for r in rows]

        plot_series(ts, [cpu], ["CPU %"], "CPU Utilization", "percent", os.path.join(outdir, "host_cpu_percent.png"))
        plot_series(ts, [mem], ["Memory %"], "Memory Utilization", "percent", os.path.join(outdir, "host_mem_percent.png"))

        if any(v is not None for v in ctemp):
            plot_series(ts, [[v if v is not None else math.nan for v in ctemp]], ["CPU temp"], "CPU Temperature", "°C", os.path.join(outdir, "host_cpu_temp.png"))

        if any(v is not None for v in rapl):
            plot_series(ts, [[v if v is not None else math.nan for v in rapl]], ["CPU pkg (RAPL)"], "CPU Package Power (RAPL)", "Watts", os.path.join(outdir, "host_cpu_rapl_watts.png"))

    # GPU charts (per GPU index)
    cols, gpus = query_df(conn,
        "SELECT DISTINCT gpu_index,name FROM gpu_metrics WHERE ts BETWEEN ? AND ? ORDER BY gpu_index",
        (t_from, t_to)
    )
    for gpu_index, name in gpus:
        cols, rows = query_df(conn,
            "SELECT ts,util_percent,mem_used_gb,mem_total_gb,power_w,temp_c FROM gpu_metrics WHERE ts BETWEEN ? AND ? AND gpu_index=? ORDER BY ts",
            (t_from, t_to, gpu_index)
        )
        if not rows: continue
        ts = [r[0] for r in rows]
        util = [r[1] for r in rows]
        mem_used = [r[2] for r in rows]
        mem_total = rows[0][3] if rows else None
        power = [r[4] for r in rows]
        temp = [r[5] for r in rows]

        safe = lambda arr: [v if v is not None else math.nan for v in arr]
        base = f"gpu{gpu_index}_{name.replace(' ','_').replace('/','-')}"

        plot_series(ts, [util], [f"GPU{gpu_index} util%"], f"{name} - Utilization", "percent", os.path.join(outdir, f"{base}_util.png"))
        plot_series(ts, [mem_used], [f"GPU{gpu_index} mem used (/{mem_total:.2f} GB)" if mem_total else f"GPU{gpu_index} mem used"], f"{name} - VRAM Used", "GB", os.path.join(outdir, f"{base}_mem_used.png"))
        if any(v is not None for v in power):
            plot_series(ts, [safe(power)], [f"GPU{gpu_index} power W"], f"{name} - Power", "Watts", os.path.join(outdir, f"{base}_power.png"))
        if any(v is not None for v in temp):
            plot_series(ts, [safe(temp)], [f"GPU{gpu_index} temp °C"], f"{name} - Temperature", "°C", os.path.join(outdir, f"{base}_temp.png"))

    cols, devs = query_df(conn,
    "SELECT DISTINCT device FROM disk_metrics WHERE ts BETWEEN ? AND ? ORDER BY device",
    (t_from, t_to))

    for (device,) in devs:
        cols, rows = query_df(conn, """
            SELECT ts, read_bytes, write_bytes, read_count, write_count, busy_ms
            FROM disk_metrics
            WHERE ts BETWEEN ? AND ? AND device=?
            ORDER BY ts
        """, (t_from, t_to, device))
        if not rows or len(rows) < 2:
            continue
        ts  = [r[0] for r in rows]
        rB  = [r[1] for r in rows]
        wB  = [r[2] for r in rows]
        rC  = [r[3] for r in rows]
        wC  = [r[4] for r in rows]
        bms = [r[5] for r in rows]

        # 轉速率
        ts1, rMBps = deriv_per_sec(ts, rB);  rMBps = [x/1_000_000 for x in rMBps]
        _,   wMBps = deriv_per_sec(ts, wB);  wMBps = [x/1_000_000 for x in wMBps]
        _,   rIOPS = deriv_per_sec(ts, rC)
        _,   wIOPS = deriv_per_sec(ts, wC)

        # 利用率(%) ≈ Δbusy_ms / Δt
        util = []
        for (t0,b0),(t1,b1) in zip(zip(ts, bms), zip(ts[1:], bms[1:])):
            dt = max(1e-6, t1 - t0)
            util.append( max(0.0, min(100.0, (b1 - b0) / dt / 10.0)) )  # 1000ms=100%, *1/10

        base = f"disk_{device.replace('/','-')}"
        plot_series(ts1, [rMBps, wMBps], ["read MB/s","write MB/s"],
                    f"{device} - Throughput", "MB/s", os.path.join(outdir, f"{base}_mbps.png"))
        plot_series(ts1, [rIOPS, wIOPS], ["read IOPS","write IOPS"],
                    f"{device} - IOPS", "ops/s", os.path.join(outdir, f"{base}_iops.png"))
        plot_series(ts1, [util], ["util %"],
                    f"{device} - Utilization (approx.)", "percent", os.path.join(outdir, f"{base}_util.png"))

    conn.close()
    print(f"[plot] charts saved to: {outdir}")

# ----------- CLI -----------
def main():
    ap = argparse.ArgumentParser(description="Lightweight system metrics collector & plotter (time-range)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("collect", help="run collector loop")
    ap_c.add_argument("--db", default="metrics.db")
    ap_c.add_argument("--interval", type=int, default=2, help="sampling seconds")

    ap_p = sub.add_parser("plot", help="plot charts for a time range")
    ap_p.add_argument("--db", default="metrics.db")
    ap_p.add_argument("--from", dest="time_from", required=True, help='e.g. "2025-09-08 10:00" (local) or epoch seconds')
    ap_p.add_argument("--to", dest="time_to", required=True, help='e.g. "2025-09-08 12:00" (local) or epoch seconds')
    ap_p.add_argument("--out", dest="outdir", default="charts")

    args = ap.parse_args()

    if args.cmd == "collect":
        collect_loop(args.db, args.interval)
    else:
        # plot
        t_from = parse_time(args.time_from)
        t_to = parse_time(args.time_to)
        if t_to <= t_from:
            raise SystemExit("--to must be greater than --from")
        plot_range(args.db, t_from, t_to, args.outdir)

if __name__ == "__main__":
    main()