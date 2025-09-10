# sysmon.py
import argparse, sqlite3, time, os, math, subprocess, csv
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
        if not self.path:
            return None
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
            return (duj / 1_000_000.0) / dt  # μJ→J，再/秒=瓦
        except Exception:
            return None

# ----------- GPU (NVML / nvidia-smi) -----------
def nvml_init():
    if not _NVML_OK:
        return False
    try:
        pynvml.nvmlInit()
        return True
    except Exception:
        return False

def nvml_shutdown():
    if _NVML_OK:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass

def poll_gpus_nvml():
    """Return list of dict per GPU via NVML"""
    out = []
    if not _NVML_OK:
        return out
    try:
        count = pynvml.nvmlDeviceGetCount()
        for i in range(count):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            try:
                name = pynvml.nvmlDeviceGetName(h)
                name = name.decode("utf-8") if isinstance(name, (bytes, bytearray)) else str(name)
            except Exception:
                name = f"GPU-{i}"
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
            except Exception:
                util = None
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                mem_used_gb = mem.used / (1024**3)
                mem_total_gb = mem.total / (1024**3)
            except Exception:
                mem_used_gb = None
                mem_total_gb = None
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
                "mem_used_gb": float(mem_used_gb) if mem_used_gb is not None else None,
                "mem_total_gb": float(mem_total_gb) if mem_total_gb is not None else None,
                "power_w": power_w,
                "temp_c": float(temp_c) if temp_c is not None else None,
            })
    except Exception:
        if os.environ.get("SYSMON_DEBUG") == "1":
            import traceback; traceback.print_exc()
        return []
    return out

def poll_gpus_smi():
    """
    使用 nvidia-smi 查詢：index,name,util,power,temp,mem_used,mem_total
    回傳與 NVML 版相同結構
    """
    q = "index,name,utilization.gpu,power.draw,temperature.gpu,memory.used,memory.total"
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits"],
            stderr=subprocess.STDOUT, timeout=3
        ).decode()
    except Exception:
        return []
    res = []
    reader = csv.reader(out.strip().splitlines())
    for row in reader:
        if not row:
            continue
        try:
            idx, name, util, power, temp, mem_used, mem_total = [c.strip() for c in row]
            res.append({
                "gpu_index": int(idx),
                "name": name,
                "util": float(util) if util not in ("N/A", "") else None,
                "mem_used_gb": (float(mem_used) / 1024.0) if mem_used not in ("N/A","") else None,
                "mem_total_gb": (float(mem_total) / 1024.0) if mem_total not in ("N/A","") else None,
                "power_w": float(power) if power not in ("N/A","") else None,
                "temp_c": float(temp) if temp not in ("N/A","") else None,
            })
        except Exception:
            continue
    return res

# ----------- Host sensors -----------
def host_temps_c():
    """Pick a representative CPU temp (max of cores) if available."""
    try:
        temps = psutil.sensors_temperatures()
        if not temps:
            return None
        candidates = []
        for label, entries in temps.items():
            for e in entries:
                if e.current is not None:
                    score = 2 if ("core" in label.lower() or "cpu" in label.lower()) else 1
                    candidates.append((score, e.current))
        if not candidates:
            return None
        max_score = max(s for s,_ in candidates)
        vals = [v for s,v in candidates if s == max_score]
        return max(vals) if vals else None
    except Exception:
        return None

def poll_disks():
    # per-disk counters + total
    try:
        per = psutil.disk_io_counters(perdisk=True, nowrap=True) or {}
        total = psutil.disk_io_counters(perdisk=False, nowrap=True)
    except Exception:
        return {}, None
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
def collect_loop(db_path, interval, debug=False, gpu_via="nvml"):
    conn = init_db(db_path)
    rapl = RaplMeter()
    have_nvml = (gpu_via == "nvml") and nvml_init()
    print(f"[collector] writing to {db_path} every {interval}s; NVML={have_nvml}, RAPL={'on' if rapl.path else 'off'}, gpu_via={gpu_via}")
    if debug and have_nvml:
        try:
            drv = pynvml.nvmlSystemGetDriverVersion()
            drv = drv.decode() if isinstance(drv, (bytes, bytearray)) else str(drv)
            print("[nvml] driver:", drv)
            print("[nvml] cuda:", getattr(pynvml, "nvmlSystemGetCudaDriverVersion_v2", lambda: "N/A")())
            print("[nvml] count:", pynvml.nvmlDeviceGetCount())
        except Exception as e:
            print("[nvml] sanity error:", repr(e))
    try:
        psutil.cpu_percent(interval=None)  # prime
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
                # GPU metrics
                gpu_list = poll_gpus_nvml() if have_nvml else poll_gpus_smi()
                for g in gpu_list:
                    conn.execute(
                        "INSERT INTO gpu_metrics(ts,gpu_index,name,util_percent,mem_used_gb,mem_total_gb,power_w,temp_c) VALUES(?,?,?,?,?,?,?,?)",
                        (ts, g["gpu_index"], g["name"], g["util"], g["mem_used_gb"], g["mem_total_gb"], g["power_w"], g["temp_c"])
                    )
                # Disk metrics (per device)
                for dev, v in per.items():
                    conn.execute(
                        "INSERT INTO disk_metrics(ts,device,read_bytes,write_bytes,read_count,write_count,busy_ms) VALUES(?,?,?,?,?,?,?)",
                        (ts, dev, v["read_bytes"], v["write_bytes"], v["read_count"], v["write_count"], v["busy_ms"])
                    )
                # Disk total
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

# ----------- Plot helpers -----------
def parse_time(s):
    # Accept "YYYY-mm-dd HH:MM" or epoch seconds
    s = s.strip()
    if s.isdigit():
        return float(s)
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
    for y, lab in zip(ys, labels):
        plt.plot(x, y, label=lab, linewidth=1.5)
    plt.title(title)
    plt.xlabel("time")
    plt.ylabel(ylab)
    if len(labels) > 1:
        plt.legend(loc="best")
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
    out_ts, out_v = [], []
    for (t0, v0), (t1, v1) in zip(zip(ts, vals), zip(ts[1:], vals[1:])):
        dt = max(1e-6, t1 - t0)
        out_ts.append(t1)
        out_v.append((v1 - v0) / dt)
    return out_ts, out_v

def safe_nan(arr):
    import math
    return [(x if x is not None else math.nan) for x in arr]

def bin_series(ts, vals, bin_sec):
    """將時間序列以 bin_sec 做平均分箱；返回 (binned_ts, binned_vals)。"""
    if not ts or bin_sec is None or bin_sec <= 0:
        return ts, vals
    out_t, out_v = [], []
    acc_v, acc_n = 0.0, 0
    cur_bin_start = ts[0] - (ts[0] % bin_sec)
    cur_bin_end = cur_bin_start + bin_sec
    import math
    for t, v in zip(ts, vals):
        vv = None
        if v is not None and not (isinstance(v, float) and math.isnan(v)):
            try:
                vv = float(v)
            except Exception:
                vv = None
        while t >= cur_bin_end:
            out_t.append(cur_bin_end)
            out_v.append((acc_v/acc_n) if acc_n > 0 else math.nan)
            cur_bin_start = cur_bin_end
            cur_bin_end = cur_bin_start + bin_sec
            acc_v, acc_n = 0.0, 0
        if vv is not None and not math.isnan(vv):
            acc_v += vv
            acc_n += 1
    out_t.append(cur_bin_end)
    out_v.append((acc_v/acc_n) if acc_n > 0 else math.nan)
    return out_t, out_v

# ----------- Plotting -----------
def plot_range(db_path, t_from, t_to, outdir, bin_seconds=0):
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

        tt, vv = ts, cpu
        tt, vv = bin_series(tt, vv, bin_seconds)
        plot_series(tt, [vv], ["CPU %"], "CPU Utilization", "percent",
                    os.path.join(outdir, "host_cpu_percent.png"))

        tt, vv = ts, mem
        tt, vv = bin_series(tt, vv, bin_seconds)
        plot_series(tt, [vv], ["Memory %"], "Memory Utilization", "percent",
                    os.path.join(outdir, "host_mem_percent.png"))

        if any(v is not None for v in ctemp):
            vv = [v if v is not None else math.nan for v in ctemp]
            tt, vv = bin_series(ts, vv, bin_seconds)
            plot_series(tt, [vv], ["CPU temp"], "CPU Temperature", "°C",
                        os.path.join(outdir, "host_cpu_temp.png"))

        if any(v is not None for v in rapl):
            vv = [v if v is not None else math.nan for v in rapl]
            tt, vv = bin_series(ts, vv, bin_seconds)
            plot_series(tt, [vv], ["CPU pkg (RAPL)"], "CPU Package Power (RAPL)", "Watts",
                        os.path.join(outdir, "host_cpu_rapl_watts.png"))

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
        if not rows:
            continue
        ts = [r[0] for r in rows]
        util = [r[1] for r in rows]
        mem_used = [r[2] for r in rows]
        mem_total = rows[0][3] if rows else None
        power = [r[4] for r in rows]
        temp = [r[5] for r in rows]

        s = lambda arr: [v if v is not None else math.nan for v in arr]
        base = f"gpu{gpu_index}_{str(name).replace(' ','_').replace('/','-')}"

        tt, vv = bin_series(ts, s(util), bin_seconds)
        plot_series(tt, [vv], [f"GPU{gpu_index} util%"],
                    f"{name} - Utilization", "percent",
                    os.path.join(outdir, f"{base}_util.png"))

        tt, vv = bin_series(ts, s(mem_used), bin_seconds)
        label = f"GPU{gpu_index} mem used (/{mem_total:.2f} GB)" if isinstance(mem_total,(int,float)) else f"GPU{gpu_index} mem used"
        plot_series(tt, [vv], [label],
                    f"{name} - VRAM Used", "GB",
                    os.path.join(outdir, f"{base}_mem_used.png"))

        if any(v is not None for v in power):
            tt, vv = bin_series(ts, s(power), bin_seconds)
            plot_series(tt, [vv], [f"GPU{gpu_index} power W"],
                        f"{name} - Power", "Watts",
                        os.path.join(outdir, f"{base}_power.png"))
        if any(v is not None for v in temp):
            tt, vv = bin_series(ts, s(temp), bin_seconds)
            plot_series(tt, [vv], [f"GPU{gpu_index} temp °C"],
                        f"{name} - Temperature", "°C",
                        os.path.join(outdir, f"{base}_temp.png"))

    # Disk charts (per device + total)
    cols, devs = query_df(conn,
        "SELECT DISTINCT device FROM disk_metrics WHERE ts BETWEEN ? AND ? ORDER BY device",
        (t_from, t_to)
    )
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

        ts1, rMBps = deriv_per_sec(ts, rB);  rMBps = [x/1_000_000 for x in rMBps]
        _,   wMBps = deriv_per_sec(ts, wB);  wMBps = [x/1_000_000 for x in wMBps]
        _,   rIOPS = deriv_per_sec(ts, rC)
        _,   wIOPS = deriv_per_sec(ts, wC)

        util = []
        for (t0,b0),(t1,b1) in zip(zip(ts, bms), zip(ts[1:], bms[1:])):
            dt = max(1e-6, t1 - t0)
            util.append(max(0.0, min(100.0, (b1 - b0) / dt / 10.0)))  # 1000ms=100%

        # 對速率/利用率再做分箱
        ts_r, rMBps = bin_series(ts1, rMBps, bin_seconds)
        _,    wMBps = bin_series(ts1, wMBps, bin_seconds)
        ts_i, rIOPS = bin_series(ts1, rIOPS, bin_seconds)
        _,    wIOPS = bin_series(ts1, wIOPS, bin_seconds)
        ts_u, util  = bin_series(ts1, util,  bin_seconds)

        base = f"disk_{device.replace('/','-')}"
        plot_series(ts_r, [rMBps, wMBps], ["read MB/s","write MB/s"],
                    f"{device} - Throughput", "MB/s",
                    os.path.join(outdir, f"{base}_mbps.png"))
        plot_series(ts_i, [rIOPS, wIOPS], ["read IOPS","write IOPS"],
                    f"{device} - IOPS", "ops/s",
                    os.path.join(outdir, f"{base}_iops.png"))
        plot_series(ts_u, [util], ["util %"],
                    f"{device} - Utilization (approx.)", "percent",
                    os.path.join(outdir, f"{base}_util.png"))

    conn.close()
    print(f"[plot] charts saved to: {outdir}")

# ----------- CLI -----------
def main():
    ap = argparse.ArgumentParser(description="Lightweight system metrics collector & plotter (time-range)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    ap_c = sub.add_parser("collect", help="run collector loop")
    ap_c.add_argument("--db", default="metrics.db")
    ap_c.add_argument("--interval", type=int, default=2, help="sampling seconds")
    ap_c.add_argument("--debug", action="store_true")
    ap_c.add_argument("--gpu-via", choices=["nvml","smi"], default="nvml",
                      help="GPU metrics via NVML (default) or nvidia-smi fallback")

    ap_p = sub.add_parser("plot", help="plot charts for a time range")
    ap_p.add_argument("--db", default="metrics.db")
    ap_p.add_argument("--from", dest="time_from", required=True,
                      help='e.g. "2025-09-10 10:00" (local) or epoch seconds')
    ap_p.add_argument("--to", dest="time_to", required=True,
                      help='e.g. "2025-09-10 12:00" (local) or epoch seconds')
    ap_p.add_argument("--out", dest="outdir", default="charts")
    ap_p.add_argument("--bin-seconds", type=int, default=0,
                      help="繪圖端分箱秒數（0=不分箱，建議 30/60/300 等）")

    args = ap.parse_args()

    if args.cmd == "collect":
        collect_loop(args.db, args.interval, args.debug, args.gpu_via)
    else:
        t_from = parse_time(args.time_from)
        t_to = parse_time(args.time_to)
        if t_to <= t_from:
            raise SystemExit("--to must be greater than --from")
        plot_range(args.db, t_from, t_to, args.outdir, args.bin_seconds)

if __name__ == "__main__":
    main()