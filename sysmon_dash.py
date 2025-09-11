# sysmon_dash.py
import argparse, sqlite3, math
from datetime import datetime, timezone, timedelta

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update
import plotly.graph_objs as go

# ---------- DB helpers ----------
def open_conn(db_path):
    return sqlite3.connect(db_path, check_same_thread=False)

def qdf(conn, sql, args=()):
    df = pd.read_sql_query(sql, conn, params=args)
    return df

def get_ts_minmax(conn):
    try:
        df = qdf(conn, "SELECT MIN(ts) AS tmin, MAX(ts) AS tmax FROM host_metrics")
        if df.empty or pd.isna(df.loc[0, "tmin"]):
            return None, None
        return float(df.loc[0, "tmin"]), float(df.loc[0, "tmax"])
    except Exception:
        return None, None

def bin_series(ts, vals, bin_sec):
    """對 (ts, vals) 以 bin_sec 做平均分箱；回傳 new_ts, new_vals"""
    if bin_sec is None or bin_sec <= 0 or len(ts) == 0:
        return ts, vals
    s = pd.Series(vals, index=pd.to_datetime(ts, unit="s", utc=True)).astype("float64")
    rs = s.resample(f"{int(bin_sec)}S").mean()
    new_ts = (rs.index.view("int64") // 10**9).tolist()
    new_vals = rs.values.tolist()
    return new_ts, new_vals

def deriv_per_sec(ts, vals):
    """把累積量轉成速率；回傳 (ts1, rate) 與 ts[1:] 對齊"""
    if len(ts) < 2:
        return [], []
    dt = pd.Series(ts).diff().fillna(0).values
    dv = pd.Series(vals).diff().fillna(0).values
    dt[dt == 0] = 1e-6
    rate = (dv / dt)[1:]
    return ts[1:], rate.tolist()

def safe_float_list(a):
    return [ (float(x) if x is not None and not (isinstance(x, float) and math.isnan(x)) else None) for x in a ]

def ts_to_local_str(ts):
    return datetime.fromtimestamp(ts).strftime("%m-%d %H:%M:%S")

# ---------- Data loaders ----------
def load_host(conn, t_from, t_to):
    df = qdf(conn, """
        SELECT ts, cpu_percent, mem_percent, cpu_temp_c, rapl_watts
        FROM host_metrics
        WHERE ts BETWEEN ? AND ?
        ORDER BY ts
    """, (t_from, t_to))
    return df

def load_disk_total(conn, t_from, t_to):
    df = qdf(conn, """
        SELECT ts, read_bytes, write_bytes, read_count, write_count, busy_ms
        FROM disk_metrics
        WHERE ts BETWEEN ? AND ? AND device='__total__'
        ORDER BY ts
    """, (t_from, t_to))
    return df

def load_gpu(conn, t_from, t_to):
    df = qdf(conn, """
        SELECT ts, gpu_index, name, util_percent, mem_used_gb, mem_total_gb, power_w, temp_c
        FROM gpu_metrics
        WHERE ts BETWEEN ? AND ?
        ORDER BY gpu_index, ts
    """, (t_from, t_to))
    return df

# ---------- Figures ----------
def fig_lines(x, ys, names, title, ytitle):
    fig = go.Figure()
    for y, n in zip(ys, names):
        fig.add_trace(go.Scatter(x=[ts_to_local_str(t) for t in x], y=y, mode="lines", name=n))
    fig.update_layout(title=title, xaxis_title="time", yaxis_title=ytitle,
                      height=300, margin=dict(l=40,r=20,t=40,b=40))
    return fig

def make_host_figs(df_host, bin_sec):
    figs = {}
    if df_host.empty:
        return figs
    ts = df_host["ts"].astype(float).tolist()

    # CPU %
    cpu = safe_float_list(df_host["cpu_percent"].tolist())
    t, v = bin_series(ts, cpu, bin_sec)
    figs["cpu"] = fig_lines(t, [v], ["CPU %"], "Host CPU Utilization", "percent")

    # Mem %
    mem = safe_float_list(df_host["mem_percent"].tolist())
    t, v = bin_series(ts, mem, bin_sec)
    figs["mem"] = fig_lines(t, [v], ["Memory %"], "Host Memory Utilization", "percent")

    # CPU Temp
    if df_host["cpu_temp_c"].notna().any():
        ctemp = [ (float(x) if pd.notna(x) else None) for x in df_host["cpu_temp_c"].tolist() ]
        t, v = bin_series(ts, ctemp, bin_sec)
        figs["ctemp"] = fig_lines(t, [v], ["CPU temp"], "Host CPU Temperature", "°C")

    # RAPL
    if df_host["rapl_watts"].notna().any():
        rapl = [ (float(x) if pd.notna(x) else None) for x in df_host["rapl_watts"].tolist() ]
        t, v = bin_series(ts, rapl, bin_sec)
        figs["rapl"] = fig_lines(t, [v], ["CPU pkg (RAPL)"], "CPU Package Power (RAPL)", "Watts")

    return figs

def make_disk_total_figs(df_disk, bin_sec):
    figs = {}
    if df_disk.empty or len(df_disk) < 2:
        return figs
    ts = df_disk["ts"].astype(float).tolist()
    rB = df_disk["read_bytes"].astype(float).tolist()
    wB = df_disk["write_bytes"].astype(float).tolist()
    rC = df_disk["read_count"].astype(float).tolist()
    wC = df_disk["write_count"].astype(float).tolist()
    bms= df_disk["busy_ms"].astype(float).tolist()

    # Throughput
    t1, rMBps = deriv_per_sec(ts, rB); rMBps = [x/1_000_000 for x in rMBps]
    _,  wMBps = deriv_per_sec(ts, wB); wMBps = [x/1_000_000 for x in wMBps]
    t1b, rMBps = bin_series(t1, rMBps, bin_sec)
    _,    wMBps = bin_series(t1, wMBps, bin_sec)
    figs["mbps"] = fig_lines(t1b, [rMBps, wMBps], ["read MB/s","write MB/s"], "Disk Total - Throughput", "MB/s")

    # IOPS
    t2, rIOPS = deriv_per_sec(ts, rC)
    _,  wIOPS = deriv_per_sec(ts, wC)
    t2b, rIOPS = bin_series(t2, rIOPS, bin_sec)
    _,    wIOPS = bin_series(t2, wIOPS, bin_sec)
    figs["iops"] = fig_lines(t2b, [rIOPS, wIOPS], ["read IOPS","write IOPS"], "Disk Total - IOPS", "ops/s")

    # Util %
    util = []
    for (t0,b0),(t1v,b1) in zip(zip(ts, bms), zip(ts[1:], bms[1:])):
        dt = max(1e-6, t1v - t0)
        util.append(max(0.0, min(100.0, (b1 - b0) / dt / 10.0)))  # 1000ms -> 100%
    t3 = ts[1:]
    t3b, util = bin_series(t3, util, bin_sec)
    figs["util"] = fig_lines(t3b, [util], ["util %"], "Disk Total - Utilization (approx.)", "percent")
    return figs

def make_gpu_figs(df_gpu, bin_sec, selected_gpu_indices=None):
    figs = {}
    if df_gpu.empty:
        return figs

    # 過濾選擇的 GPU（若傳入 None 或空，代表顯示全部）
    if selected_gpu_indices:
        df_gpu = df_gpu[df_gpu["gpu_index"].astype(int).isin([int(i) for i in selected_gpu_indices])]

    for gidx, gdf in df_gpu.groupby("gpu_index"):
        name = str(gdf["name"].iloc[0]) if "name" in gdf else f"GPU-{gidx}"
        ts = gdf["ts"].astype(float).tolist()
        util = safe_float_list(gdf["util_percent"].tolist())
        memu = safe_float_list(gdf["mem_used_gb"].tolist())
        poww = safe_float_list(gdf["power_w"].tolist())
        temp = safe_float_list(gdf["temp_c"].tolist())

        traces = []
        tbin = None

        if any(v is not None for v in util):
            tbin, utilb = bin_series(ts, [v if v is not None else float("nan") for v in util], bin_sec)
            traces.append(("util %", utilb))
        if any(v is not None for v in memu):
            tbin2, memb = bin_series(ts, [v if v is not None else float("nan") for v in memu], bin_sec)
            tbin = tbin or tbin2
            traces.append(("mem used (GB)", memb))
        if any(v is not None for v in poww):
            tbin3, powb = bin_series(ts, [v if v is not None else float("nan") for v in poww], bin_sec)
            tbin = tbin or tbin3
            traces.append(("power W", powb))
        if any(v is not None for v in temp):
            tbin4, tempb = bin_series(ts, [v if v is not None else float("nan") for v in temp], bin_sec)
            tbin = tbin or tbin4
            traces.append(("temp °C", tempb))

        if traces and tbin:
            fig = go.Figure()
            for label, series in traces:
                fig.add_trace(go.Scatter(x=[ts_to_local_str(t) for t in tbin],
                                         y=series, mode="lines", name=label))
            fig.update_layout(title=f"GPU{gidx} - {name}",
                              xaxis_title="time", yaxis_title="value",
                              height=300, margin=dict(l=40,r=20,t=40,b=40))
            figs[int(gidx)] = fig
    return figs

# ---------- Dash App ----------
def make_app(db_path):
    conn0 = open_conn(db_path)
    tmin, tmax = get_ts_minmax(conn0)
    conn0.close()

    # 預設時間：若有資料，取 tmax 往回 2h；否則用現在
    now_epoch = datetime.now(timezone.utc).timestamp()
    if tmax is None:
        tmax = now_epoch
        tmin = tmax - 2*3600
    default_from = datetime.fromtimestamp(tmax - 2*3600).strftime("%Y-%m-%d %H:%M")
    default_to   = datetime.fromtimestamp(tmax).strftime("%Y-%m-%d %H:%M")

    app = Dash(__name__)
    app.title = "sysmon dashboard"

    app.layout = html.Div([
        html.H2("System Monitoring Dashboard (Plotly/Dash)"),
        html.Div([
            html.Label("DB path"),
            dcc.Input(id="db-path", type="text", value=db_path, style={"width":"320px","marginRight":"8px"}),

            html.Label("From"),
            dcc.Input(id="from-time", type="text", value=default_from, placeholder="YYYY-mm-dd HH:MM",
                      style={"width":"170px","marginRight":"8px"}),

            html.Label("To"),
            dcc.Input(id="to-time", type="text", value=default_to, placeholder="YYYY-mm-dd HH:MM",
                      style={"width":"170px","marginRight":"8px"}),

            html.Label("Bin seconds"),
            dcc.Input(id="bin-sec", type="number", value=60, min=0, step=10, style={"width":"120px","marginRight":"8px"}),

            html.Button("Refresh", id="refresh-btn", n_clicks=0, style={"marginRight":"12px"}),

            # Auto refresh controls
            dcc.Checklist(id="auto-on", options=[{"label":" Auto refresh","value":"on"}], value=["on"],
                          style={"marginRight":"8px"}),
            dcc.Input(id="auto-sec", type="number", value=30, min=5, step=5,
                      style={"width":"100px","marginRight":"8px"}),
            html.Span("sec", style={"marginRight":"12px"}),

            # GPU filter
            html.Label("GPU filter"),
            dcc.Dropdown(id="gpu-filter", multi=True, options=[], value=[], placeholder="All GPUs",
                         style={"minWidth":"220px"}),
        ], style={"display":"flex","flexWrap":"wrap","gap":"10px","alignItems":"center","marginBottom":"10px"}),

        # Interval timer
        dcc.Interval(id="auto-intv", interval=30_000, n_intervals=0, disabled=False),

        html.Hr(),

        html.Div([
            html.Div([dcc.Graph(id="host-cpu")], className="card"),
            html.Div([dcc.Graph(id="host-mem")], className="card"),
            html.Div([dcc.Graph(id="host-ctemp")], className="card"),
            html.Div([dcc.Graph(id="host-rapl")], className="card"),
        ], style={"display":"grid","gridTemplateColumns":"repeat(auto-fit, minmax(380px, 1fr))","gap":"12px"}),

        html.H3("Disk Total"),
        html.Div([
            html.Div([dcc.Graph(id="disk-mbps")], className="card"),
            html.Div([dcc.Graph(id="disk-iops")], className="card"),
            html.Div([dcc.Graph(id="disk-util")], className="card"),
        ], style={"display":"grid","gridTemplateColumns":"repeat(auto-fit, minmax(380px, 1fr))","gap":"12px"}),

        html.H3("GPUs"),
        html.Div(id="gpu-graphs", style={"display":"grid","gridTemplateColumns":"repeat(auto-fit, minmax(380px, 1fr))","gap":"12px"}),
    ], style={"padding":"10px"})

    # ---- Control: auto refresh interval (enable/disable + seconds) ----
    @app.callback(
        Output("auto-intv","interval"),
        Output("auto-intv","disabled"),
        Input("auto-on","value"),
        Input("auto-sec","value"),
        prevent_initial_call=False
    )
    def set_interval(auto_on_vals, sec):
        try:
            sec = int(sec or 30)
        except Exception:
            sec = 30
        sec = max(5, sec)
        enabled = ("on" in (auto_on_vals or []))
        return sec * 1000, (not enabled)

    # ---- Main refresh: triggered by interval tick, manual refresh, time/bin/db change, gpu filter change ----
    @app.callback(
        Output("host-cpu","figure"),
        Output("host-mem","figure"),
        Output("host-ctemp","figure"),
        Output("host-rapl","figure"),
        Output("disk-mbps","figure"),
        Output("disk-iops","figure"),
        Output("disk-util","figure"),
        Output("gpu-graphs","children"),
        Output("gpu-filter","options"),
        Output("gpu-filter","value"),
        Input("auto-intv","n_intervals"),
        Input("refresh-btn","n_clicks"),
        Input("gpu-filter","value"),
        State("db-path","value"),
        State("from-time","value"),
        State("to-time","value"),
        State("bin-sec","value"),
        prevent_initial_call=False
    )
    def refresh(_, __, gpu_selected, db_path_in, t_from_str, t_to_str, bin_sec):
        # parse times
        def parse_dt(s):
            return datetime.strptime(s.strip(), "%Y-%m-%d %H:%M").replace(tzinfo=None).timestamp()
        try:
            t_from = parse_dt(t_from_str)
            t_to = parse_dt(t_to_str)
            if t_to <= t_from: raise ValueError
        except Exception:
            empty = go.Figure()
            return empty, empty, empty, empty, empty, empty, empty, [html.Div("Invalid time range", style={"color":"red"})], [], []

        # open DB
        try:
            c = open_conn(db_path_in)
        except Exception as e:
            empty = go.Figure()
            return empty, empty, empty, empty, empty, empty, empty, [html.Div(f"DB open error: {e}", style={"color":"red"})], [], []

        # load data
        try:
            df_host = load_host(c, t_from, t_to)
            df_disk = load_disk_total(c, t_from, t_to)
            df_gpu  = load_gpu(c, t_from, t_to)
        finally:
            c.close()

        # build gpu dropdown options from current DB slice
        if df_gpu.empty:
            gpu_options = []
            all_gpu_indices = []
        else:
            gmeta = df_gpu.groupby("gpu_index")["name"].agg(lambda s: str(s.iloc[-1]) if len(s)>0 else "").reset_index()
            gpu_options = [{"label": f"GPU{int(r.gpu_index)} - {r.name}", "value": int(r.gpu_index)} for _, r in gmeta.iterrows()]
            all_gpu_indices = [int(r.gpu_index) for _, r in gmeta.iterrows()]

        # normalize selected list (None/[] => all)
        selected = gpu_selected if (gpu_selected is not None and len(gpu_selected)>0) else all_gpu_indices

        # make figs
        bin_sec = int(bin_sec or 0)
        host_figs = make_host_figs(df_host, bin_sec)
        disk_figs = make_disk_total_figs(df_disk, bin_sec)
        gpu_figs  = make_gpu_figs(df_gpu, bin_sec, selected_gpu_indices=selected)

        host_cpu = host_figs.get("cpu", go.Figure())
        host_mem = host_figs.get("mem", go.Figure())
        host_ctp = host_figs.get("ctemp", go.Figure())
        host_rapl= host_figs.get("rapl", go.Figure())
        disk_mbps = disk_figs.get("mbps", go.Figure())
        disk_iops = disk_figs.get("iops", go.Figure())
        disk_util = disk_figs.get("util", go.Figure())

        # GPU graph cards
        gpu_children = []
        if gpu_figs:
            for gidx in sorted(gpu_figs.keys()):
                gpu_children.append(html.Div([dcc.Graph(figure=gpu_figs[gidx])], className="card"))
        else:
            gpu_children = [html.Div("No GPU data.", style={"color":"#666"})]

        # 若使用者尚未選取任何值（或新偵測到的 GPU 變化），預設勾選全部
        selected_out = selected if selected else []

        return (host_cpu, host_mem, host_ctp, host_rapl,
                disk_mbps, disk_iops, disk_util,
                gpu_children, gpu_options, selected_out)

    return app

# ---------- CLI ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive dashboard for sysmon (Plotly/Dash)")
    parser.add_argument("--db", dest="db", required=True, help="SQLite DB from sysmon.py collector")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    args = parser.parse_args()

    db_path = args.db
    app = make_app(db_path)
    app.run(debug=False, host=args.host, port=args.port)