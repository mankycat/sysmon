# Dependencies

```
pip install psutil pynvml matplotlib
```

# How to use - sysmon.py

## 1) 持續收集（每 n 秒一筆）
```

python sysmon.py collect --db metrics.db --interval 2

# 收集（NVML；若抓不到改成 --gpu-via smi）
python sysmon.py collect --db metrics.db --interval 2 --debug
python sysmon.py collect --db metrics.db --interval 2 --gpu-via smi
```


## 2) 取區間圖表（輸出到 charts/）
```
python sysmon.py plot --db metrics.db --from "2025-09-10 09:00" --to "2025-09-10 12:00" --out charts

# 例如把 2s 原始資料，以 60 秒分箱後繪圖
python sysmon.py plot --db metrics.db --from "2025-09-10 00:00" --to "2025-09-11 00:00" --out charts --bin-seconds 60
```

# Dashboard (web) - sysmon_dash.py

Requirements
```
pip install dash plotly pandas
```

activate
```
python sysmon_dash.py --db metrics.db --port 8050 --host 0.0.0.0
# (可改 --port 與 --host)
```



# linux system service for sysmon.py
```
sudo nano /etc/systemd/system/sysmon.service
```

```
[Unit]
Description=SysMon Collector (CPU/GPU/Disk -> SQLite)
Wants=network-online.target
After=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=<# put working path here>
Environment=SYSMON_DEBUG=0
ExecStart=<python venv>/bin/python <python file path>/sysmon.py collect \
  --db /data/bruce/metrics.db --interval 2 --gpu-via smi
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

# linux system service for dashboard
```
sudo nano /etc/systemd/system/sysmon-dash.service
```

```
[Unit]
Description=SysMon Plotly Dashboard
After=network.target sysmon.service

[Service]
Type=simple
User=root
WorkingDirectory=<# put working path here>
ExecStart=<python venv path>/bin/python <python path>/sysmon_dash.py \
  --db <output db path>/metrics.db --host 0.0.0.0 --port <listen port>
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```