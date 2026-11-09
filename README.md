# Dependencies

```
pip install psutil matplotlib nvidia-ml-py
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

## 回看離線資料與 pandas 相容性

在 macOS / Linux 回看伺服器 UTC 時間的資料時，可將 dashboard 固定以 UTC 啟動：

```sh
TZ=UTC python sysmon_dash.py --db /absolute/path/to/metrics.db --host 127.0.0.1 --port 8050
```

From / To 請填 UTC 的 `YYYY-mm-dd HH:MM`。`Bin seconds` 可用 `60` 或 `300` 做平均分箱，`0` 則顯示原始資料。分箱已使用新版 pandas 支援的小寫秒單位 `s`，並以不依賴 DatetimeIndex 儲存精度的方式轉回 epoch seconds，避免 pandas 3 出現 `Invalid frequency: 60S` 或錯誤的 1970 年時間軸。
磁碟速率計算也會先複製需修改的陣列，以相容 pandas 3 的 Copy-on-Write 行為，避免 `assignment destination is read-only`。

安裝 dashboard 相依套件後，在專案根目錄執行回歸測試（不會啟動 dashboard 或修改資料庫）：

```sh
python -m unittest discover -s tests -v
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


# shell script

Add Permission
```
chmod +x start_sysmon_tunnel.sh stop_sysmon_tunnel.sh
```

Start tunnel
```
./start_sysmon_tunnel.sh
```

Stop tunnel
```
./stop_sysmon_tunnel.sh
```
