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

Dashboard 的篩選與圖表時間固定為 UTC，不再依賴 Mac / Linux 本機時區或 `TZ` 環境變數。回看已完成的 run，建議：

```sh
python sysmon_dash.py --run-dir /absolute/path/to/RUN_ID \
  --buffer-minutes 60 --bin-seconds 60 --host 127.0.0.1 --port 8050
```

From / To 請填 UTC 的 `YYYY-mm-dd HH:MM`。`Bin seconds` 可用 `60` 或 `300` 做平均分箱，`0` 則顯示原始資料。分箱已使用新版 pandas 支援的小寫秒單位 `s`，並以不依賴 DatetimeIndex 儲存精度的方式轉回 epoch seconds，避免 pandas 3 出現 `Invalid frequency: 60S` 或錯誤的 1970 年時間軸。
`--run-dir` 會讀取 `metadata/start-time.txt` / `end-time.txt`，自動定位 `metrics/sysmon/metrics.db`。預設前後各加 60 分鐘，並向外取整至分鐘。Auto refresh 預設關閉；需要時才加 `--auto-refresh --refresh-seconds 30`。
舊的 `--db` 指令仍可使用：若 DB 位於標準 run 結構，會自動找到 metadata；否則回看 DB 最後兩小時。亦可同時提供 `--from` 與 `--to`（ISO 8601 或 epoch；不帶時區的 CLI 輸入視為 UTC），覆寫 metadata；`--buffer-minutes 0` 可取消 buffer。缺少或無效的 run metadata 會明確報錯，不會默默換成其他 run 的時間。
磁碟速率計算也會先複製需修改的陣列，以相容 pandas 3 的 Copy-on-Write 行為，避免 `assignment destination is read-only`。

安裝 dashboard 相依套件後，在專案根目錄執行回歸測試（不會啟動 dashboard 或修改資料庫）：

```sh
python -m unittest discover -s tests -v
```

## 匯出訓練時間窗統計

新增的 `sysmon_stats.py` 只需 Python 標準函式庫，不需 GPU / Docker / pandas。它以唯讀方式開 DB，預設只統計 metadata 的 start 到 end，不含 dashboard 的前後 buffer，不需重收 metrics。

```sh
python sysmon_stats.py --run-dir /absolute/path/to/RUN_ID \
  --expected-gpus 8 --out-dir /absolute/path/to/RUN_ID/qa-summary

# 不使用 metadata，直接指定 UTC 時間窗
python sysmon_stats.py --db /absolute/path/to/metrics.db \
  --from '2026-11-06T13:18:39+00:00' --to '2026-11-08T13:58:13+00:00' \
  --expected-gpus 8 --out-dir /absolute/path/to/summary
```

輸出 `summary.json` 與 `summary.md`（可直接閱讀或貼入 wiki）。既有輸出預設不覆蓋；改用新目錄，或明確加 `--overwrite`。已知 collector 採樣週期時可加 `--sampling-interval 60`，否則依各資料序列的時間差中位數估算。需要把閒置上下文算進統計時，才明確加 `--buffer-minutes 60`。

- 每張 GPU、全部 GPU 樣本合併分布：utilization、溫度、功耗、VRAM 的 mean / P95 / peak / 有效與缺失樣本數。VRAM 的原始欄位雖名為 `_gb`，collector 以 1024 進位收集，統計單位標為 GiB。合併功耗平均不是整機 GPU 總功耗。
- 同時點的 GPU 總功耗與總 VRAM：只採用預期所有 GPU 都有資料、且指標有效的同一時間戳。這不是整台 server 或插座功耗。
- Host：CPU / RAM 百分比、CPU 溫度與 RAPL 功耗；DB 未收集的項目顯示 null / N/A，不當成零。
- Disk：相鄰有效計數器差值推算的讀寫 MB/s、IOPS、近似 utilization；重置造成的負差值不納入統計。
- 資料完整性：樣本起訖、採樣中位間隔、最大缺口、缺 GPU / 缺值 / 邊界未覆蓋等警告。

Mean 為原始有效樣本的算術平均，不做分箱、補零或時間加權。P95 使用排序後 `(n-1)*0.95` 線性內插，並不是「利用率達 95%」。Peak 是採樣最大值，不是未採樣到的瞬間最大值。斷點或缺值不內插；若採樣不均勻，平均值不代表精確的時間加權負載。請保留警告並人工判讀，不以報表自動宣告硬體 PASS。



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
