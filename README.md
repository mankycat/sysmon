# Dependencies

```
pip install psutil pynvml matplotlib
```

# How to use

```
# 1) 持續收集（每 2 秒一筆）
python sysmon.py collect --db metrics.db --interval 2

# 收集（NVML；若抓不到改成 --gpu-via smi）
python sysmon.py collect --db metrics.db --interval 2 --debug
python sysmon.py collect --db metrics.db --interval 2 --gpu-via smi


# 2) 取區間圖表（輸出到 charts/）
python sysmon.py plot --db metrics.db --from "2025-09-10 09:00" --to "2025-09-10 12:00" --out charts
```