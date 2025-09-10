# Dependencies

```
pip install psutil pynvml matplotlib
```

# How to use

```
# 1) 持續收集（每 2 秒一筆）
python sysmon.py collect --db metrics.db --interval 2

# 2) 取區間圖表（輸出到 charts/）
python sysmon.py plot --db metrics.db --from "2025-09-08 10:00" --to "2025-09-08 12:00" --out charts
```