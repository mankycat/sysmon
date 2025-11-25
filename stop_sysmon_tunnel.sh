#!/usr/bin/env bash
# 一鍵停止 Sysmon Dashboard SSH Tunnel (macOS / Linux)

PID_FILE="/tmp/sysmon_tunnel.pid"

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "[sysmon-tunnel] 停止 SSH Tunnel (PID=$PID) ..."
    kill "$PID" 2>/dev/null || true
    sleep 1
    if kill -0 "$PID" 2>/dev/null; then
      echo "[sysmon-tunnel] 進程仍在，強制 kill -9"
      kill -9 "$PID" 2>/dev/null || true
    fi
  else
    echo "[sysmon-tunnel] PID 檔存在但進程不在，清理 PID 檔"
  fi
  rm -f "$PID_FILE"
else
  echo "[sysmon-tunnel] 找不到 PID 檔，可能 tunnel 已經關閉"
fi

#保險：順便把殘留的 ssh 轉送也殺掉（可選）
# pkill -f "ssh -p 50008 -N -L 58085:localhost:58050" 2>/dev/null || true

echo "[sysmon-tunnel] 完成"