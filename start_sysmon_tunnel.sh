#!/usr/bin/env bash
# 一鍵啟動 Sysmon Dashboard SSH Tunnel (macOS / Linux)

# === 可依需要修改的參數 ===
REMOTE_USER="root" # remote user
REMOTE_HOST="" # remote ip
SSH_PORT=50008 # ssh port

LOCAL_PORT=58050        # 本機瀏覽器要連的 port
REMOTE_PORT=58050       # 遠端 sysmon dashboard listen 的 port

PID_FILE="/tmp/sysmon_tunnel.pid"

# === 不用改的部分 ===
set -e

if [[ -f "$PID_FILE" ]]; then
  PID=$(cat "$PID_FILE" 2>/dev/null || echo "")
  if [[ -n "$PID" ]] && kill -0 "$PID" 2>/dev/null; then
    echo "[sysmon-tunnel] 已經在執行中 (PID=$PID)"
    echo "[sysmon-tunnel] 打開瀏覽器訪問: http://localhost:${LOCAL_PORT}"
    exit 0
  else
    echo "[sysmon-tunnel] 發現殘留 PID 檔，將其移除"
    rm -f "$PID_FILE"
  fi
fi

echo "[sysmon-tunnel] 建立 SSH Tunnel ..."
echo "[sysmon-tunnel] 本機 http://localhost:${LOCAL_PORT} -> ${REMOTE_HOST}:${REMOTE_PORT} (SSH port ${SSH_PORT})"

# -N: 不執行遠端命令
# -L: 本地 port 轉遠端
# -o ServerAlive*: 避免長時間 idle 被關閉
ssh -p "${SSH_PORT}" \
    -N \
    -L "${LOCAL_PORT}:localhost:${REMOTE_PORT}" \
    -o ServerAliveInterval=30 \
    -o ServerAliveCountMax=3 \
    "${REMOTE_USER}@${REMOTE_HOST}" &

PID=$!
echo "$PID" > "$PID_FILE"

echo "[sysmon-tunnel] 啟動完成，PID=${PID}"
echo "[sysmon-tunnel] 請在瀏覽器開啟: http://localhost:${LOCAL_PORT}"