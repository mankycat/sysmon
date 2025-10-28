FROM python:3.14-alpine

RUN pip install --no-cache-dir psutil pynvml matplotlib dash plotly pandas

WORKDIR /app
COPY sysmon.py .
COPY sysmon_dash.py .
COPY start.sh .

RUN chmod +x start.sh

ENV INVERTAL=60
ENV DASH_PORT=8050
ENV DASH_HOST=0.0.0.0

# 執行啟動腳本
CMD ["./start.sh"]