FROM nvcr.io/nvidia/cuda:12.8.1-cudnn-devel-rockylinux9

RUN dnf install -y python3 && \
    dnf install -y python3-pip && \
    dnf install -y mesa-libGL && \
    ln -sf /usr/bin/python3 /usr/local/bin/python && \
    ln -sf /usr/bin/pip3 /usr/local/bin/pip
RUN pip install --no-cache-dir psutil nvidia-ml-py matplotlib dash plotly pandas

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