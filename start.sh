#!/bin/sh

mkdir -p data
# Start the data collector in the background
python sysmon.py collect --db data/metrics.db --interval ${INVERTAL} &

# Start the dash dashboard in the foreground
python sysmon_dash.py --db data/metrics.db --port ${DASH_PORT} --host ${DASH_HOST}
