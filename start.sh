#!/bin/sh

# Start the data collector in the background
python sysmon.py collect --db metrics.db --interval ${INVERTAL} &

# Start the dash dashboard in the foreground
python sysmon_dash.py --db metrics.db --port ${DASH_PORT} --host ${DASH_HOST}
