#!/bin/bash
# Wrapper to run hindsight daily pipeline detached from Hermes
cd /home/wyr
python3 /home/wyr/.hermes/scripts/hindsight_daily_noagent.py > /home/wyr/.hermes/logs/hindsight_daily_$(date +%Y%m%d_%H%M%S).log 2>&1