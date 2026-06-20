#!/bin/bash
# Wrapper to run hindsight daily pipeline detached from Hermes
cd /home/wyr
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
HOME=/home/wyr HERMES_HOME=/home/wyr/.hermes \
python3 /home/wyr/.hermes/scripts/hindsight_daily_noagent.py \
  > /home/wyr/.hermes/logs/hindsight_daily_$(date +%Y%m%d_%H%M%S).log 2>&1
