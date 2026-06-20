#!/bin/bash
# Wrapper to run hindsight wiki auto-maintenance (biweekly on even ISO weeks)
cd /home/wyr
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
HOME=/home/wyr HERMES_HOME=/home/wyr/.hermes \
python3 /home/wyr/.hermes/scripts/wiki_auto_maintenance_cron_runner.py \
  > /home/wyr/.hermes/logs/hindsight_wiki_$(date +%Y%m%d_%H%M%S).log 2>&1
