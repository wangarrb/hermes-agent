#!/bin/bash
# Wrapper to run hindsight weekly pipeline detached from Hermes
# Note: weekly mode does not support --include-wiki; wiki is a separate cron job
cd /home/wyr
unset HTTP_PROXY HTTPS_PROXY http_proxy https_proxy
HOME=/home/wyr HERMES_HOME=/home/wyr/.hermes \
python3 /home/wyr/.hermes/scripts/hindsight_memory_pipeline.py \
  weekly \
  --execute --confirm run-hindsight-pipeline \
  > /home/wyr/.hermes/logs/hindsight_weekly_$(date +%Y%m%d_%H%M%S).log 2>&1
