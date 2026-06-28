#!/bin/bash
# Hermes 代理切换脚本
ENV_FILE="$HOME/.hermes/.env"

case "$1" in
  win|windows)
    sed -i 's|http://127.0.0.1:7890|http://192.168.30.138:7897|g' "$ENV_FILE"
    sed -i 's|http://192.168.30.138:7897|http://192.168.30.138:7897|g' "$ENV_FILE"
    echo "✓ 切换到 Windows 代理 (192.168.30.138:7897)"
    ;;
  local|backup)
    sed -i 's|http://192.168.30.138:7897|http://127.0.0.1:7890|g' "$ENV_FILE"
    echo "✓ 切换到本地代理 (127.0.0.1:7890)"
    ;;
  status|show)
    grep -E "^(https_proxy|http_proxy)=" "$ENV_FILE"
    ;;
  test)
    proxy=$(grep "^https_proxy=" "$ENV_FILE" | cut -d'=' -f2)
    echo "测试当前代理: $proxy"
    curl -x "$proxy" --max-time 10 -o /dev/null -s -w "状态: %{http_code} 时间: %{time_total}s\n" \
      https://opencode.ai/zen/go/v1/models \
      -H "Authorization: Bearer $(grep OPENCODE_GO_API_KEY "$ENV_FILE" | cut -d'=' -f2)"
    ;;
  *)
    echo "用法: switch-proxy.sh {win|local|status|test}"
    echo "  win     - 切换到 Windows 代理"
    echo "  local   - 切换到本地备用代理"
    echo "  status  - 显示当前代理"
    echo "  test    - 测试当前代理连通性"
    ;;
esac
