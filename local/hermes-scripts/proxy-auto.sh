#!/bin/bash
case "$1" in
  status)
    systemctl --user status hermes-proxy-monitor.service --no-pager | head -15
    echo ""
    echo "=== 最近日志 ==="
    tail -3 ~/.hermes/logs/proxy-monitor.log 2>/dev/null
    ;;
  stop)
    systemctl --user stop hermes-proxy-monitor.service
    echo "✓ 监控已停止"
    ;;
  start)
    systemctl --user start hermes-proxy-monitor.service
    echo "✓ 监控已启动"
    ;;
  restart)
    systemctl --user restart hermes-proxy-monitor.service
    echo "✓ 监控已重启"
    ;;
  log)
    tail -20 ~/.hermes/logs/proxy-monitor.log
    ;;
  test)
    source ~/.hermes/.env
    curl -x "$https_proxy" --max-time 10 -o /dev/null -s -w "代理: $https_proxy\n状态: %{http_code} 时间: %{time_total}s\n" \
      https://opencode.ai/zen/go/v1/models
    ;;
  *)
    echo "用法: proxy-auto {status|start|stop|restart|log|test}"
    echo ""
    echo "  status  - 查看服务状态"
    echo "  start   - 启动监控"
    echo "  stop    - 停止监控"
    echo "  restart - 重启监控"
    echo "  log     - 查看日志"
    echo "  test    - 手动测试代理"
    ;;
esac
