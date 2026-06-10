# NO_PROXY CIDR 格式不兼容问题

## 症状

Hindsight recall/retain/reflect 等 API 调用失败，报错：
```
Failed to search memory: Cannot connect to host 127.0.0.1:7890 ssl:default
```

而直接 `curl http://127.0.0.1:8888/health` 和 `docker ps` 都正常——Hindsight 服务本身在运行。

## 根因

`hindsight_client_api/rest.py` 中的 `aiohttp.ClientSession(trust_env=True)` 会自动读取环境变量 `http_proxy`/`https_proxy`/`no_proxy` 来决定是否走代理。

Python 的 `urllib.request.proxy_bypass()` 函数（aiohttp 底层依赖）**不支持 CIDR 格式**的 NO_PROXY。它只支持：
- 精确 IP（`127.0.0.1`）
- 域名精确匹配（`localhost`）
- 域名通配（`.example.com`）
- `*`（完全绕过）

它**不支持**：
- `127.0.0.0/8` —— CIDR 子网表示法
- `192.168.0.0/16`

**验证**：
```bash
python3 -c "
import os, urllib.request
os.environ['no_proxy'] = 'localhost,127.0.0.0/8'
print(urllib.request.proxy_bypass('127.0.0.1'))
# 返回 False = 走代理（错误），应返回 True = 绕过（正确）
"
```

## 修复

在 `.env` 中把 `127.0.0.0/8` 改为精确 IP `127.0.0.1`：

```bash
# 修改前（不支持 CIDR）
NO_PROXY=localhost,127.0.0.0/8,192.168.0.0/16,...

# 修改后（精确 IP）
NO_PROXY=localhost,127.0.0.1,192.168.0.0/16,...
```

也可以用 `sed` 修改：
```bash
sed -i 's|127.0.0.0/8|127.0.0.1|' ~/.hermes/.env
```

## 生效条件

修改 `.env` 后，Hermes 新进程（新 session）会读取新的环境变量。当前已在运行的 session 需要 `/reset` 或重启 Hermes 才能生效。

## 影响范围

所有使用了 `hindsight_client_api`（底层依赖 `aiohttp` 的 `trust_env=True`）的 Python 进程都会受影响。包括：
- Hermes 中的 HindsightMemoryProvider（作为 plugin 加载）
- 任何直接使用 `hindsight_client` Python 包的脚本
- 其他使用了 Python `urllib` 或 `aiohttp` 且设置了 `trust_env=True` 的应用

注意修改 `.env` 会影响整个系统的代理行为，不仅仅是 Hindsight。

## 参考

- `hindsight_client_api/rest.py` 第 93 行：`self._pool_manager = aiohttp.ClientSession(connector=connector, trust_env=True)`
- `urllib.request.proxy_bypass` 源码：`Lib/urllib/request.py` 中只支持精确匹配和通配符，不支持 CIDR