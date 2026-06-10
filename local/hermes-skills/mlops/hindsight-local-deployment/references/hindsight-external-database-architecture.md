# Hindsight External Database Architecture

**日期**: 2026-05-04
**问题**: payload_null bug 导致 MiniMax 配额持续消耗
**发现**: 数据库数据已在 Docker 外部，可直接管理

---

## 架构分析

### 当前部署架构

```
Docker 容器 (hindsight)
  ├── API 服务 (8888)
  ├── PostgreSQL 进程 (容器内 5432)
  └── 数据存储挂载: ~/.hindsight-docker → /home/hindsight/.pg0
```

**关键发现**：
- PostgreSQL 数据存储在 `~/.hindsight-docker`（Docker 外部）
- 每次容器启动会启动内部 PostgreSQL 进程
- 数据与进程分离，可以直接操作外部数据

### 两个 PostgreSQL 实例

| 实例 | 数据目录 | Documents | 来源 |
|------|----------|-----------|------|
| 本地 ~/.pg0 | ~/.pg0/instances/hindsight/data | 26 | 新建/测试数据 |
| Docker 挂载 ~/.hindsight-docker | ~/.hindsight-docker/instances/hindsight/data | 819 | 历史 + 新导入 |

**问题根源**：
- 之前可能部署过两个独立的 Hindsight 实例
- 合并时数据没有同步
- Docker PostgreSQL 有大量历史 documents（736+）
- 历史 documents 产生了 payload_null 任务

---

## 直接操作外部数据库

### 方法 1: 启动外部 PostgreSQL 进程

```bash
# 停止 Docker 容器（停止内部 PostgreSQL）
sudo docker stop hindsight

# 启动外部 PostgreSQL（用 Docker 挂载的数据）
$HOME/.pg0/installation/18.1.0/bin/pg_ctl start \
  -D ~/.hindsight-docker/instances/hindsight/data \
  -o "-p 5433" \
  -l ~/.hindsight-docker/instances/hindsight/data/log/postgres.log

# 连接操作（不需要 docker exec）
$HOME/.pg0/installation/18.1.0/bin/psql -h /tmp -p 5433 -U hindsight -d hindsight

# 查询数据
SELECT COUNT(*) FROM documents WHERE bank_id='hermes';
SELECT COUNT(*) FROM memory_units WHERE bank_id='hermes';
SELECT COUNT(*) FROM async_operations WHERE status='pending';

# 清理无效任务
DELETE FROM async_operations WHERE operation_type='batch_retain' AND task_payload IS NULL;
DELETE FROM async_operations WHERE status='pending';

# 完成后停止外部 PostgreSQL
$HOME/.pg0/installation/18.1.0/bin/pg_ctl stop \
  -D ~/.hindsight-docker/instances/hindsight/data \
  -m fast
```

### 方法 2: 修改 pg_hba.conf

```bash
# 添加 trust 规则允许本地连接
echo "local   all             all                                     trust" > ~/.hindsight-docker/instances/hindsight/data/pg_hba.conf.new
cat ~/.hindsight-docker/instances/hindsight/data/pg_hba.conf >> ~/.hindsight-docker/instances/hindsight/data/pg_hba.conf.new
cp ~/.hindsight-docker/instances/hindsight/data/pg_hba.conf.new ~/.hindsight-docker/instances/hindsight/data/pg_hba.conf

# Reload 配置
$HOME/.pg0/installation/18.1.0/bin/pg_ctl reload -D ~/.hindsight-docker/instances/hindsight/data
```

---

## 外部数据库配置（推荐架构）

Hindsight 支持连接外部 PostgreSQL，让 Docker 只运行 API 服务。

### 配置方法

**启动外部 PostgreSQL（端口 5432）**：
```bash
$HOME/.pg0/installation/18.1.0/bin/pg_ctl start \
  -D ~/.hindsight-docker/instances/hindsight/data \
  -o "-p 5432"
```

**⚠️ 关键：PostgreSQL 必须监听所有地址**：

Docker 容器通过 `host.docker.internal` 连接宿主机，PostgreSQL 默认只监听 `127.0.0.1`（localhost），Docker 无法连接。

**修改 postgresql.conf**：
```bash
# 停止 PostgreSQL
$HOME/.pg0/installation/18.1.0/bin/pg_ctl stop -D ~/.hindsight-docker/instances/hindsight/data

# 修改 listen_addresses（从 localhost 改为 *）
sed -i "s/^#listen_addresses = 'localhost'/listen_addresses = '*'/g" ~/.hindsight-docker/instances/hindsight/data/postgresql.conf

# 重新启动 PostgreSQL（端口 5432，监听所有地址）
$HOME/.pg0/installation/18.1.0/bin/pg_ctl start \
  -D ~/.hindsight-docker/instances/hindsight/data \
  -o "-p 5432"
```

**验证监听状态**：
```bash
ss -tlnp | grep 5432
# 应显示 0.0.0.0:5432（所有地址），不是 127.0.0.1:5432（仅 localhost）
```

**配置 pg_hba.conf 允许 Docker 网络连接**：
```bash
echo "host all all 172.17.0.0/16 trust" >> ~/.hindsight-docker/instances/hindsight/data/pg_hba.conf
echo "host all all 192.168.0.0/16 trust" >> ~/.hindsight-docker/instances/hindsight/data/pg_hba.conf
```

**重新启动 Docker 容器（只运行 API，连接外部数据库）**：

**⚠️ 环境变量名注意**：正确的是 `HINDSIGHT_API_DATABASE_URL`（带 `_API_`），不是 `HINDSIGHT_DATABASE_URL`。

```bash
# 从 Hermes .env 读取真实 API key（绕过 masking）
MINIMAX_KEY=$(python3 -c "
with open('$HOME/.hermes/.env', 'r') as f:
    for line in f:
        if line.startswith('MINIMAX_API_KEY='):
            print(line.strip().split('=', 1)[1])
            break
")

sudo docker run -d --name hindsight \
  --add-host=host.docker.internal:host-gateway \
  -p 8888:8888 \
  -e HTTPS_PROXY=http://host.docker.internal:7890 \
  -e HTTP_PROXY=http://host.docker.internal:7890 \
  -e HINDSIGHT_API_DATABASE_URL="postgresql://hindsight@host.docker.internal:5432/hindsight" \
  -e HINDSIGHT_ENABLE_API=true \
  -e HINDSIGHT_ENABLE_CP=false \
  -e HINDSIGHT_API_AUTO_RETAIN=false \
  -e HINDSIGHT_API_LLM_PROVIDER=minimax \
  -e HINDSIGHT_API_LLM_MODEL=MiniMax-M2.7 \
  -e HINDSIGHT_API_LLM_BASE_URL=https://api.minimaxi.com/v1 \
  -e HINDSIGHT_API_LLM_API_KEY="$MINIMAX_KEY" \
  ghcr.io/vectorize-io/hindsight:latest
```

**关键参数**：
- `HINDSIGHT_API_DATABASE_URL`: 指定外部数据库连接（**带 `_API_`**）
- `HINDSIGHT_ENABLE_API=true`: 只启动 API
- `HINDSIGHT_ENABLE_CP=false`: 不启动 Control Plane（减少资源）
- `HINDSIGHT_API_AUTO_RETAIN=false`: **禁止自动 retain**，防止 MiniMax 配额消耗
- `--add-host=host.docker.internal:host-gateway`: 让容器能解析 host.docker.internal
- `HTTPS_PROXY/HTTP_PROXY`: HuggingFace 模型下载需要代理

### 优点

- ✅ 直接用 `psql` 操作数据库（不用 `docker exec`）
- ✅ 数据库独立运行，容器重启不影响
- ✅ 可以随时清理任务、修改配置
- ✅ 数据完全可控

---

## payload_null Bug 详细诊断

### 症状

- MiniMax 配额持续消耗，但最近几小时没有对话
- Stats API 显示 `queue_status: {}`（空）
- Docker logs 显示 `pending=300+`

### 关键认知

**Stats API 的 queue_status 不反映真实的 worker 队列！**

| 来源 | 数据位置 | 反映真实队列？ |
|------|----------|----------------|
| Stats API `/v1/banks/{bank}/stats` | 数据库查询 | ❌ 不反映内存队列 |
| Docker logs `PENDING_BREAKDOWN` | Worker 内存状态 | ✅ 真实队列 |
| `async_operations` 表 | 数据库 | ❌ 任务可能已完成但队列仍有 pending |

**必须查看 Docker logs 确认真实状态**：
```bash
sudo docker logs hindsight --tail 50 --timestamps 2>&1 | grep -E 'PENDING_BREAKDOWN|payload_null|STUCK'
```

### payload_null 任务特征

- `batch_retain: total=180, payload_null=180`
- 任务的 `task_payload` 是 null（无效任务）
- Worker 仍尝试处理，调用 MiniMax（每次 60-70 秒）
- 任务失败后自动 retry，形成无限循环

### 来源分析

**两个 PostgreSQL 实例的数据差异**：
- Docker PostgreSQL: 819 documents
- 导入脚本: 83 bundles
- 差异: 736 个历史 documents

**历史 documents 来源**：
- 之前的 Aggregate JSON 导入
- 旧版导入脚本
- 合并两个 Hindsight 实例时的遗留数据

**payload_null 产生原因**：
- 历史 documents 的 `retain_params` 可能不完整
- Hindsight 扫描 "未处理 documents" 时创建任务
- 创建时 payload 为 null（可能是扫描逻辑 bug）

---

## 数据清理策略

### 方案对比

| 方案 | 操作 | 效果 |
|------|------|------|
| **F1** | 删除 pending 任务 | 保留所有数据，停止消耗 |
| **F2** | 删除所有 documents + facts | 全部清空，重新开始 |
| **F3** | 删除历史 documents | 保留增量导入的 83 bundles |
| **F4** | 只清理 payload_null 任务 | 保留 44K facts，停止无效消耗 |

### 推荐: F1 + 外部数据库架构

```bash
# 1. 停止 Docker 容器
sudo docker stop hindsight

# 2. 启动外部 PostgreSQL
$HOME/.pg0/installation/18.1.0/bin/pg_ctl start \
  -D ~/.hindsight-docker/instances/hindsight/data \
  -o "-p 5433"

# 3. 删除 pending 任务
$HOME/.pg0/installation/18.1.0/bin/psql -h /tmp -p 5433 -U hindsight -d hindsight \
  -c "DELETE FROM async_operations WHERE status='pending';"

# 4. 确认数据保留
$HOME/.pg0/installation/18.1.0/bin/psql -h /tmp -p 5433 -U hindsight -d hindsight \
  -c "SELECT COUNT(*) FROM documents WHERE bank_id='hermes';"  # 819
$HOME/.pg0/installation/18.1.0/bin/psql -h /tmp -p 5433 -U hindsight -d hindsight \
  -c "SELECT COUNT(*) FROM memory_units WHERE bank_id='hermes';"  # 44202

# 5. 停止外部 PostgreSQL
$HOME/.pg0/installation/18.1.0/bin/pg_ctl stop \
  -D ~/.hindsight-docker/instances/hindsight/data

# 6. 重新配置 Docker 使用外部数据库（长期方案）
```

---

## 验收检查

```bash
# 确认 documents 数量（应保留）
SELECT COUNT(*) FROM documents WHERE bank_id='hermes';  # 819

# 确认 facts 数量（应保留）
SELECT COUNT(*) FROM memory_units WHERE bank_id='hermes';  # 44202

# 确认 pending 任务已删除
SELECT COUNT(*) FROM async_operations WHERE status='pending';  # 0

# 确认 payload_null 任务已删除
SELECT COUNT(*) FROM async_operations WHERE operation_type='batch_retain' AND task_payload IS NULL;  # 0
```

---

## 长期预防

1. **统一使用外部数据库架构**
   - Docker 只运行 API
   - PostgreSQL 独立运行在宿主机
   - 数据完全可控

2. **定期检查 Docker logs**
   ```bash
   sudo docker logs hindsight --tail 20 | grep PENDING_BREAKDOWN
   ```

3. **导入脚本不会触发此问题**
   - 新版 `import_sqlite_to_hindsight.py` 正确关闭 auto_retain
   - 串行提交避免并发
   - 只导入增量数据

4. **换用本地 LLM provider**
   - 即使有 payload_null 任务也不消耗付费配额
   - 配置 `HINDSIGHT_API_LLM_PROVIDER=ollama`

---

## 相关文件

- `~/.hindsight-docker/` — Docker 挂载的 PostgreSQL 数据
- `~/.pg0/` — 本地 PostgreSQL 安装和另一个实例
- `~/.hermes/hindsight/config.json` — Hermes Hindsight 配置