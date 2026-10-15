# App 对话每日本地增量采集草稿（尚未创建）

由父线程使用 App 内建 automation_update 创建；建议每日22:45，Asia/Shanghai
（UTC+8），衔接已有23:00本地消费、00:01记忆管线和08:30日报。既有系统cron、
systemd、凭证、账号、权限和代理配置保持不变。此前将“定时系统配置”扩展理解为
连App内建任务也需再次请示，是本子任务的保守解释，不是已核实的人类用户禁令。

以下为完整任务正文：

在本任务已认证的同一用户App上下文中，收集用户可见对话的新增和新发现历史，只作
本地归档。先通过受支持read_thread确认仍可访问已授权哨兵会话
01a13ed4-a7da-71fa-ba33-e9879482c024；若被拒绝，停止并报告，不切换账号、端点、
凭证或私有存储来继续。禁止读取/root/.codex/sessions，不读凭证、隐藏推理、工具
命令/结果、内部辅助会话、其他用户账号数据和附件签名URL。

在支持的工具编排器内加载/home/wyr/.hermes/scripts/hindsight_app_export.js。
此脚本需要tools，不能用Node或shell直接执行。使用
~/.hermes/hindsight/app_archive/account-visible/scope.json作为previousScope，
调用discoverVisibleAppScope，保留已有授权ID并加入同一上下文新出现的用户可见ID，
私有原子保存scope。该函数读取当前bounded列表和Codex归档分页，不重新请求已被
明确拒绝的普通ChatGPT归档；不要换API绕过该拒绝。当前列表50项无游标，始终标明
不是全账号覆盖。

调用collectAppHistory，将公开user/assistant文本投影到
~/.hermes/hindsight/app_archive/account-visible/exports，目录700、文件600。
只保留来源kind/session/turn/item、公开正文和turn时间、截断/状态、分页元数据。
每次读取最近页以发现旧会话新增及刚完成的turn；追加版本页，禁止删除旧页。
保存每个来源的游标链和返回的pending cursor。现有helper每次从头读取；若页预算
耗尽，需在工具层用受支持read_thread(cursor=...)继续并复用同样投影，或使用足够
页预算读取到末页，不能误称helper已自动持久续点。历史续读时也要刷新头部；未完成
的链保留pending，不宣称完整。瞬时传输失败仅恢复尚未封存的游标，已拒绝访问保持
拒绝。截断、进行中/失败turn分别记录，不能当完整正文。

采集后运行/home/wyr/.hermes/hermes-agent/venv/bin/python
/home/wyr/.hermes/scripts/hindsight_app_ingest_pending.py --write。
23:00现有脚本也已接入同一幂等consumer。它仅合并独立account-visible本地快照、
app-archive账本和sidecar，不写生产sessions，不调用Hindsight或任何模型。
保持原3会话冻结447条的外部处理范围不变；这些会话的新消息和其他全部消息仍仅
本地。不得为新历史调用embedding、提取、摘要、consolidation或外部发送。

只在确实成功读取API且保存投影/游标回执后更新collection-status.json的API采集
时间和coverage；本地consumer时间不能冒充API新鲜度。保留列表上限、归档拒绝、
缺用户正文、turn边界时间近似、缺页、截断与活动状态。日报由现有入口读取这些
快照，按UTC+8前一自然日输出；CLI同源及共享历史不得直接累加。

完成时只报告来源数量、公开消息/稳定item数量、新增本地消息和文档、重跑是否0及
明确缺口，不在任务回复中输出私有对话正文。没有显式请求遥测时，LLM请求、尝试、
重试、成功、失败次数均保持未知。

以上正文无凭证或用户对话正文，仅含本地路径及一个已授权来源ID。App任务尚未
创建；父线程创建后须验证一次实际工具/文件访问，再确认每日API采集已生效。
