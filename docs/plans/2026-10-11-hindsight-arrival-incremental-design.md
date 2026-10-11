# Hindsight：按到达增量、禁止历史重放及可恢复执行

## 1. 用户批准的目标与决策变更

用户要求：成功完成记忆之后仅处理新增数据，最好永远不要回补；不要动不动失败后需要人工全量重跑。已批准：保留现有记忆，禁止自动历史重放，修复端到端增量与失败恢复。迁移中无法证明完成的旧缺口单列为历史待核查，不自动补跑，也不得标为成功。

本规格替换原 runbook 的「weekly 必须全历史付费 LLM integration」。新的正常路径为：新到达数据 → 冻结不可变小批次 → 逐单元远端确认 → 新增摘要与现有记忆的本地合并 → 同一冻结候选的质量门禁与发布。全历史的本地只读校验可以保留，但历史原文/已封存摘要不得因调度、模型切换、文件缺失或缓存变动重新提交给 LLM。

本规格是实现契约，不是实现完成报告。不通过门禁时保留旧发布版本，不自动修复历史，也不降低质量阈值。

## 2. 已核实的现状与反例

源码根为 `local/hermes-scripts/`。

- `offline_hindsight_reflect_consolidate.py:678–688,1435–1477` 读取所有日期及所有 daily Markdown generation；`hindsight_memory_pipeline.py:308–374` 固定 weekly all-history。初始 375 pending 中，16 为 daily、359 为 weekly；weekly 历史输入 18,622,002 字符，不是新到达数据量。
- 日期 checkpoint 使用 event_date，不是到达消费游标；`latest_successful_date=2026-10-11` 是有洞的最大日期，`baseline_through=2026-09-23`。不能把二者或当前时间直接当作所有数据已经处理完。
- 会话以 session/part document ID replace；增长会话重新分块和提交旧正文。lean manifest 执行时从可变源文件重新取正文，可能与冻结 metadata/hash 不一致。
- async daily POST 接受即记录 processed，未保存或核验 operation ID；session runner 把 failed 操作移出等待集合，随后把 selected records 写为成功。父任务运行既有 `test_execute_does_not_update_submit_state_when_async_operation_fails` 已复现该失败。
- RateLimitError 被 broad except 捕获，自适应 429 分支失效；永久 400/401 也重试；无单元总期限/实际调用预算。stdout/stderr 在子进程结束后分开重放，运行中缺少有效进度。
- full --skip-daily 的 weekly catch-up 没有完整 observations 窗口责任。失败恢复后存在 pending_consolidation=56、操作队列空且开关 false；队列空不能当作 consolidation 完成。
- publisher 仅检查 manifest 存在却 glob 卡片目录。候选目录存在不在 manifest 中的旧卡片，可能混合 generation。
- 当前实际生产 canonical 为 Oct3 写入的 91 documents / 74,369 observations，非 NULL 1024 维向量；latest 本地候选和旧 gate 不是其可靠替代。对应 rebuild ID 为 `offline-v2-rebuild::20261002-165118::7bce4665`。

只读审计：`/home/wyr/.hermes/cache/scratch/hindsight-weekly-v2-incrementality-readonly-audit.md`。现有安装 0.9.0 的 OpenAPI 已确认支持 caller-supplied UUID operation_id；相同 UUID 的调用方必须绑定同一 payload/hash。

## 3. 范围和不变式

### 范围

现有 Hermes/Codex 数据来源的增量发现、清洗、retain、daily 反思、canonical 本地合并、gate/publish、执行报告和 observations 生命周期。保留已有 bank、One API 路由、模型及本地 embedding；不增加备用 provider，不更改其他 Hermes profile 配置。

复用现有清洗、敏感数据过滤、bank 隔离、原子写入、operation 查询、Retry-After 解析、flock 和 reducer 的投影/卡片组件。新的状态逻辑放在局部小模块，不继续往超大 facade 追加整个框架，不新增依赖或常驻进程。

### 硬不变式

1. 已确认成功的 input/batch 不因日期、模型、label、endpoint、分块索引、旧缓存文件变化而重放。
2. 输入 identity 与执行 recipe/provenance 分离；换模型只影响尚未生成的新增或失败单元，不使 sealed input 重新 pending。
3. event_date 是事件语义，不是增量边界。新到达的旧事件作为新的 arrival batch 处理，不重开旧 daily 文件。
4. 无新输入且无到期 pending 时，付费 LLM、retain POST、consolidation trigger、embedding 和 publish 均为零；允许只读 health/status。
5. async accepted 不是 retained/consolidated 成功；未知状态不推进成功水位。
6. 成功阶段不可由后续 gate/publish/wiki 失败反向失效。只能恢复未完成单元/阶段。
7. 失败、退出、清理异常均形成持久报告；pending/deferred/degraded 不冒充 completed。
8. 不把 max(successful date)、run finished_at 或 cutover_at 伪造为输入完成证明。
9. 历史未知/缺失资料不触发自动 LLM replay；原始材料与现有记忆不删除。

## 4. 增量边界与源数据身份

### 4.1 到达与成功的区别

每轮开始冻结输入上界 H，读取尚未捕获的新到达范围并保留低位 pending。成功后提交对应已确认范围，last_success_at 仅是回执时间。运行期间到达的数据留给后续批次，不使用运行结束时间跳过它们。

不直接把 `(created_at, id)` 当作无损 CDC：晚提交事务、旧记录更新及数据库重置需要处理。源适配器应记录来源 epoch 和稳定事件身份；已捕获 immutable batch 的内容是恢复依据。Hermes SQLite message ID 可用于发现，但压缩会克隆消息，必须结合来源 lineage/display generation/稳定消息 fingerprint 避免把旧副本识别为新对话。JSON/Codex snapshot 是可覆盖视图；需要内容前后稳定性校验以及已捕获消息范围，而不是仅用文件 mtime。

源数据被外部删除或覆盖后无法凭空恢复；本实现保证已捕获数据持久恢复，不虚称只靠间歇快照具备完整 CDC。来源 epoch 变化或历史截断进入明确 source_changed 状态，不自动从头扫描并付费重放。

### 4.2 新增会话内容

已成功捕获的 session prefix/range 不重新清洗分块。仅新增消息范围冻结到新的批次，批次 document ID 绑定 bank/profile/source epoch/session/范围及 payload hash。禁止把后缀发送到旧 session document ID；Hindsight upsert 会替换旧文档及其关联记忆。

事件去重保留 role、源消息 timestamp、可用的原始稳定 ID/lineage及内容 hash；不能只按文本去重，否则不同时间的合法重复事件会漏记。必要上下文是引用已有记忆的有限上下文，不把已成功正文当作新增待 retain 内容。

### 4.3 冻结内容

manifest 引用权限受控的 immutable batch payload，并包含 SHA-256、来源范围、清洗版本、bank、profile、source epoch。执行时验证冻结 payload，而不是用旧 manifest metadata 配当前文件的新正文。新文件增长进入下一轮；发现不一致时不提交。

LLM 输入按 record/message 边界组成小批次，不截断事实，不重切已 sealed block。source identity 不含易变输出路径/周标签/数组索引。

## 5. 状态账本和恢复

一个本地 bank-scoped 事务型账本（Python stdlib SQLite），无新服务。以本地 SQLite 提供 batch 状态、输入覆盖、stage receipts、租约/并发 fencing 的一致性；原始内容放在权限 0600 的受控冻结文件，不塞入状态摘要或日志。既有 progress/checkpoint 文件保持兼容只读审计，不把旧 processed 集合无条件迁移成成功。

状态对象：

- source：bank/profile/epoch、discovery cursor、已捕获稳定事件 identity。
- batch：稳定 batch_id、范围和 payload hash、immutable payload path、phase、recipe、operation UUID、attempts、next_retry_at、last_error_class。
- stage receipt：generated artifact hash/path；operation/children remote success；consolidation proof；frozen merge candidate/gate/publish/readback hashes。
- run：input high fence、实际 unit/char/call 使用量、逐阶段状态、cleanup 状态和错误。
- generation：parent seed、exact manifest/index hashes、consumed batch IDs、发布目标和 readback identity。

最小阶段：captured → generated → submitted/unknown → retained → consolidated → locally_merged → gated → published/readback_verified。

发现游标可以在 captured 后前进，但低位未确认 pending 必须保留。retain/daily/merge/publish 前沿分别管理；latest successful event date不参与选择。

### LLM 已生成、POST 失败

立即持久化 schema-valid generated 输出及 hash，恢复时直接从 generated 继续 POST；已知成功的 LLM 不再调用。无法解析、空响应和缺必需字段不生成可提交状态。

### POST 超时或 ACK 丢失

POST 前持久化 caller UUID 与 exact payload hash。相同 operation UUID仅可重试相同 immutable payload；先 GET 精确 operation/child 状态对账。同 UUID不同 payload明确拒绝；409、未知 operation 或 bank mismatch均不得判成功。

逐单元保存远端完成回执，不能等全部 selected records结束后一起记录。failed/cancelled/error 不能写入 successful submit-state。batch parent完成但子操作失败也不能封存。

## 6. 正常执行和费用限制

正常日/周入口共享同一 bank lock和账本。默认禁用历史 replay；显式 legacy repair仅可通过隔离的人工操作流程，不由 cron/full flags隐式恢复。

1. 读状态与新数据；捕获 immutable delta并冻结 H。
2. 在固定小预算内选择未完成/到期单元，不包括 legacy_unknown。常规 pending-unit上限12、input字符上限500000不提高。
3. 生成与提交各单元，立即保存阶段回执；独立单元可继续，失败单元保留 deferred/blocked。
4. 原生 retain/consolidation验证完成后，只把新确认摘要交给本地 canonical merge。
5. 若本轮缺少必要输入，candidate明确 partial/deferred，不发布半成品；新输入不会因部分旧失败被重传。
6. 无新增 candidate时返回 no_op，不为「没有新信息」重新支付整历史 LLM或强求正 recall增益。
7. 有 candidate时按冻结hash执行原门禁；通过后发布并readback。

实际请求次数纳入费用上限，不能继续把「每unit最多2calls」当成真实重试上界。预算报告区分：新增、已有generated待POST、远端pending、deferred、legacy_unknown、已消费，并使用与执行完全相同的选择谓词。

## 7. 重试、错误分类和清理

### 7.1 有界恢复

保留同一 provider，不增加 fallback。永久认证/参数/权限错误立即停止相关阶段并报告；不盲重试400/401。429正确透传并解析秒数或HTTP-date Retry-After。选定5xx/transport按有界退避恢复，通用502不冒充已定位的某后端问题。

每个单元同时受尝试次数、请求timeout、总deadline、run实际calls预算控制。到预算或deadline时持久化 next_retry_at/deferred，交后续轮次，不无限sleep、不退出后遗失上下文。重试策略配置在pipeline_config而非新增行为env变量；不得在本任务中擅自修改生产配置。

连续相同坏JSON/上下文上限不能无限重试；保留冻结输入，明确 invalid_output/payload_blocked。只有未封存单元可按记录边界作有限拆分，须保持source coverage并预检预算。

### 7.2 observations 窗口

每个可能产生 retain/catch-up 的入口都拥有同一锁和窗口，包括 full --skip-daily。仅在 producer即将提交时开启；退出时先确认相关operation及native backlog状态。没有producer且子操作和consolidation均清零，才恢复false并双字段读回。

若失败后仍有真实 backlog，保存cleanup_pending，执行有界受控drain/后续恢复，不盲关导致永远停滞，也不把队列空当作完成。恢复权限、已有56 backlog的处理单独列入生产cutover报告；设计/测试阶段不PATCH生产。

清理失败和核心失败各有独立错误字段，不能在finally中改变量却仍返回旧成功码。所有退出路径必须写run_report。

### 7.3 日志与可选尾部

子进程实时合并stdout/stderr，保留原事件顺序；每阶段起止、预算、attempt、next_retry_at和结果可见，不输出prompt/key。

核心记忆完成状态与wiki/research digest分开；可选任务失败只报告degraded，不能覆盖核心结论或重复触发记忆流程。digest读取有界数据/当前验证结果，不全库扫描按全文长度排序，不引用旧聊天中的FAIL冒充当前代码状态。

## 8. Canonical：sealed state + delta，不再付费全历史 weekly

已有生产memory/canonical作为legacy seed保留；已确认新增摘要仅消费一次。本地typed merge使用topic/type/insight/applicability/validity等语义字段及源证据身份，不能仅用insight文本合并并丢掉条件。保留不同适用域、冲突、supersedes、open_questions以及完整evidence/index；top-N卡片只是投影，不能替代长期详细状态。

历史已发布row如果无法安全重建结构化字段，保留opaque legacy row及hash，不凭字符串猜测改写原意。已有合法但未消费的summary可按证明作为本地delta，不重新处理历史原始数据。

每个candidate使用新的immutable generation目录、精确文件列表/内容hash。publisher只读manifest声明的files，拒绝stale/额外文件混入；不依赖latest目录glob。保留parent generation用于rollback，不删除旧资料。

第一版优先保证旧memory不丢和零历史LLM：可复用既有事务/backup publisher，但完整候选必须包含完整seed，且冻结gate与发布payload一致。优化为delta writes或embedding cache时，必须另有稳定ID及删除范围测试；不允许为了效率跳过embedding或写NULL。

## 9. Gate与成功语义

candidate schema/parse/coverage有效；native相关子操作与consolidation确认完成；阻断conflict为零；generic/local eval成对且满足既有阈值与case-regression约束；embedding数量/顺序/维度1024/nonNULL有效；事务备份、commit和精确readback匹配冻结generation。

gate失败：保留旧production和新candidate，标blocked；不能自动旧史重跑或force publish。publish重试只针对同payloadhash候选，不能重复retain/daily/LLM。已成功published且无delta直接no_op，不重新要求从同一基线获得正收益。

成功完成记忆需区分retained/consolidated（原始增量已可查询）与canonical published；后者失败不否认前者已成功。总体报告必须同时列出阶段状态，不能仅靠exit0或wiki文件宣称端到端成功。

## 10. 迁移、运行安全与回退

1. 暂停生产日/周写入cron需单独用户确认；不动日报统计、Codex本地快照同步、既已暂停的Mental Models。
2. 先只读导出/验证生产canonical seed、progress、checkpoint、来源identity；legacy已成功回执精确核实后sealed。覆盖证明不足的旧输入标legacy_unknown/history_review，不标success，不自动进入pending paid队列。
3. 只用显式迁移清单给未知旧输入建立非完成排除记录；保留来源和旧bank数据。切换时点仅是policy界限，不是完成watermark。
4. source baseline包含当前已观察范围，未来新增采用arrival identity；后续迟到旧事件仍按新arrival处理。损坏/截断/epoch变化显式报source_changed，不自动回放。
5. 代码/状态迁移先shadow和隔离HOME/loopback API试验。生产状态迁移、observations调整及付费增量试跑再次确认实际范围和预算。
6. 备份状态和parent seed，回退代码不删除新旧memory；旧cron不得未经保护重新运行历史模式。
7. 不触碰当前daily_report/tests/package-lock等其他未提交修改；commit只按本任务明确文件分组，push目标需要单独确认。

## 11. 可验证验收

- 同输入跑两次：第二次paid LLM/POST/embedding/publish/trigger全部0。
- 追加旧会话：只提交新范围的新document ID，旧memory保留；快照增长不使旧hash匹配新正文。
- 晚到旧event_date、同timestamp不同事件、run期间新增、来源epoch变化、会话压缩克隆：边界明确，不错漏或重放已sealed内容。
- A/C成功B失败后重启：仅B恢复；生成→提交→远端完成各断点恢复不重复已确认成功阶段。
- async失败不记成功；ACK丢失使用同UUID/payload恢复，实际支持通过隔离服务验证；不同payloadUUID复用拒绝。
- 401/永久400不重试；429两种Retry-After正确；持续502到期限/callcap后延期，状态/预算不丢，无provider替换。
- 超预算分批pending，不清缓存、不扩大历史输入；每轮费用与delta相关，而不是与历史文件量相关。
- 换模型/label、重复generation、旧文件缺失：sealed成功内容零自动LLM；损坏状态报告明确。
- 同insight不同topic/type/applicability不错误合并；seed/evidence完整保留；本地逐批merge与同规则batchmerge一致。
- candidate manifest与目录额外旧文件隔离；old gate/malformed output/queue unknown/embedding错维度或NULL全部拒绝publish。
- daily成功publish失败：恢复只publish；无delta不要求重复正recall增益；wiki/digest失败不改变核心成功状态。
- full-skip、失败清理、并发producer、SIGTERM：bank窗口/锁完整，失败report持久，backlog不被误称drained。
- bundle声明文件与canonical逐字节一致；隔离安装advertised tests与真实CLI流程通过。

测试命令必须使用仓库 `scripts/run_tests.sh`。当前venv均缺pytest，已验证 `HERMES_PYTHON=/home/wyr/miniconda/bin/python` 可通过项目runner运行；测试临时目录应配置到Hermes scratch，不能安装依赖或改生产venv来绕开。

## 12. 当前交付阶段

只读审计完成，用户已批准本规格所述总体方向；源码尚未实施。本规格经用户复核后进入分任务实现/独立复审/隔离验收。所有未实现验收项必须保持未完成，不能以规格文件或单元测试替代真实路径证据。
