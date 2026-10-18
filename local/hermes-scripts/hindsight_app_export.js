// Run in the supported tool orchestrator; no credentials, hidden API or local
// browser storage. Each invocation refetches the head so active turns can finish.
// scope is an explicit thread-ID -> parent metadata map, never all accounts.
async function discoverVisibleAppScope({ previousScope = {}, maxArchivedPages = 100 } = {}) {
  const unpack = result => {
    if (result.isError) throw new Error('supported listing refused; do not substitute an access path');
    return JSON.parse(result.content.find(x => x.type === 'text').text);
  };
  const current = unpack(await tools.mcp__codex_app__list_threads({ limit: 50 }));
  const scope = { ...previousScope };
  const add = row => {
    if (!['codex', 'chatgpt'].includes(row.kind)) return;
    if (scope[row.id]?.kind && scope[row.id].kind !== row.kind) throw new Error('source kind changed');
    scope[row.id] = { parent_session_id: null, relation_basis: 'unknown', ...scope[row.id], kind: row.kind };
  };
  [...(current.pinnedThreads || []), ...(current.threads || [])].forEach(add);
  let cursor = null, archivedComplete = false;
  const seen = new Set();
  for (let n = 0; n < maxArchivedPages; n++) {
    const args = { source: 'codex', limit: 50 };
    if (cursor) args.cursor = cursor;
    const page = unpack(await tools.mcp__codex_app__list_archived_threads(args));
    (page.threads || []).forEach(add);
    if (!page.nextCursor) { archivedComplete = true; break; }
    if (seen.has(page.nextCursor)) throw new Error('archive cursor stalled');
    seen.add(page.nextCursor); cursor = page.nextCursor;
  }
  return { scope, coverage: {
    current_nonpinned_limit: 50, current_pagination_available: false,
    archived_codex_complete: archivedComplete, archived_codex_resume_cursor: archivedComplete ? null : cursor,
    archived_chatgpt: 'refused_in_current_task_context_not_requested_again',
    account_complete: false, source: 'supported_user_visible_listing_only'
  }};
}

async function collectAppHistory({ scope, outputDir, maxPages = 100, knownTurnIdsBySession = {}, onSourceDone = null }) {
  const unpack = (result) => {
    if (result.isError) {
      const failure = new Error('supported thread API failed');
      const detail = (result.content || []).filter(x => x.type === 'text').map(x => x.text).join('\n');
      failure.code = /denied|forbidden|not authorized|permission|only available from local desktop/i.test(detail)
        ? 'access_refused' : 'supported_api_error';
      throw failure;
    }
    return JSON.parse(result.content.find(x => x.type === 'text').text);
  };
  const listing = unpack(await tools.mcp__codex_app__list_threads({ limit: 50 }));
  const visible = new Set([...(listing.pinnedThreads || []), ...(listing.threads || [])]
    .filter(t => ['codex', 'chatgpt'].includes(t.kind) && scope[t.id]).map(t => t.id));
  const report = [];
  const run = Date.now();
  if (!outputDir.startsWith('/') || outputDir.includes('..') || /[\n\r]/.test(outputDir))
    throw new Error('absolute staging output directory required');
  if (!Number.isInteger(maxPages) || maxPages < 1) throw new Error('positive page budget required');
  const quote = value => "'" + value.replace(/'/g, "'\\''") + "'";
  const privateDirectory = "import os,stat,sys; p=sys.argv[1]; os.makedirs(p,mode=0o700,exist_ok=True); s=os.stat(p); assert s.st_uid==os.getuid() and not (stat.S_IMODE(s.st_mode)&0o077), 'private user-owned staging directory required'";
  const completedExec = async args => {
    let result = await tools.exec_command(args), output = result.output || '';
    while (result.exit_code === undefined && result.session_id) {
      result = await tools.write_stdin({ session_id: result.session_id, chars: '', yield_time_ms: 1000, max_output_tokens: 1000 });
      output += result.output || '';
    }
    return { ...result, output };
  };
  const prepared = await completedExec({cmd: `rtk proxy /usr/bin/python3 -c ${quote(privateDirectory)} ${quote(outputDir)}`});
  if (prepared.exit_code !== 0) throw new Error('private staging directory check failed');
  const writePrivate = async (path, value) => {
    const body = JSON.stringify(value, null, 2);
    const saved = await tools.apply_patch(`*** Begin Patch\n*** Add File: ${path}\n${body.split('\n').map(l => '+' + l).join('\n')}\n*** End Patch`);
    if (saved?.isError) throw new Error('staging write failed');
    const secureAndHash = "import os,sys,hashlib; p=sys.argv[1]; os.chmod(p,0o600); print(hashlib.sha256(open(p,'rb').read()).hexdigest())";
    const secured = await completedExec({cmd: `rtk proxy /usr/bin/python3 -c ${quote(secureAndHash)} ${quote(path)}`});
    if (secured.exit_code !== 0) throw new Error('private staging write failed');
    const sha256 = secured.output.trim();
    if (!/^[0-9a-f]{64}$/.test(sha256)) throw new Error('staging file hash missing');
    return sha256;
  };
  for (const id of Object.keys(scope)) {
    if (!/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(id))
      throw new Error('invalid scoped thread ID');
    const anchors = new Set(knownTurnIdsBySession[id] || []);
    let cursor = null, done = false, count = 0, checkpoint = false, failure = null;
    let truncated = 0, deferred = 0, olderCursor = null;
    const paths = [], pageReceipts = [];
    const seen = new Set();
    try {
    for (let n = 0; n < maxPages; n++) {
      const args = { threadId: id, turnLimit: 10, includeOutputs: false, maxOutputCharsPerItem: 20000 };
      if (cursor) args.cursor = cursor;
      const page = unpack(await tools.mcp__codex_app__read_thread(args));
      if (page.schemaVersion !== 1 || page.thread.id !== id ||
          !['codex', 'chatgpt'].includes(page.thread.kind) ||
          typeof page.page?.hasMore !== 'boolean' ||
          (scope[id].kind && scope[id].kind !== page.thread.kind))
        throw new Error('unexpected thread/history schema');
      const safe = {
        schema: 'codex-app-read-thread/1', schemaVersion: page.schemaVersion,
        thread: Object.fromEntries(['id', 'kind', 'hostId', 'createdAt', 'title']
          .filter(k => page.thread[k] !== undefined).map(k => [k, page.thread[k]])),
        requested_cursor: cursor, page: page.page,
        turns: page.turns.map(t => ({
          ...Object.fromEntries(['id', 'status', 'startedAt', 'completedAt'].map(k => [k, t[k]])),
          items: t.items.filter(i => ['userMessage', 'agentMessage'].includes(i.type)).map(i => {
            const row = Object.fromEntries(['id', 'type', 'text', 'phase', 'truncated']
              .filter(k => i[k] !== undefined).map(k => [k, i[k]]));
            if (Array.isArray(i.content)) row.content = i.content
              .filter(p => p && ['text', 'inputText', 'input_text'].includes(p.type))
              .map(p => ({ type: p.type, text: p.text || '' }));
            return row;
          })
        }))
      };
      safe.collected_at = new Date().toISOString();
      const path = `${outputDir}/page-${id}-${run}-${String(n).padStart(3, '0')}.json`;
      const sha256 = await writePrivate(path, safe);
      paths.push(path);
      pageReceipts.push({ path, sha256, requested_cursor: cursor, next_cursor: page.page.nextCursor || null,
        has_more: page.page.hasMore, collected_at: safe.collected_at });
      count += safe.turns.length;
      truncated += safe.turns.reduce((a, t) => a + t.items.filter(i => i.truncated).length, 0);
      deferred += safe.turns.filter(t => t.status !== 'completed').length;
      if (!page.page.hasMore) { done = true; break; }
      if (!page.page.nextCursor || seen.has(page.page.nextCursor)) throw new Error('pagination stalled');
      seen.add(page.page.nextCursor); cursor = page.page.nextCursor;
      // Anchors come from this same session's previously sealed completed
      // turns. Save the entire page before stopping, so a new-message gap and
      // a newly completed head turn are both preserved. Always start at head.
      if (safe.turns.some(t => t.status === 'completed' && anchors.has(t.id))) {
        checkpoint = true; olderCursor = cursor; break;
      }
    }
    } catch (error) {
      failure = error.code || 'collection_failed';
    }
    const row = { thread: id, listed_in_current_bounded_view: visible.has(id),
      turns: count, pages: paths.length, page_paths: paths, page_receipts: pageReceipts,
      collected_at: new Date().toISOString(), reached_oldest: done,
      incremental_checkpoint_reached: checkpoint,
      incremental_interval_complete: !failure && (done || checkpoint),
      resume_cursor: done || checkpoint ? null : cursor, older_history_cursor: olderCursor,
      truncated_items: truncated, deferred_turns: deferred, error_class: failure,
      history_view: 'read_thread_summary', llm_events: 'unknown' };
    row.receipt_path = `${outputDir}/collection-receipt-${id}-${run}.json`;
    await writePrivate(row.receipt_path, row);
    report.push(row);
    if (onSourceDone) await onSourceDone(row);
  }
  return report;
}
