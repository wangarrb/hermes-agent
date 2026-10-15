// Run in the supported tool orchestrator; no credentials, hidden API or local
// browser storage. Each invocation refetches the head so active turns can finish.
// scope is an explicit thread-ID -> parent metadata map, never all accounts.
async function collectAppHistory({ scope, outputDir, maxPages = 100 }) {
  const unpack = (result) => {
    if (result.isError) throw new Error('supported thread API failed');
    return JSON.parse(result.content.find(x => x.type === 'text').text);
  };
  const listing = unpack(await tools.mcp__codex_app__list_threads({ limit: 50 }));
  const visible = new Set([...listing.pinnedThreads, ...listing.threads]
    .filter(t => t.kind === 'codex' && scope[t.id]).map(t => t.id));
  const report = [];
  const run = Date.now();
  if (!outputDir.startsWith('/') || outputDir.includes('..') || /[\n\r]/.test(outputDir))
    throw new Error('absolute staging output directory required');
  for (const id of Object.keys(scope)) {
    if (!/^[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}$/.test(id))
      throw new Error('invalid scoped thread ID');
    let cursor = null, done = false, count = 0;
    const seen = new Set();
    for (let n = 0; n < maxPages; n++) {
      const args = { threadId: id, turnLimit: 10, includeOutputs: false, maxOutputCharsPerItem: 10000 };
      if (cursor) args.cursor = cursor;
      const page = unpack(await tools.mcp__codex_app__read_thread(args));
      if (page.schemaVersion !== 1 || page.thread.id !== id || page.thread.kind !== 'codex')
        throw new Error('unexpected thread/history schema');
      const safe = {
        schema: 'codex-app-read-thread/1', schemaVersion: page.schemaVersion,
        thread: Object.fromEntries(['id', 'kind', 'hostId', 'createdAt', 'title']
          .filter(k => page.thread[k] !== undefined).map(k => [k, page.thread[k]])),
        requested_cursor: cursor, page: page.page,
        turns: page.turns.map(t => ({
          ...Object.fromEntries(['id', 'status', 'startedAt', 'completedAt'].map(k => [k, t[k]])),
          items: t.items.filter(i => ['userMessage', 'agentMessage'].includes(i.type)).map(i =>
            Object.fromEntries(['id', 'type', 'text', 'content', 'phase', 'truncated']
              .filter(k => i[k] !== undefined).map(k => [k, i[k]])))
        }))
      };
      const body = JSON.stringify(safe, null, 2);
      const path = `${outputDir}/page-${id}-${run}-${String(n).padStart(3, '0')}.json`;
      const saved = await tools.apply_patch(`*** Begin Patch\n*** Add File: ${path}\n${body.split('\n').map(l => '+' + l).join('\n')}\n*** End Patch`);
      if (saved?.isError) throw new Error('staging page write failed');
      count += safe.turns.length;
      if (!page.page.hasMore) { done = true; break; }
      if (!page.page.nextCursor || seen.has(page.page.nextCursor)) throw new Error('pagination stalled');
      seen.add(page.page.nextCursor); cursor = page.page.nextCursor;
    }
    report.push({ thread: id, listed_in_current_bounded_view: visible.has(id),
      turns: count, reached_oldest: done, resume_cursor: done ? null : cursor,
      history_view: 'read_thread_summary', llm_events: 'unknown' });
  }
  return report;
}
