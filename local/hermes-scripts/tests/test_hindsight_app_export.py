"""Exercise the real JS producer with supported-tool fixtures and private files.

Node executes only local staging commands in pytest's temporary directory. All
listing/history tools are mocks; no account API, model, or production path is used.
"""
import hashlib
import json
from pathlib import Path
import shutil
import stat
import subprocess

import pytest


PRODUCER = Path(__file__).resolve().parents[1] / "hindsight_app_export.js"
FIRST = "11111111-1111-1111-1111-111111111111"
SECOND = "22222222-2222-2222-2222-222222222222"

NODE_HARNESS = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const path = require('node:path');
const cp = require('node:child_process');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const calls = [], events = [], pending = new Map();
let yielded = false, invalidHashReturned = false, nextSession = 70;
const packed = value => ({content: [{type: 'text', text: JSON.stringify(value)}]});
const checkPath = p => {
  const resolved = path.resolve(p);
  if (!resolved.startsWith(path.resolve(input.outputDir) + path.sep))
    throw new Error('fixture write outside private staging');
  return resolved;
};
const tools = {
  async mcp__codex_app__list_threads(args) {
    calls.push({tool: 'list', args});
    return packed({pinnedThreads: [], threads: Object.keys(input.scope).map(id =>
      ({id, kind: input.scope[id].kind || 'codex'}))});
  },
  async mcp__codex_app__read_thread(args) {
    calls.push({tool: 'read', args});
    const response = input.responses[args.threadId]?.[args.cursor || 'HEAD'];
    if (!response) throw new Error('unexpected fixture history read');
    return response.isError ? response : packed(response);
  },
  async apply_patch(patch) {
    const lines = patch.split('\n');
    if (lines[0] !== '*** Begin Patch' || !lines[1].startsWith('*** Add File: ')
        || lines.at(-1) !== '*** End Patch') throw new Error('unexpected fixture patch');
    const destination = checkPath(lines[1].slice('*** Add File: '.length));
    const bodyLines = lines.slice(2, -1);
    if (bodyLines.some(line => !line.startsWith('+'))) throw new Error('invalid fixture addition');
    fs.writeFileSync(destination, bodyLines.map(line => line.slice(1)).join('\n') + '\n',
      {flag: 'wx', mode: 0o644});
    events.push({operation: 'write', path: destination});
    return {isError: false};
  },
  async exec_command(args) {
    if (!args.cmd.startsWith('rtk proxy /usr/bin/python3 -c ')
        || !args.cmd.includes(input.outputDir)) throw new Error('unexpected fixture command');
    const output = cp.execSync(args.cmd, {encoding: 'utf8'});
    if (input.invalidHash && !invalidHashReturned && args.cmd.includes('hashlib')) {
      invalidHashReturned = true;
      return {exit_code: 0, output: 'not-a-sha256\n'};
    }
    if (input.yieldHash && !yielded && args.cmd.includes('hashlib')) {
      yielded = true;
      const id = nextSession++;
      pending.set(id, [output.slice(17, 39), output.slice(39)]);
      events.push({operation: 'exec_yield', session_id: id});
      return {session_id: id, output: output.slice(0, 17)};
    }
    return {exit_code: 0, output};
  },
  async write_stdin(args) {
    calls.push({tool: 'wait_exec', args});
    const chunks = pending.get(args.session_id);
    if (!chunks) throw new Error('unexpected fixture exec wait');
    const output = chunks.shift();
    if (chunks.length) return {session_id: args.session_id, output};
    pending.delete(args.session_id);
    events.push({operation: 'exec_complete', session_id: args.session_id});
    return {exit_code: 0, output};
  }
};
const context = vm.createContext({tools});
vm.runInContext(fs.readFileSync(input.producer, 'utf8') +
  '\nglobalThis.fixtureCollect = collectAppHistory;', context);
(async () => {
  try {
    const report = await context.fixtureCollect({scope: input.scope,
      outputDir: input.outputDir, maxPages: input.maxPages,
      knownTurnIdsBySession: input.knownTurnIdsBySession || {}});
    process.stdout.write(JSON.stringify({report, calls, events, pendingExec: pending.size}));
  } catch (error) {
    process.stdout.write(JSON.stringify({fatal: error.message, calls, events}));
  }
})();
"""


def turn(identity, *, status="completed", items=None):
    return {
        "id": identity, "status": status, "startedAt": 1792022400,
        "completedAt": 1792022460 if status == "completed" else None,
        "items": items if items is not None else [
            {"id": identity + "-u", "type": "userMessage", "content": [
                {"type": "text", "text": "Public request for " + identity}]},
            {"id": identity + "-a", "type": "agentMessage", "text": "Public reply for " + identity},
        ],
    }


def page(session, turns, *, next_cursor=None):
    return {
        "schemaVersion": 1,
        "thread": {"id": session, "kind": "codex", "createdAt": 1792022300},
        "page": {"hasMore": next_cursor is not None, "nextCursor": next_cursor},
        "turns": turns,
    }


def collect(tmp_path, responses, *, known=None, max_pages=10, **options):
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is required for the supported App JS producer fixture")
    output_dir = tmp_path / "private-export"
    request = {
        "producer": str(PRODUCER), "outputDir": str(output_dir),
        "scope": {sid: {"kind": "codex"} for sid in responses},
        "responses": responses, "knownTurnIdsBySession": known or {},
        "maxPages": max_pages, **options,
    }
    result = subprocess.run(
        [node, "-e", NODE_HARNESS], input=json.dumps(request),
        capture_output=True, text=True, check=True, timeout=60,
    )
    value = json.loads(result.stdout)
    assert "fatal" not in value, value
    return value, output_dir


def assert_receipt(row, output_dir):
    receipt = Path(row["receipt_path"])
    assert receipt.parent == output_dir
    assert stat.S_IMODE(output_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(receipt.stat().st_mode) == 0o600
    assert json.loads(receipt.read_text()) == row
    assert row["collected_at"].endswith("Z")
    assert row["pages"] == len(row["page_paths"]) == len(row["page_receipts"])
    for path, entry in zip(row["page_paths"], row["page_receipts"]):
        path = Path(path)
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
        assert entry["path"] == str(path)
        assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text())
        assert entry["requested_cursor"] == data["requested_cursor"]
        assert entry["next_cursor"] == data["page"]["nextCursor"]
        assert entry["has_more"] == data["page"]["hasMore"]
        assert entry["collected_at"] == data["collected_at"]


def test_incremental_checkpoint_preserves_new_gap_across_two_pages(tmp_path):
    result, directory = collect(tmp_path, {FIRST: {
        "HEAD": page(FIRST, [turn("newest")], next_cursor="second"),
        "second": page(FIRST, [turn("new-gap"), turn("sealed-anchor")], next_cursor="older"),
    }}, known={FIRST: ["sealed-anchor"]})
    row = result["report"][0]
    reads = [c["args"] for c in result["calls"] if c["tool"] == "read"]
    assert [r.get("cursor") for r in reads] == [None, "second"]
    assert all(r["turnLimit"] == 10 and r["includeOutputs"] is False
               and r["maxOutputCharsPerItem"] == 20000 for r in reads)
    assert [t["id"] for p in row["page_paths"] for t in json.loads(Path(p).read_text())["turns"]] == [
        "newest", "new-gap", "sealed-anchor"]
    assert row["incremental_checkpoint_reached"] is True
    assert row["incremental_interval_complete"] is True
    assert row["reached_oldest"] is False
    assert row["resume_cursor"] is None
    assert row["older_history_cursor"] == "older"
    assert_receipt(row, directory)


def test_new_session_reads_to_eof(tmp_path):
    result, directory = collect(tmp_path, {FIRST: {
        "HEAD": page(FIRST, [turn("head")], next_cursor="old"),
        "old": page(FIRST, [turn("oldest")]),
    }})
    row = result["report"][0]
    assert row["pages"] == 2
    assert row["reached_oldest"] is True
    assert row["incremental_checkpoint_reached"] is False
    assert row["incremental_interval_complete"] is True
    assert row["resume_cursor"] is row["older_history_cursor"] is None
    assert_receipt(row, directory)


@pytest.mark.parametrize('pagination', [
    {}, {'hasMore': None}, {'hasMore': 'false'}, {'hasMore': 'true'},
    {'hasMore': ''}, {'hasMore': 0}, {'hasMore': 1},
])
def test_nonboolean_history_pagination_is_refused_before_eof(tmp_path, pagination):
    invalid = page(FIRST, [turn('untrusted-page')])
    invalid['page'] = pagination
    result, directory = collect(tmp_path, {FIRST: {'HEAD': invalid}})
    row = result['report'][0]
    assert row['error_class'] == 'collection_failed'
    assert row['pages'] == row['turns'] == 0
    assert row['reached_oldest'] is False
    assert row['incremental_checkpoint_reached'] is False
    assert row['incremental_interval_complete'] is False
    assert row['page_paths'] == row['page_receipts'] == []
    assert_receipt(row, directory)


def test_page_budget_keeps_pending_cursor(tmp_path):
    result, directory = collect(tmp_path, {FIRST: {
        "HEAD": page(FIRST, [turn("unsealed")], next_cursor="pending-old"),
    }}, max_pages=1, known={FIRST: ["not-yet-reached"]})
    row = result["report"][0]
    assert row["pages"] == 1
    assert row["resume_cursor"] == "pending-old"
    assert row["older_history_cursor"] is None
    assert row["reached_oldest"] is False
    assert row["incremental_checkpoint_reached"] is False
    assert row["incremental_interval_complete"] is False
    assert row["error_class"] is None
    assert_receipt(row, directory)


def test_in_progress_known_turn_does_not_stop_pagination(tmp_path):
    result, directory = collect(tmp_path, {FIRST: {
        "HEAD": page(FIRST, [turn("known-but-running", status="inProgress")], next_cursor="next"),
        "next": page(FIRST, [turn("new-gap"), turn("completed-anchor")], next_cursor="older"),
    }}, known={FIRST: ["known-but-running", "completed-anchor"]})
    row = result["report"][0]
    assert row["pages"] == 2
    assert row["deferred_turns"] == 1
    assert row["incremental_checkpoint_reached"] is True
    assert_receipt(row, directory)


def test_public_projection_drops_tools_reasoning_and_attachment_urls(tmp_path):
    items = [
        {"id": "u", "type": "userMessage", "content": [
            {"type": "text", "text": "Visible request", "privateExtra": "TEXT_METADATA_SECRET"},
            {"type": "image_url", "image_url": {"url": "fixture://ATTACHMENT_SECRET"}},
            {"type": "file", "url": "fixture://FILE_SECRET"},
        ]},
        {"id": "a", "type": "agentMessage", "text": "Visible reply", "truncated": True,
         "tool_calls": [{"arguments": "AGENT_TOOL_SECRET"}]},
        {"id": "r", "type": "reasoning", "summary": "REASONING_SECRET"},
        {"id": "c", "type": "commandExecution", "command": "COMMAND_SECRET"},
        {"id": "o", "type": "functionCallOutput", "output": "OUTPUT_SECRET"},
    ]
    result, directory = collect(tmp_path, {FIRST: {"HEAD": page(FIRST, [turn("public", items=items)])}})
    row = result["report"][0]
    stored = Path(row["page_paths"][0]).read_text()
    assert "Visible request" in stored and "Visible reply" in stored
    assert "SECRET" not in stored
    projected = json.loads(stored)["turns"][0]["items"]
    assert [i["type"] for i in projected] == ["userMessage", "agentMessage"]
    assert row["truncated_items"] == 1
    assert_receipt(row, directory)


@pytest.mark.parametrize("detail,expected", [
    ("permission denied by supported API", "access_refused"),
    ("temporary service failure", "supported_api_error"),
])
def test_api_failure_writes_source_receipt_and_continues(tmp_path, detail, expected):
    result, directory = collect(tmp_path, {
        FIRST: {"HEAD": {"isError": True, "content": [{"type": "text", "text": detail}]}},
        SECOND: {"HEAD": page(SECOND, [turn("next-source")])},
    })
    failed, successful = result["report"]
    assert failed["error_class"] == expected and failed["pages"] == 0
    assert failed["incremental_interval_complete"] is False
    assert successful["thread"] == SECOND and successful["incremental_interval_complete"] is True
    for row in result["report"]:
        assert_receipt(row, directory)
    writes = [e["path"] for e in result["events"] if e["operation"] == "write"]
    assert writes.index(failed["receipt_path"]) < writes.index(successful["page_paths"][0])


def test_yielded_exec_finishes_before_page_hash_is_accepted(tmp_path):
    result, directory = collect(tmp_path, {FIRST: {"HEAD": page(FIRST, [turn("yield")])}}, yieldHash=True)
    assert len([c for c in result["calls"] if c["tool"] == "wait_exec"]) == 2
    assert result["pendingExec"] == 0
    assert result["report"][0]["error_class"] is None
    assert_receipt(result["report"][0], directory)
    operations = [e["operation"] for e in result["events"]]
    assert operations.index("exec_yield") < operations.index("exec_complete")


def test_later_source_failure_preserves_completed_source_receipt(tmp_path):
    result, directory = collect(tmp_path, {
        FIRST: {"HEAD": page(FIRST, [turn("already-saved")])},
        SECOND: {"HEAD": {"isError": True, "content": [
            {"type": "text", "text": "permission denied"}]}},
    })
    completed, failed = result["report"]
    assert completed["incremental_interval_complete"] is True
    assert failed["error_class"] == "access_refused"
    assert_receipt(completed, directory)
    assert_receipt(failed, directory)
    writes = [e["path"] for e in result["events"] if e["operation"] == "write"]
    assert writes.index(completed["receipt_path"]) < writes.index(failed["receipt_path"])


def test_missing_sha_is_not_recorded_as_a_successful_page(tmp_path):
    result, directory = collect(tmp_path, {FIRST: {"HEAD": page(FIRST, [turn("bad-hash")])}}, invalidHash=True)
    row = result["report"][0]
    assert row["pages"] == 0 and row["page_receipts"] == []
    assert row["error_class"] == "collection_failed"
    assert row["incremental_interval_complete"] is False
    assert_receipt(row, directory)
