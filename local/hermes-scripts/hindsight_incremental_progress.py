"""Private, stdlib-only runtime timings; no content, identifiers or credentials.

One run owns one heartbeat. Nested spans / publisher callbacks share that owner,
not the legacy latest/history observer. Diagnostics never authorize publication.
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps
import json
from pathlib import Path
import queue
import sys
import threading
import time

OPERATIONS = frozenset({
    'pipeline', 'capture', 'preflight', 'window', 'delivery', 'facts', 'scoped_drain',
    'canonical', 'native_stage', 'merge', 'gate', 'publish', 'optional_tail',
    'seed_state', 'saved_manifest_validate', 'validate_generation', 'candidate',
    'initial_seed', 'seed_read', 'candidate_load', 'parent_load', 'rows_fetch',
    'rows_hash', 'read_remote', 'remote_documents_fetch', 'remote_units_fetch',
    'remote_identity', 'remote_documents_hash', 'remote_units_hash',
    'remote_combined_hash', 'seed_preserved', 'validate_gate', 'readback',
    'readback_generation', 'publish_generation', 'readonly_connect',
    'readonly_transaction', 'prewrite_check', 'precommit_check',
    'publisher_publish', 'embeddings', 'remote_state_hash', 'cleanup',
    'source_parent_merge', 'source_parent_gate', 'source_parent_publish',
    'delta_load',
})
_ACTIVE: ContextVar[ProgressReporter | None] = ContextVar('hindsight_incremental_progress', default=None)
_UNBOUND = object()


def active_reporter():
    return _ACTIVE.get()


@contextmanager
def reporter_context(reporter):
    token = _ACTIVE.set(reporter)
    try:
        yield
    finally:
        _ACTIVE.reset(token)


def _counts(*, row_count=None, document_count=None, unit_count=None, byte_count=None,
            requested_count=None, received_count=None, calls=None):
    values = dict(row_count=row_count, document_count=document_count, unit_count=unit_count,
                  byte_count=byte_count, requested_count=requested_count, received_count=received_count, calls=calls)
    if any(type(value) is not int or value < 0 for value in values.values() if value is not None):
        raise ValueError('progress counts require nonnegative integers')
    return {key: value for key, value in values.items() if value is not None}


def _operation(operation):
    if operation not in OPERATIONS:
        raise ValueError('unknown progress operation')
    return operation


def _process_metrics():
    values = {'process_cpu_seconds': round(time.process_time(), 3)}
    try:
        # A small kernel-owned file, never SQL, argv, env, paths or provider I/O.
        for line in Path('/proc/self/status').read_text().splitlines():
            field, _, value = line.partition(':')
            if field in {'VmRSS', 'VmSwap'}:
                values['process_rss_kib' if field == 'VmRSS' else 'process_swap_kib'] = int(value.split()[0])
    except (OSError, ValueError):
        return values
    return values


class Counts:
    def __init__(self, reporter=None, frame=None):
        self.reporter, self.frame = reporter, frame

    def update(self, *, row_count=None, document_count=None, unit_count=None, byte_count=None,
               requested_count=None, received_count=None, calls=None):
        values = _counts(row_count=row_count, document_count=document_count, unit_count=unit_count,
                         byte_count=byte_count, requested_count=requested_count, received_count=received_count, calls=calls)
        if self.reporter is not None and self.frame is not None:
            with self.reporter.lock:
                self.frame['counts'].update(values)


class ProgressReporter:
    def __init__(self, owner, *, interval=60.0, clock=time.monotonic, sink=None, queue_size=256):
        if type(queue_size) is not int or queue_size < 1:
            raise ValueError('diagnostic queue must have a finite positive capacity')
        self.owner, self.interval, self.clock = owner, interval, clock
        self.sink = sys.stderr if sink is None else sink
        self.lock = threading.RLock()
        self.pending = queue.Queue(maxsize=queue_size)
        self.stopped = threading.Event()
        self.disabled = False
        self.error_type = None
        self.started = clock()
        self.stack = [{'operation': 'pipeline', 'started': self.started, 'counts': {}}]
        self.thread = threading.Thread(target=self._heartbeat, name='hindsight-publication-heartbeat', daemon=True)

    def emit(self, event, frame, *, error_type=None, metrics=False):
        with self.lock:
            now = self.clock()
            record = {'event': event, 'operation': frame['operation'],
                      'elapsed_seconds': round(max(0.0, now - frame['started']), 3),
                      'run_elapsed_seconds': round(max(0.0, now - self.started), 3), **frame['counts']}
        if error_type is not None:
            record['error_type'] = error_type
        if metrics:
            record.update(_process_metrics())
        if self.disabled or self.stopped.is_set():
            return
        try:
            self.pending.put_nowait(json.dumps(record, allow_nan=False, sort_keys=True) + '\n')
        except queue.Full:
            # Diagnostics may be dropped; backpressure must never delay business.
            self.disabled = True

    def fail(self, error):
        self.error_type = type(error).__name__

    def _heartbeat(self):
        # This is the only thread allowed to touch the sink. It consumes the
        # bounded queue and schedules heartbeats; callers never wait for I/O.
        next_beat = time.monotonic() + self.interval
        while not (self.stopped.is_set() and self.pending.empty()):
            try:
                item = self.pending.get(timeout=max(0.0, next_beat - time.monotonic()))
            except queue.Empty:
                item = None
            else:
                try:
                    if item is not None and not self.disabled:
                        try:
                            self.sink.write(item)
                            self.sink.flush()
                        except (OSError, ValueError):
                            self.disabled = True
                finally:
                    self.pending.task_done()
            if not self.stopped.is_set() and time.monotonic() >= next_beat:
                if not self.disabled:
                    with self.lock:
                        frame = self.stack[-1]
                    self.emit('heartbeat', frame, metrics=True)
                next_beat = time.monotonic() + self.interval

    def start(self):
        self.emit('start', self.stack[0])
        self.thread.start()

    def close(self):
        self.emit('error' if self.error_type else 'done', self.stack[0], error_type=self.error_type)
        self.stopped.set()
        try:
            self.pending.put_nowait(None)  # Wake an idle consumer even when disabled.
        except queue.Full:
            # A full queue already wakes the consumer; it will drain or discard it.
            self.disabled = True
        self.thread.join(timeout=1.0)
        # A pathological sink may still hold the daemon; business cleanup is bounded.


@contextmanager
def reporting_run(owner, *, interval=60.0, clock=time.monotonic, sink=None, queue_size=256):
    previous = active_reporter()
    if previous is not None and previous.owner is owner:
        yield previous
        return
    reporter = ProgressReporter(owner, interval=interval, clock=clock, sink=sink, queue_size=queue_size)
    token = _ACTIVE.set(reporter)
    try:
        reporter.start()
        yield reporter
    except BaseException as error:
        reporter.fail(error)
        raise
    finally:
        try:
            if reporter.thread.ident is not None:
                reporter.close()
        finally:
            _ACTIVE.reset(token)


@contextmanager
def span(operation, *, row_count=None, document_count=None, unit_count=None, byte_count=None,
         requested_count=None, received_count=None, calls=None):
    operation = _operation(operation)
    values = _counts(row_count=row_count, document_count=document_count, unit_count=unit_count,
                     byte_count=byte_count, requested_count=requested_count, received_count=received_count, calls=calls)
    reporter = active_reporter()
    if reporter is None:
        yield Counts()
        return
    frame = {'operation': operation, 'started': reporter.clock(), 'counts': values}
    with reporter.lock:
        reporter.stack.append(frame)
    try:
        reporter.emit('start', frame)
        yield Counts(reporter, frame)
    except BaseException as error:
        reporter.emit('error', frame, error_type=type(error).__name__)
        raise
    else:
        reporter.emit('done', frame)
    finally:
        with reporter.lock:
            index = next(i for i, active in enumerate(reporter.stack) if active is frame)
            del reporter.stack[index]


def timed(operation, *, reporter=_UNBOUND):
    """Bind only the reporter for callbacks; never inspect arguments or results."""
    _operation(operation)
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            token = _ACTIVE.set(reporter) if reporter is not _UNBOUND else None
            try:
                with span(operation):
                    return function(*args, **kwargs)
            finally:
                if token is not None:
                    _ACTIVE.reset(token)
        return wrapped
    return decorate


def record_stage(stage, values):
    """Project the existing pipeline ledger event onto a closed numeric schema."""
    reporter = active_reporter()
    if reporter is None or stage not in OPERATIONS:
        return
    # Counts come from actual captured batches / caller counters, never IDs or
    # free-form status/error/receipt fields from the private run report.
    counts = _counts(unit_count=values.get('new_batches'), calls=values.get('calls'))
    reporter.emit('done', {'operation': stage, 'started': reporter.started, 'counts': counts})
