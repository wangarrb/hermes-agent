import pytest

from hermes_state import SessionDB


@pytest.fixture
def db(tmp_path):
    return SessionDB(tmp_path / "state.db")


def _set_session(db: SessionDB, session_id: str, *, started_at: float, ended_at=None, end_reason=None):
    db._conn.execute(
        "UPDATE sessions SET started_at = ?, ended_at = ?, end_reason = ? WHERE id = ?",
        (started_at, ended_at, end_reason, session_id),
    )


def _set_message_time(db: SessionDB, session_id: str, timestamp: float):
    db._conn.execute(
        "UPDATE messages SET timestamp = ? WHERE session_id = ?",
        (timestamp, session_id),
    )


def test_list_sessions_rich_orders_by_last_active_not_started_at(db):
    db.create_session("newer_start", source="cli")
    db.create_session("older_but_active", source="cli")
    db.append_message("newer_start", role="user", content="started later")
    db.append_message("older_but_active", role="user", content="active later")

    _set_session(db, "newer_start", started_at=200)
    _set_session(db, "older_but_active", started_at=100)
    _set_message_time(db, "newer_start", 210)
    _set_message_time(db, "older_but_active", 300)

    rows = db.list_sessions_rich(
        source="cli",
        limit=2,
        project_compression_tips=False,
    )

    assert [row["id"] for row in rows] == ["older_but_active", "newer_start"]


def test_list_sessions_rich_projects_compression_tip_before_limiting(db):
    # Reproducer for /resume missing recent sessions: a long compression chain
    # can have an old root but a very recent continuation tip.  If LIMIT is
    # applied to roots ordered by started_at before projection, the recent tip
    # disappears behind newer short-lived roots.
    db.create_session("long_root", source="cli")
    _set_session(db, "long_root", started_at=100, ended_at=150, end_reason="compression")
    db.append_message("long_root", role="user", content="old root")
    _set_message_time(db, "long_root", 120)

    db.create_session("long_tip", source="cli", parent_session_id="long_root")
    _set_session(db, "long_tip", started_at=1000)

    for i in range(10):
        sid = f"short_{i}"
        db.create_session(sid, source="cli")
        db.append_message(sid, role="user", content=f"short {i}")
        _set_session(db, sid, started_at=200 + i)
        _set_message_time(db, sid, 200 + i)

    rows = db.list_sessions_rich(source="cli", limit=5)

    assert rows[0]["id"] == "long_tip"
    assert rows[0]["_lineage_root_id"] == "long_root"
    assert "long_tip" in [row["id"] for row in rows]


def test_list_sessions_rich_recovers_legacy_broken_compression_boundary(db):
    # Legacy rows can have a compression child whose parent was later marked
    # resumed_other/new_session/cli_close instead of compression.  That broken
    # boundary used to stop projection and hide the latest continuation.
    db.create_session("root", source="cli")
    _set_session(db, "root", started_at=100, ended_at=150, end_reason="compression")

    db.create_session("stale_parent", source="cli", parent_session_id="root")
    _set_session(db, "stale_parent", started_at=200, ended_at=400, end_reason="resumed_other")
    db.append_message("stale_parent", role="user", content="stale parent")
    _set_message_time(db, "stale_parent", 220)

    db.create_session("recovered_child", source="cli", parent_session_id="stale_parent")
    _set_session(db, "recovered_child", started_at=300, ended_at=350, end_reason="compression")

    db.create_session("tip", source="cli", parent_session_id="recovered_child")
    _set_session(db, "tip", started_at=1000)

    rows = db.list_sessions_rich(source="cli", limit=1)

    assert rows[0]["id"] == "tip"
    assert rows[0]["_lineage_root_id"] == "root"
