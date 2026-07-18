import hashlib
import json
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture(autouse=True)
def trusted_mission_control(monkeypatch):
    monkeypatch.setattr(kb, "is_mission_control_command_cgroup", lambda: True)


def _policy(**overrides):
    values = {
        "allowed_assignees": ("planner", "worker"),
        "orchestrator_assignees": ("planner",),
        "max_depth": 2,
        "max_tasks": 8,
        "max_runtime_seconds": 60,
        "max_concurrency": 2,
        "max_wall_clock_seconds": 300,
        "goal_max_turns": 5,
    }
    values.update(overrides)
    return kb.OrchestrationPolicy(**values)


def _root(conn, *, now=1_700_000_000):
    return kb.create_task(
        conn,
        title="program",
        assignee="planner",
        created_by="mission-control",
        idempotency_key=f"root-{now}",
        orchestration_policy=_policy(),
    )


def test_extend_program_deadline_is_monotonic_audited_and_idempotent(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        result = kb.extend_program_deadline(
            conn,
            root_id,
            new_deadline=now + 600,
            actor="mission-control",
            idempotency_key="extend-1",
        )
        assert result == {
            "root_id": root_id,
            "previous_deadline": now + 300,
            "effective_deadline": now + 600,
            "actor": "mission-control",
            "idempotency_key": "extend-1",
            "replayed": False,
        }

        replay = kb.extend_program_deadline(
            conn,
            root_id,
            new_deadline=now + 600,
            actor="mission-control",
            idempotency_key="extend-1",
        )
        assert replay == {**result, "replayed": True}
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == 1
        )

        with pytest.raises(ValueError, match="idempotency key request does not match"):
            kb.extend_program_deadline(
                conn,
                root_id,
                new_deadline=now + 700,
                actor="mission-control",
                idempotency_key="extend-1",
            )
        with pytest.raises(ValueError, match="must increase"):
            kb.extend_program_deadline(
                conn,
                root_id,
                new_deadline=now + 500,
                actor="mission-control",
                idempotency_key="extend-2",
            )
    finally:
        conn.close()


def test_extended_deadline_allows_expired_program_to_be_unblocked(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        monkeypatch.setattr(kb.time, "time", lambda: now + 300)
        assert kb.claim_task(conn, root_id) is None
        blocked = kb.get_task(conn, root_id)
        assert blocked is not None and blocked.status == "blocked"

        kb.extend_program_deadline(
            conn,
            root_id,
            new_deadline=now + 600,
            actor="mission-control",
            idempotency_key="extend-after-expiry",
        )
        assert kb.unblock_task(conn, root_id)
        assert kb.claim_task(conn, root_id) is not None
    finally:
        conn.close()


def test_archive_terminal_program_is_audited_and_idempotent(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        claimed = kb.claim_task(conn, root_id)
        assert claimed is not None
        assert kb.complete_task(
            conn, root_id, result="complete", expected_run_id=claimed.current_run_id
        )

        result = kb.archive_program(
            conn,
            root_id,
            actor="mission-control",
            idempotency_key="archive-1",
        )
        assert result == {
            "root_id": root_id,
            "archived_task_count": 1,
            "actor": "mission-control",
            "idempotency_key": "archive-1",
            "replayed": False,
        }
        assert kb.get_task(conn, root_id).status == "archived"
        assert kb.archive_program(
            conn,
            root_id,
            actor="mission-control",
            idempotency_key="archive-1",
        ) == {**result, "replayed": True}
    finally:
        conn.close()


def test_change_request_prepare_apply_is_bound_audited_and_exactly_once(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={
                "operation": "extend_deadline",
                "new_deadline": now + 600,
                "reason": "More time for verification",
            },
            actor="operator",
            idempotency_key="prepare-1",
        )
        assert prepared["root_id"] == root_id
        assert prepared["status"] == "prepared"
        assert prepared["request_digest"].startswith("sha256:")
        assert prepared["replayed"] is False
        assert kb.prepare_program_change_request(
            conn,
            root_id,
            change={
                "operation": "extend_deadline",
                "new_deadline": now + 600,
                "reason": "More time for verification",
            },
            actor="operator",
            idempotency_key="prepare-1",
        ) == {**prepared, "replayed": True}

        applied = kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-1",
        )
        assert applied["status"] == "applied"
        assert applied["effective_deadline"] == now + 600
        assert applied["replayed"] is False
        assert kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-1",
        ) == {**applied, "replayed": True}
        assert kb.prepare_program_change_request(
            conn,
            root_id,
            change={
                "operation": "extend_deadline",
                "new_deadline": now + 600,
                "reason": "More time for verification",
            },
            actor="operator",
            idempotency_key="prepare-1",
        ) == {**prepared, "status": "applied", "replayed": True}
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_request_events WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == 2
        )
        for statement in (
            "UPDATE program_change_request_events SET actor = 'tampered'",
            "DELETE FROM program_change_request_events",
            "UPDATE program_lifecycle_events SET actor = 'tampered'",
            "DELETE FROM program_lifecycle_events",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="append-only"):
                conn.execute(statement)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "mutation",
    [
        "UPDATE program_change_requests SET status = 'applied' WHERE request_id = ?",
        "UPDATE program_change_requests SET status = 'applied', request_digest = "
        "'sha256:0000000000000000000000000000000000000000000000000000000000000000' "
        "WHERE request_id = ?",
        "UPDATE program_change_requests SET root_id = 'forged-root' WHERE request_id = ?",
        "UPDATE program_change_requests SET prepare_actor = 'forged' WHERE request_id = ?",
        "UPDATE program_change_requests SET prepare_idempotency_key = 'forged' WHERE request_id = ?",
        "UPDATE program_change_requests SET prepare_fingerprint = "
        "'sha256:0000000000000000000000000000000000000000000000000000000000000000' "
        "WHERE request_id = ?",
        "UPDATE program_change_requests SET root_fingerprint = 'sha256:forged' "
        "WHERE request_id = ?",
        "UPDATE program_change_requests SET prepared_state_digest = 'sha256:forged' "
        "WHERE request_id = ?",
        "UPDATE program_change_requests SET prepared_at = prepared_at + 1 WHERE request_id = ?",
    ],
)
def test_prepare_replay_rejects_mutable_request_row_forgery_without_events(
    tmp_path, monkeypatch, mutation
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        change = {"operation": "extend_deadline", "new_deadline": now + 600}
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change=change,
            actor="operator",
            idempotency_key="prepare-row-seal",
        )
        conn.execute(mutation, (prepared["request_id"],))
        conn.commit()

        with pytest.raises(ValueError, match="integrity check failed"):
            kb.prepare_program_change_request(
                conn,
                root_id,
                change=change,
                actor="operator",
                idempotency_key="prepare-row-seal",
            )

        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_request_events WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == 1
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT effective_deadline FROM program_lifecycle WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == now + 300
        )
    finally:
        conn.close()


def test_applied_replay_rejects_joint_change_and_digest_forgery_against_prepared_event(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={
                "operation": "extend_deadline",
                "new_deadline": now + 600,
                "reason": "original",
            },
            actor="operator",
            idempotency_key="prepare-joint-seal",
        )
        kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-joint-seal",
        )
        forged_json = '{"new_deadline":1700000600,"operation":"extend_deadline","reason":"forged"}'
        forged_digest = (
            "sha256:" + hashlib.sha256(forged_json.encode("utf-8")).hexdigest()
        )
        conn.execute(
            "UPDATE program_change_requests SET change_json = ?, request_digest = ? "
            "WHERE request_id = ?",
            (forged_json, forged_digest, prepared["request_id"]),
        )
        conn.commit()
        lifecycle_before = dict(
            conn.execute(
                "SELECT * FROM program_lifecycle WHERE root_id = ?", (root_id,)
            ).fetchone()
        )
        event_count = conn.execute(
            "SELECT COUNT(*) FROM program_change_request_events WHERE request_id = ?",
            (prepared["request_id"],),
        ).fetchone()[0]

        with pytest.raises(ValueError, match="integrity check failed"):
            kb.apply_program_change_request(
                conn,
                root_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key="apply-joint-seal",
            )

        assert (
            dict(
                conn.execute(
                    "SELECT * FROM program_lifecycle WHERE root_id = ?", (root_id,)
                ).fetchone()
            )
            == lifecycle_before
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_request_events WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == event_count
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("column", "tampered_value"),
    [
        (
            "result_json",
            '{"actor":"forged","effective_deadline":1700999999,'
            '"idempotency_key":"apply-replay-seal","operation":"extend_deadline",'
            '"replayed":false,"request_id":"forged","root_id":"forged",'
            '"status":"applied"}',
        ),
        ("change_json", '{"operation":"archive"}'),
        ("request_digest", "sha256:" + "0" * 64),
        ("applied_at", 1_700_000_001),
    ],
)
def test_applied_change_replay_rejects_mutable_request_or_result_tamper_without_side_effects(
    tmp_path, monkeypatch, column, tampered_value
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator",
            idempotency_key="prepare-replay-seal",
        )
        kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-replay-seal",
        )
        lifecycle_before = dict(
            conn.execute(
                "SELECT * FROM program_lifecycle WHERE root_id = ?", (root_id,)
            ).fetchone()
        )
        request_event_count = conn.execute(
            "SELECT COUNT(*) FROM program_change_request_events WHERE request_id = ?",
            (prepared["request_id"],),
        ).fetchone()[0]
        lifecycle_event_count = conn.execute(
            "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
            (root_id,),
        ).fetchone()[0]

        conn.execute(
            f"UPDATE program_change_requests SET {column} = ? WHERE request_id = ?",
            (tampered_value, prepared["request_id"]),
        )
        conn.commit()

        with pytest.raises(ValueError, match="integrity check failed"):
            kb.apply_program_change_request(
                conn,
                root_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key="apply-replay-seal",
            )

        assert (
            dict(
                conn.execute(
                    "SELECT * FROM program_lifecycle WHERE root_id = ?", (root_id,)
                ).fetchone()
            )
            == lifecycle_before
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_request_events WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == request_event_count
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == lifecycle_event_count
        )
    finally:
        conn.close()


def test_applied_archive_change_replay_verifies_and_returns_sealed_result(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        conn.execute("UPDATE tasks SET status = 'done' WHERE id = ?", (root_id,))
        conn.commit()
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "archive", "reason": "Delivery verified"},
            actor="operator",
            idempotency_key="prepare-archive-replay",
        )
        applied = kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-archive-replay",
        )
        assert applied == {
            "request_id": prepared["request_id"],
            "root_id": root_id,
            "operation": "archive",
            "status": "applied",
            "archived_task_count": 1,
            "actor": "approver",
            "idempotency_key": "apply-archive-replay",
            "replayed": False,
        }
        assert kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-archive-replay",
        ) == {**applied, "replayed": True}
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_request_events WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == 2
        )
    finally:
        conn.close()


def test_change_request_fails_closed_for_wrong_root_stale_state_and_key_reuse(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=now)
        other_id = _root(conn, now=now + 1)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator",
            idempotency_key="prepare-1",
        )
        with pytest.raises(ValueError, match="idempotency key request does not match"):
            kb.prepare_program_change_request(
                conn,
                root_id,
                change={"operation": "extend_deadline", "new_deadline": now + 700},
                actor="operator",
                idempotency_key="prepare-1",
            )
        with pytest.raises(ValueError, match="root binding"):
            kb.apply_program_change_request(
                conn,
                other_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key="wrong-root",
            )

        kb.extend_program_deadline(
            conn,
            root_id,
            new_deadline=now + 500,
            actor="operator",
            idempotency_key="intervening-change",
        )
        with pytest.raises(ValueError, match="stale"):
            kb.apply_program_change_request(
                conn,
                root_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key="apply-stale",
            )
        assert (
            conn.execute(
                "SELECT status FROM program_change_requests WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == "prepared"
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    "change",
    [
        {"operation": "extend_deadline", "new_deadline": 123, "extra": True},
        {"operation": "archive", "reason": "bad\ntext"},
        {"operation": "unknown"},
    ],
)
def test_change_request_rejects_non_strict_or_unclean_content(tmp_path, change):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        with pytest.raises(ValueError):
            kb.prepare_program_change_request(
                conn,
                root_id,
                change=change,
                actor="operator",
                idempotency_key="prepare-invalid",
            )
        assert (
            conn.execute("SELECT COUNT(*) FROM program_change_requests").fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_concurrent_change_request_apply_uses_separate_connections_exactly_once(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    db_path = tmp_path / "kanban.db"
    with kb.connect(db_path) as setup:
        root_id = _root(setup, now=now)
        prepared = kb.prepare_program_change_request(
            setup,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator",
            idempotency_key="prepare-concurrent",
        )

    barrier = threading.Barrier(2)

    def apply_once():
        conn = kb.connect(db_path)
        try:
            barrier.wait(timeout=5)
            return kb.apply_program_change_request(
                conn,
                root_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key="apply-concurrent",
            )
        finally:
            conn.close()

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _: apply_once(), range(2)))
    assert sorted(result["replayed"] for result in results) == [False, True]
    with kb.connect(db_path) as verify:
        assert (
            verify.execute(
                "SELECT COUNT(*) FROM program_change_request_events "
                "WHERE request_id = ? AND event_type = 'applied'",
                (prepared["request_id"],),
            ).fetchone()[0]
            == 1
        )


def test_lifecycle_schema_preflight_rejects_near_shape_without_mutation(tmp_path):
    db_path = tmp_path / "kanban.db"
    conn = kb.connect(db_path)
    conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("DROP TABLE program_change_requests")
    conn.execute(
        "CREATE TABLE program_change_requests (request_id TEXT PRIMARY KEY, root_id TEXT)"
    )
    conn.commit()
    conn.close()
    kb._INITIALIZED_PATHS.discard(str(db_path.resolve()))
    before = db_path.read_bytes()
    before_schema = (
        sqlite3
        .connect(db_path)
        .execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        )
        .fetchall()
    )

    with pytest.raises(ValueError, match="unsupported program lifecycle schema"):
        kb.connect(db_path)

    assert db_path.read_bytes() == before
    assert not db_path.with_name(db_path.name + "-wal").exists()
    assert not db_path.with_name(db_path.name + "-shm").exists()
    after = (
        sqlite3
        .connect(db_path)
        .execute(
            "SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name"
        )
        .fetchall()
    )
    assert after == before_schema


def _creation_authority(claimed):
    return kb.CreationAuthority(
        task_id=claimed.id,
        run_id=claimed.current_run_id,
        claim_lock=claimed.claim_lock,
        actor_profile=claimed.assignee,
    )


def test_extended_effective_deadline_controls_child_creation_admission(
    tmp_path, monkeypatch
):
    started = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: started)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=started)
        claimed = kb.claim_task(conn, root_id, ttl_seconds=1_000)
        assert claimed is not None
        authority = _creation_authority(claimed)
        kb.extend_program_deadline(
            conn,
            root_id,
            new_deadline=started + 600,
            actor="mission-control",
            idempotency_key="extend-for-child",
        )

        monkeypatch.setattr(kb.time, "time", lambda: started + 301)
        child_id = kb.create_task(
            conn,
            title="authorized after original deadline",
            assignee="worker",
            current_orchestrator_task_id=root_id,
            creation_authority=authority,
        )
        assert kb.get_task(conn, child_id) is not None

        before = conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0]
        monkeypatch.setattr(kb.time, "time", lambda: started + 600)
        with pytest.raises(ValueError, match="deadline exceeded"):
            kb.create_task(
                conn,
                title="too late",
                assignee="worker",
                current_orchestrator_task_id=root_id,
                creation_authority=authority,
            )
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == before
    finally:
        conn.close()


def test_direct_extension_rejects_deadline_not_after_transaction_time_without_mutation(
    tmp_path, monkeypatch
):
    started = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: started)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=started)
        monkeypatch.setattr(kb.time, "time", lambda: started + 500)
        with pytest.raises(ValueError, match="future"):
            kb.extend_program_deadline(
                conn,
                root_id,
                new_deadline=started + 400,
                actor="mission-control",
                idempotency_key="past-candidate",
            )
        # The failed transaction must also roll back lazy lifecycle creation.
        assert (
            conn.execute(
                "SELECT effective_deadline FROM program_lifecycle WHERE root_id = ?",
                (root_id,),
            ).fetchone()
            is None
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_prepared_extension_expiring_before_apply_fails_without_mutation(
    tmp_path, monkeypatch
):
    started = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: started)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=started)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": started + 400},
            actor="operator",
            idempotency_key="prepare-expiring",
        )
        monkeypatch.setattr(kb.time, "time", lambda: started + 400)
        with pytest.raises(ValueError, match="future"):
            kb.apply_program_change_request(
                conn,
                root_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key="apply-expired",
            )
        assert (
            conn.execute(
                "SELECT status FROM program_change_requests WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == "prepared"
        )
        assert (
            conn.execute(
                "SELECT effective_deadline FROM program_lifecycle WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == started + 300
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_request_events "
                "WHERE request_id = ? AND event_type = 'applied'",
                (prepared["request_id"],),
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_prepare_rejects_already_past_extension_without_mutation(tmp_path, monkeypatch):
    started = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: started)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=started)
        monkeypatch.setattr(kb.time, "time", lambda: started + 500)
        with pytest.raises(ValueError, match="future"):
            kb.prepare_program_change_request(
                conn,
                root_id,
                change={"operation": "extend_deadline", "new_deadline": started + 400},
                actor="operator",
                idempotency_key="prepare-past",
            )
        assert (
            conn.execute("SELECT COUNT(*) FROM program_change_requests").fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_request_events"
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == 0
        )
        assert (
            conn.execute(
                "SELECT effective_deadline FROM program_lifecycle WHERE root_id = ?",
                (root_id,),
            ).fetchone()
            is None
        )
    finally:
        conn.close()


@pytest.mark.parametrize("tampered_column", ["change_json", "request_digest"])
def test_apply_rejects_tampered_prepared_request_seal_without_side_effects(
    tmp_path, monkeypatch, tampered_column
):
    started = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: started)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=started)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": started + 600},
            actor="operator",
            idempotency_key=f"prepare-tamper-{tampered_column}",
        )
        if tampered_column == "change_json":
            value = json.dumps(
                {"operation": "extend_deadline", "new_deadline": started + 900},
                sort_keys=True,
                separators=(",", ":"),
            )
        else:
            value = "sha256:" + "0" * 64
        conn.execute(
            f"UPDATE program_change_requests SET {tampered_column} = ? WHERE request_id = ?",
            (value, prepared["request_id"]),
        )

        with pytest.raises(ValueError, match="integrity"):
            kb.apply_program_change_request(
                conn,
                root_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key=f"apply-tamper-{tampered_column}",
            )
        assert (
            conn.execute(
                "SELECT status FROM program_change_requests WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == "prepared"
        )
        assert (
            conn.execute(
                "SELECT effective_deadline FROM program_lifecycle WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == started + 300
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("project_id", "different-project"),
        ("tenant", "different-tenant"),
        (
            "orchestration_policy",
            _policy(max_concurrency=1).to_json(),
        ),
        ("created_at", 1_700_000_001),
    ],
)
def test_apply_binds_material_root_authority_without_side_effects(
    tmp_path, monkeypatch, column, value
):
    started = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: started)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, now=started)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": started + 600},
            actor="operator",
            idempotency_key=f"prepare-root-{column}",
        )
        conn.execute(f"UPDATE tasks SET {column} = ? WHERE id = ?", (value, root_id))

        with pytest.raises(ValueError, match="stale|root binding"):
            kb.apply_program_change_request(
                conn,
                root_id,
                request_id=prepared["request_id"],
                actor="approver",
                idempotency_key=f"apply-root-{column}",
            )
        assert (
            conn.execute(
                "SELECT status FROM program_change_requests WHERE request_id = ?",
                (prepared["request_id"],),
            ).fetchone()[0]
            == "prepared"
        )
        assert (
            conn.execute(
                "SELECT effective_deadline FROM program_lifecycle WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == started + 300
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_lifecycle_events WHERE root_id = ?",
                (root_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()
