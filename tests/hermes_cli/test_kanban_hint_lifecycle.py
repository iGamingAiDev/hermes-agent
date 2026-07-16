from __future__ import annotations

import json
import sqlite3
import threading
import time

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_program_control as pc


@pytest.fixture
def active(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        task = kb.create_task(conn, title="worker", assignee="worker")
        conn.execute("UPDATE tasks SET status='ready',orchestration_root_id=id WHERE id=?", (task,))
        claimed = kb.claim_task(conn, task, claimer="host:claim")
        assert claimed is not None
        row = conn.execute("SELECT current_run_id FROM tasks WHERE id=?", (task,)).fetchone()
        yield conn, task, int(row[0])


def _hint(conn, task_id, suffix, text="hint"):
    root = conn.execute("SELECT orchestration_root_id FROM tasks WHERE id=?", (task_id,)).fetchone()[0]
    conn.execute(
        "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
        "expected_node_version,committed_node_version,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
        (root, f"h_{suffix}", task_id, text, "control:owner", suffix, 0, 1, "recorded", int(suffix)),
    )


def test_poll_order_bounds_ack_idempotency_and_event_privacy(active):
    conn, task, run = active
    for i in range(1, 7):
        _hint(conn, task, str(i), "secret-" + str(i))
    got = pc.poll_hints(conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker")
    assert [x["hint_id"] for x in got] == ["h_1", "h_2", "h_3", "h_4"]
    assert pc.ack_hint(conn, hint_id="h_1", task_id=task, run_id=run,
                       claim_lock="host:claim", profile="worker",
                       state="incorporated", reason_code="incorporated")
    assert pc.ack_hint(conn, hint_id="h_1", task_id=task, run_id=run,
                       claim_lock="host:claim", profile="worker",
                       state="incorporated", reason_code="incorporated")
    with pytest.raises(pc.ProgramControlError, match="hint_ack_conflict"):
        pc.ack_hint(conn, hint_id="h_1", task_id=task, run_id=run,
                    claim_lock="host:claim", profile="worker",
                    state="rejected", reason_code="unsafe")
    payloads = [json.loads(r[0]) for r in conn.execute(
        "SELECT payload FROM task_events WHERE kind LIKE 'hint_%' AND payload IS NOT NULL"
    )]
    assert payloads
    assert all(set(p) == {"hint_id", "node_id", "run_id", "state", "reason_code"} for p in payloads)
    assert "secret" not in json.dumps(payloads)


def test_ack_batch_rolls_back_updates_and_events_on_second_row_abort(active):
    conn, task, run = active
    _hint(conn, task, "1")
    _hint(conn, task, "2")
    batch = pc.poll_hints(conn, task_id=task, run_id=run,
                          claim_lock="host:claim", profile="worker")
    before_events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
    conn.execute(
        "CREATE TRIGGER abort_second_ack BEFORE UPDATE OF state ON program_hints "
        "WHEN OLD.hint_id='h_2' AND NEW.state='incorporated' "
        "BEGIN SELECT RAISE(ABORT, 'second ack aborted'); END"
    )
    with pytest.raises(sqlite3.IntegrityError, match="second ack aborted"):
        pc.ack_hints(conn, hint_ids=[item["hint_id"] for item in batch],
                     task_id=task, run_id=run, claim_lock="host:claim", profile="worker",
                     state="incorporated", reason_code="incorporated")
    assert dict(conn.execute("SELECT hint_id,state FROM program_hints")) == {
        "h_1": "seen", "h_2": "seen"
    }
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == before_events


def test_ack_batch_rejects_mixed_replay_duplicates_bool_and_stale_without_mutation(active):
    conn, task, run = active
    for suffix in ("1", "2"):
        _hint(conn, task, suffix)
    pc.poll_hints(conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker")
    pc.ack_hint(conn, hint_id="h_1", task_id=task, run_id=run,
                claim_lock="host:claim", profile="worker",
                state="incorporated", reason_code="incorporated")
    snapshot = list(conn.execute("SELECT hint_id,state FROM program_hints ORDER BY hint_id"))
    events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
    for ids, expected in [(["h_1", "h_2"], "hint_ack_conflict"),
                          (["h_2", "h_2"], "invalid_hint_ids"),
                          ([True], "invalid_hint_ids")]:
        with pytest.raises(pc.ProgramControlError, match=expected):
            pc.ack_hints(conn, hint_ids=ids, task_id=task, run_id=run,
                         claim_lock="host:claim", profile="worker",
                         state="incorporated", reason_code="incorporated")
    assert list(conn.execute("SELECT hint_id,state FROM program_hints ORDER BY hint_id")) == snapshot
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events

    conn.execute("UPDATE tasks SET claim_expires=0 WHERE id=?", (task,))
    with pytest.raises(pc.ProgramControlError, match="stale_hint_authority"):
        pc.ack_hints(conn, hint_ids=["h_2"], task_id=task, run_id=run,
                     claim_lock="host:claim", profile="worker",
                     state="incorporated", reason_code="incorporated")
    assert conn.execute("SELECT state FROM program_hints WHERE hint_id='h_2'").fetchone()[0] == "seen"


def test_ack_batch_exact_full_replay_is_idempotent_without_events(active):
    conn, task, run = active
    for suffix in ("1", "2"):
        _hint(conn, task, suffix)
    batch = pc.poll_hints(conn, task_id=task, run_id=run,
                          claim_lock="host:claim", profile="worker")
    ids = [item["hint_id"] for item in batch]
    assert pc.ack_hints(conn, hint_ids=ids, task_id=task, run_id=run,
                        claim_lock="host:claim", profile="worker",
                        state="incorporated", reason_code="incorporated")
    events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
    assert pc.ack_hints(conn, hint_ids=ids, task_id=task, run_id=run,
                        claim_lock="host:claim", profile="worker",
                        state="incorporated", reason_code="incorporated")
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events


@pytest.mark.parametrize("invalidate", ["expiry", "completion", "reclaim"])
def test_ack_exact_batch_replay_precedes_stale_authority_checks(active, invalidate):
    conn, task, run = active
    for suffix in ("1", "2"):
        _hint(conn, task, suffix)
    ids = [item["hint_id"] for item in pc.poll_hints(
        conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker"
    )]
    assert pc.ack_hints(conn, hint_ids=ids, task_id=task, run_id=run,
                        claim_lock="host:claim", profile="worker",
                        state="incorporated", reason_code="incorporated")
    if invalidate == "expiry":
        conn.execute("UPDATE tasks SET claim_expires=0 WHERE id=?", (task,))
    elif invalidate == "completion":
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (task,))
    else:
        assert kb.reclaim_task(conn, task, signal_fn=lambda *_: None)
    snapshot = list(conn.execute("SELECT hint_id,state,terminal_reason_code FROM program_hints ORDER BY hint_id"))
    events = conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0]
    assert pc.ack_hints(conn, hint_ids=ids, task_id=task, run_id=run,
                        claim_lock="host:claim", profile="worker",
                        state="incorporated", reason_code="incorporated")
    assert list(conn.execute("SELECT hint_id,state,terminal_reason_code FROM program_hints ORDER BY hint_id")) == snapshot
    assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] == events


def test_reclaim_reconciles_seen_and_preserves_recorded(active):
    conn, task, run = active
    _hint(conn, task, "1")
    _hint(conn, task, "2")
    assert len(pc.poll_hints(conn, task_id=task, run_id=run,
                             claim_lock="host:claim", profile="worker")) == 2
    _hint(conn, task, "3")
    assert kb.reclaim_task(conn, task, signal_fn=lambda *_: None)
    states = dict(conn.execute("SELECT hint_id,state FROM program_hints"))
    assert states == {"h_1": "reconcile", "h_2": "reconcile", "h_3": "recorded"}
    with pytest.raises(pc.ProgramControlError, match="hint_ack_conflict"):
        pc.ack_hint(conn, hint_id="h_1", task_id=task, run_id=run,
                    claim_lock="host:claim", profile="worker",
                    state="incorporated", reason_code="incorporated")


def test_poll_participates_in_outer_rollback(active):
    conn, task, run = active
    _hint(conn, task, "1")
    conn.execute("BEGIN IMMEDIATE")
    assert pc.poll_hints(conn, task_id=task, run_id=run,
                         claim_lock="host:claim", profile="worker")
    conn.rollback()
    assert conn.execute("SELECT state FROM program_hints").fetchone()[0] == "recorded"


def test_two_connections_cannot_deliver_same_hint(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect_closing() as setup:
        task = kb.create_task(setup, title="worker", assignee="worker")
        setup.execute("UPDATE tasks SET status='ready',orchestration_root_id=id WHERE id=?", (task,))
        kb.claim_task(setup, task, claimer="host:claim")
        run = setup.execute("SELECT current_run_id FROM tasks WHERE id=?", (task,)).fetchone()[0]
        _hint(setup, task, "1")
    barrier = threading.Barrier(2)
    results = []
    def poll():
        with kb.connect_closing() as conn:
            barrier.wait()
            results.append(pc.poll_hints(conn, task_id=task, run_id=run,
                                         claim_lock="host:claim", profile="worker"))
    threads = [threading.Thread(target=poll) for _ in range(2)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    assert sorted(len(result) for result in results) == [0, 1]


@pytest.mark.parametrize("state,reason,delivered", [
    ("recorded", None, False), ("seen", None, True),
    ("incorporated", "incorporated", True),
    ("deferred", "terminal_before_delivery", False),
    ("deferred", "not_applicable", True), ("deferred", "superseded", True),
    ("rejected", "unsafe", True), ("rejected", "invalid", True),
    ("reconcile", "stale_seen", True),
])
def test_schema_accepts_each_canonical_lifecycle_shape(active, state, reason, delivered):
    conn, task, run = active
    root = conn.execute("SELECT orchestration_root_id FROM tasks WHERE id=?", (task,)).fetchone()[0]
    terminal = state not in {"recorded", "seen"}
    conn.execute(
        "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
        "expected_node_version,committed_node_version,state,run_id,claim_lock,profile,"
        "created_at,delivered_at,terminal_at,terminal_reason_code) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (root, f"shape-{state}-{delivered}", task, "x", "control:owner", f"shape-{state}-{delivered}",
         0, 1, state, run if delivered else None, "host:claim" if delivered else None,
         "worker" if delivered else None, 1, 1 if delivered else None,
         2 if terminal else None, reason),
    )


@pytest.mark.parametrize("state,reason", [
    ("recorded", "terminal_before_delivery"), ("seen", "stale_seen"),
    ("incorporated", "stale_seen"), ("deferred", "unsafe"),
    ("rejected", "superseded"), ("reconcile", "incorporated"),
])
def test_schema_rejects_wrong_state_reason_pairs_direct_sql(active, state, reason):
    conn, task, run = active
    root = conn.execute("SELECT orchestration_root_id FROM tasks WHERE id=?", (task,)).fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
            "expected_node_version,committed_node_version,state,run_id,claim_lock,profile,"
            "created_at,delivered_at,terminal_at,terminal_reason_code) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (root, "bad", task, "x", "control:owner", "bad", 0, 1, state, run,
             "host:claim", "worker", 1, 1, 2, reason),
        )


@pytest.mark.parametrize("reason,delivered", [
    ("terminal_before_delivery", True),
    ("not_applicable", False),
    ("superseded", False),
])
def test_deferred_schema_rejects_reason_delivery_cross_products(
    active, reason, delivered
):
    conn, task, run = active
    root = conn.execute(
        "SELECT orchestration_root_id FROM tasks WHERE id=?", (task,)
    ).fetchone()[0]
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute(
            "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
            "expected_node_version,committed_node_version,state,run_id,claim_lock,profile,"
            "created_at,delivered_at,terminal_at,terminal_reason_code) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (root, f"bad-{reason}-{delivered}", task, "x", "control:owner",
             f"bad-{reason}-{delivered}", 0, 1, "deferred", run if delivered else None,
             "host:claim" if delivered else None, "worker" if delivered else None,
             1, 1 if delivered else None, 2, reason),
        )


def test_nonterminal_reclaim_preserves_recorded_but_reconciles_seen(active):
    conn, task, run = active
    _hint(conn, task, "1")
    _hint(conn, task, "2")
    pc.poll_hints(conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker")
    _hint(conn, task, "3")
    assert kb.reclaim_task(conn, task, signal_fn=lambda *_: None)
    assert dict(conn.execute("SELECT hint_id,state FROM program_hints")) == {
        "h_1": "reconcile", "h_2": "reconcile", "h_3": "recorded"
    }


def test_terminal_complete_defers_recorded_and_reconciles_seen(active):
    conn, task, run = active
    _hint(conn, task, "1")
    pc.poll_hints(conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker")
    _hint(conn, task, "2")
    assert kb.complete_task(conn, task, expected_run_id=run)
    assert dict(conn.execute("SELECT hint_id,state FROM program_hints")) == {
        "h_1": "reconcile", "h_2": "deferred"
    }


def test_poll_uses_closed_expiry_boundary_consistent_with_claim_reclaim(active, monkeypatch):
    conn, task, run = active
    now = int(time.time())
    monkeypatch.setattr(pc.time, "time", lambda: now)
    conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (now, task))
    conn.execute("UPDATE task_runs SET claim_expires=? WHERE id=?", (now, run))
    _hint(conn, task, "1")
    assert pc.poll_hints(conn, task_id=task, run_id=run,
                         claim_lock="host:claim", profile="worker")
    conn.execute("UPDATE tasks SET claim_expires=? WHERE id=?", (now - 1, task))
    with pytest.raises(pc.ProgramControlError, match="stale_hint_authority"):
        pc.poll_hints(conn, task_id=task, run_id=run,
                      claim_lock="host:claim", profile="worker")


def test_poll_codepoint_bound_has_no_partial_item(active):
    conn, task, run = active
    _hint(conn, task, "1", "a" * 7999)
    _hint(conn, task, "2", "U0001f40dU0001f40d")
    got = pc.poll_hints(conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker")
    assert [len(item["text"]) for item in got] == [7999]
    assert conn.execute("SELECT state FROM program_hints WHERE hint_id='h_2'").fetchone()[0] == "recorded"


def test_reconcile_cas_emits_no_duplicate_event(active):
    conn, task, run = active
    _hint(conn, task, "1")
    pc.poll_hints(conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker")
    conn.execute("UPDATE tasks SET claim_expires=0 WHERE id=?", (task,))
    assert pc.reconcile_stale_hints(conn) == 1
    assert pc.reconcile_stale_hints(conn) == 0
    assert conn.execute("SELECT COUNT(*) FROM task_events WHERE kind='hint_reconcile'").fetchone()[0] == 1


@pytest.mark.parametrize("state,reason", [
    ("incorporated", "not_applicable"), ("deferred", "unsafe"),
    ("rejected", "superseded"), ("reconcile", "stale_seen"),
])
def test_ack_enforces_exact_public_terminal_allowlists(active, state, reason):
    conn, task, run = active
    _hint(conn, task, "1")
    pc.poll_hints(conn, task_id=task, run_id=run, claim_lock="host:claim", profile="worker")
    with pytest.raises(pc.ProgramControlError):
        pc.ack_hint(conn, hint_id="h_1", task_id=task, run_id=run,
                    claim_lock="host:claim", profile="worker",
                    state=state, reason_code=reason)


def test_ack_rejects_bool_run_id_without_int_ambiguity(active):
    conn, task, _run = active
    with pytest.raises(pc.ProgramControlError, match="invalid_run_id"):
        pc.ack_hint(conn, hint_id="h_1", task_id=task, run_id=True,
                    claim_lock="host:claim", profile="worker",
                    state="incorporated", reason_code="incorporated")


def test_terminal_helper_scopes_repeated_hint_id_by_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        tasks = []
        for title in ("one", "two"):
            task = kb.create_task(conn, title=title, assignee="worker")
            conn.execute("UPDATE tasks SET status='ready',orchestration_root_id=id WHERE id=?", (task,))
            tasks.append(task)
        for task in tasks:
            root = task
            conn.execute(
                "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
                "expected_node_version,committed_node_version,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                (root, "same", task, "private", "control:owner", "same", 0, 1, "recorded", 1),
            )
        assert kb.complete_task(conn, tasks[0])
        assert dict(conn.execute("SELECT root_id,state FROM program_hints")) == {
            tasks[0]: "deferred", tasks[1]: "recorded"
        }


def test_poll_vs_complete_serializes_to_one_exact_outcome(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect_closing() as setup:
        task = kb.create_task(setup, title="worker", assignee="worker")
        setup.execute("UPDATE tasks SET status='ready',orchestration_root_id=id WHERE id=?", (task,))
        kb.claim_task(setup, task, claimer="host:claim")
        run = setup.execute("SELECT current_run_id FROM tasks WHERE id=?", (task,)).fetchone()[0]
        _hint(setup, task, "1")
    barrier = threading.Barrier(2)
    outcomes = []
    def poll():
        with kb.connect_closing() as conn:
            barrier.wait()
            try:
                outcomes.append(("poll", len(pc.poll_hints(conn, task_id=task, run_id=run,
                                                            claim_lock="host:claim", profile="worker"))))
            except pc.ProgramControlError as exc:
                outcomes.append(("poll", exc.code))
    def complete():
        with kb.connect_closing() as conn:
            barrier.wait()
            outcomes.append(("complete", kb.complete_task(conn, task, expected_run_id=run)))
    threads = [threading.Thread(target=poll), threading.Thread(target=complete)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    with kb.connect_closing() as conn:
        row = conn.execute("SELECT state FROM program_hints").fetchone()[0]
        assert row in {"deferred", "reconcile"}
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (task,)).fetchone()[0] == "done"
    assert ("complete", True) in outcomes
    assert any(item in outcomes for item in (("poll", 1), ("poll", "stale_hint_authority")))


def test_ack_vs_reclaim_serializes_terminal_state(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect_closing() as setup:
        task = kb.create_task(setup, title="worker", assignee="worker")
        setup.execute("UPDATE tasks SET status='ready',orchestration_root_id=id WHERE id=?", (task,))
        kb.claim_task(setup, task, claimer="host:claim")
        run = setup.execute("SELECT current_run_id FROM tasks WHERE id=?", (task,)).fetchone()[0]
        _hint(setup, task, "1")
        pc.poll_hints(setup, task_id=task, run_id=run,
                      claim_lock="host:claim", profile="worker")
    barrier = threading.Barrier(2)
    outcomes = []
    def ack():
        with kb.connect_closing() as conn:
            barrier.wait()
            try:
                outcomes.append(("ack", pc.ack_hint(
                    conn, hint_id="h_1", task_id=task, run_id=run,
                    claim_lock="host:claim", profile="worker",
                    state="incorporated", reason_code="incorporated")))
            except pc.ProgramControlError as exc:
                outcomes.append(("ack", exc.code))
    def reclaim():
        with kb.connect_closing() as conn:
            barrier.wait()
            outcomes.append(("reclaim", kb.reclaim_task(conn, task, signal_fn=lambda *_: None)))
    threads = [threading.Thread(target=ack), threading.Thread(target=reclaim)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    with kb.connect_closing() as conn:
        state = conn.execute("SELECT state FROM program_hints").fetchone()[0]
    assert state in {"incorporated", "reconcile"}
    assert ("reclaim", True) in outcomes
    assert any(item in outcomes for item in (("ack", True), ("ack", "stale_hint_authority"),
                                              ("ack", "hint_ack_conflict")))
