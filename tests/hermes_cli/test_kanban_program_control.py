from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import sqlite3
import threading
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_program_control as pc
from hermes_cli._parser import build_top_level_parser


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    kb.init_db()
    with kb.connect_closing() as conn:
        policy = kb.OrchestrationPolicy(
            allowed_assignees=("planner", "worker"),
            orchestrator_assignees=("planner",), max_depth=3, max_tasks=20,
            max_runtime_seconds=60, max_concurrency=2,
            max_wall_clock_seconds=300, goal_max_turns=5,
        )
        root = kb.create_task(conn, title="Root", assignee="planner", orchestration_policy=policy)
        child = kb.create_task(conn, title="Child", assignee="worker")
        conn.execute("UPDATE tasks SET orchestration_root_id=?, orchestration_depth=1, "
                     "orchestration_parent_id=? WHERE id=?", (root, root, child))
        other = kb.create_task(conn, title="Other", assignee="planner", orchestration_policy=policy)
    return root, child, other


def _open(root, child, **updates):
    request = {
        "root_id": root, "checkpoint_id": "checkpoint-1", "node_id": child,
        "title": "Choose release strategy",
        "options": [
            {"option_id": "blue", "label": "Blue", "summary": "Use blue",
             "benefits": ["Low risk"], "risks": ["Extra capacity"],
             "reversibility": "reversible", "security_impact": "none",
             "cost_impact": "temporary", "operations_impact": "dual run"},
            {"option_id": "green", "label": "Green", "summary": "Use green",
             "benefits": ["Simple"], "risks": ["Cutover"],
             "reversibility": "partially_reversible", "security_impact": "none",
             "cost_impact": "none", "operations_impact": "cutover"},
        ],
        "recommended_option_id": "blue", "recommendation_rationale": "Safer",
        "affected_node_ids": [child], "expected_node_version": 0,
        "idempotency_key": "open-1", "actor": "agent:planner",
    }
    request.update(updates)
    return request


def _hint(root, child, **updates):
    request = {"root_id": root, "node_id": child, "text": "Check the canary first",
               "expected_node_version": 0, "idempotency_key": "hint-1",
               "actor": "control:owner"}
    request.update(updates)
    return request


def _select(root, **updates):
    request = {"root_id": root, "checkpoint_id": "checkpoint-1",
               "option_id": "blue", "expected_version": 1,
               "idempotency_key": "select-1", "actor": "control:owner"}
    request.update(updates)
    return request


def test_phase2_pending_decision_gates_recompute_dispatch_and_direct_claim(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))
        pc.open_decision(conn, _open(root, child))
        assert kb.recompute_ready(conn) == 0
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (child,)).fetchone()[0] == "todo"

        # Even stale/manual writers cannot bypass the authoritative claim gate.
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        before = conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (child,)).fetchone()[0]
        assert kb.claim_task(conn, child, claimer="manual") is None
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (child,)).fetchone()[0] == "ready"
        assert conn.execute("SELECT COUNT(*) FROM task_runs WHERE task_id=?", (child,)).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (child,)).fetchone()[0] == before

        spawned = []
        result = kb.dispatch_once(conn, spawn_fn=lambda task, *_: spawned.append(task.id))
        assert result.spawned == []
        assert spawned == []


def test_phase2_pending_decision_denies_direct_review_claim_without_mutation(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (child,))
        pc.open_decision(conn, _open(root, child))
        before_task = tuple(conn.execute(
            "SELECT status,current_run_id,claim_lock,claim_expires,started_at "
            "FROM tasks WHERE id=?", (child,),
        ).fetchone())
        before_events = conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=?", (child,),
        ).fetchone()[0]
        before_runs = conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (child,),
        ).fetchone()[0]

        assert kb.claim_review_task(
            conn, child, claimer="SECRET-REVIEW-CLAIMER"
        ) is None

        assert tuple(conn.execute(
            "SELECT status,current_run_id,claim_lock,claim_expires,started_at "
            "FROM tasks WHERE id=?", (child,),
        ).fetchone()) == before_task
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=?", (child,),
        ).fetchone()[0] == before_events
        assert conn.execute(
            "SELECT COUNT(*) FROM task_runs WHERE task_id=?", (child,),
        ).fetchone()[0] == before_runs
        assert "SECRET-REVIEW-CLAIMER" not in json.dumps(
            [dict(row) for row in conn.execute(
                "SELECT kind,payload FROM task_events WHERE task_id=?", (child,)
            )]
        )


def test_phase2_pending_decision_excludes_review_dispatch_candidate(board):
    root, child, _ = board
    spawned = []
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (child,))
        pc.open_decision(conn, _open(root, child))
        result = kb.dispatch_once(
            conn, dry_run=True,
            spawn_fn=lambda task, *_args, **_kwargs: spawned.append(task.id),
        )
        assert result.spawned == []
        assert child not in result.skipped_unassigned
        assert child not in result.skipped_nonspawnable
        assert spawned == []
        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (child,)
        ).fetchone()[0] == "review"


@pytest.mark.parametrize(
    ("status", "health_probe"),
    [
        ("ready", kb.has_spawnable_ready),
        ("review", kb.has_spawnable_review),
    ],
)
def test_phase2_pending_decision_excludes_health_probe_until_selected(
    board, monkeypatch, status, health_probe,
):
    """Telemetry must see the same decision-gated candidates as dispatch."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    root, child, other = board
    with kb.connect_closing() as conn:
        conn.execute(
            "UPDATE tasks SET status='archived' WHERE id IN (?, ?)",
            (root, other),
        )
        conn.execute("UPDATE tasks SET status=? WHERE id=?", (status, child))
        pc.open_decision(conn, _open(root, child))

        assert health_probe(conn) is False
        assert kb.dispatch_once(conn, dry_run=True).spawned == []

        pc.select_decision(conn, _select(root))

        assert health_probe(conn) is True
        assert [row[0] for row in kb.dispatch_once(conn, dry_run=True).spawned] == [child]


@pytest.mark.parametrize(
    ("status", "health_probe"),
    [
        ("ready", kb.has_spawnable_ready),
        ("review", kb.has_spawnable_review),
    ],
)
def test_phase2_health_probe_matches_dispatch_with_unaffected_legacy_task(
    board, monkeypatch, status, health_probe,
):
    """A gated task must not hide an unrelated legacy dispatch candidate."""
    from hermes_cli import profiles

    monkeypatch.setattr(profiles, "profile_exists", lambda _name: True)
    root, child, other = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='archived' WHERE id=?", (root,))
        conn.execute("UPDATE tasks SET status=? WHERE id IN (?, ?)", (status, child, other))
        pc.open_decision(conn, _open(root, child))

        result = kb.dispatch_once(conn, dry_run=True)

        assert health_probe(conn) is True
        assert [row[0] for row in result.spawned] == [other]


def test_phase2_select_releases_only_dependency_eligible_statuses(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        parent = kb.create_task(conn, title="Parent", assignee="worker")
        waiting = kb.create_task(conn, title="Waiting", assignee="worker")
        statuses = {}
        for status in ("done", "archived", "blocked", "review", "running"):
            task_id = kb.create_task(conn, title=status, assignee="worker")
            conn.execute("UPDATE tasks SET status=?, orchestration_root_id=?, orchestration_depth=1, "
                         "orchestration_parent_id=? WHERE id=?", (status, root, root, task_id))
            statuses[status] = task_id
        for task_id in (parent, waiting):
            conn.execute("UPDATE tasks SET orchestration_root_id=?, orchestration_depth=1, "
                         "orchestration_parent_id=? WHERE id=?", (root, root, task_id))
        kb.link_tasks(conn, parent, waiting)
        conn.execute("INSERT INTO task_events(task_id,kind,created_at) VALUES (?, 'blocked', 1)",
                     (statuses["blocked"],))
        affected = [child, waiting, *statuses.values()]
        pc.open_decision(conn, _open(root, child, affected_node_ids=affected))
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))

        pc.select_decision(conn, _select(root))
        rows = {row["id"]: row["status"] for row in conn.execute(
            f"SELECT id,status FROM tasks WHERE id IN ({','.join('?' for _ in affected)})", affected)}
        assert rows[child] == "ready"
        assert rows[waiting] == "todo"
        assert {status: rows[task_id] for status, task_id in statuses.items()} == {
            "done": "done", "archived": "archived", "blocked": "blocked",
            "review": "review", "running": "running",
        }


def test_phase2_select_replay_does_not_recompute_or_duplicate_events(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))
        pc.open_decision(conn, _open(root, child))
        first = pc.select_decision(conn, _select(root))
        conn.execute("UPDATE tasks SET status='todo' WHERE id=?", (child,))
        events = conn.execute("SELECT COUNT(*) FROM task_events WHERE kind='decision_checkpoint_selected'").fetchone()[0]
        replay = pc.select_decision(conn, _select(root, expected_version=999))
        assert first["deduplicated"] is False and replay["deduplicated"] is True
        assert conn.execute("SELECT status FROM tasks WHERE id=?", (child,)).fetchone()[0] == "todo"
        assert conn.execute("SELECT COUNT(*) FROM task_events WHERE kind='decision_checkpoint_selected'").fetchone()[0] == events


def test_phase2_open_and_claim_serialize_across_connections(board):
    root, child, _ = board
    path = kb.kanban_db_path()
    with kb.connect_closing() as setup:
        setup.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
    barrier = threading.Barrier(2)
    outcomes = {}

    def opener():
        with kb.connect_closing(path) as conn:
            barrier.wait()
            outcomes["open"] = pc.open_decision(conn, _open(root, child))

    def claimer():
        with kb.connect_closing(path) as conn:
            barrier.wait()
            outcomes["claim"] = kb.claim_task(conn, child, claimer="race")

    threads = [threading.Thread(target=opener), threading.Thread(target=claimer)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    with kb.connect_closing(path) as conn:
        status = conn.execute("SELECT status FROM tasks WHERE id=?", (child,)).fetchone()[0]
        assert outcomes["open"]["state"] == "pending"
        assert (outcomes["claim"] is not None and status == "running") or (
            outcomes["claim"] is None and status == "ready")


def test_phase2_open_and_review_claim_serialize_across_connections(board):
    root, child, _ = board
    path = kb.kanban_db_path()
    with kb.connect_closing(path) as setup:
        setup.execute("UPDATE tasks SET status='review' WHERE id=?", (child,))
    barrier = threading.Barrier(2)
    outcomes = {}

    def opener():
        with kb.connect_closing(path) as conn:
            barrier.wait()
            outcomes["open"] = pc.open_decision(conn, _open(root, child))

    def claimer():
        with kb.connect_closing(path) as conn:
            barrier.wait()
            outcomes["claim"] = kb.claim_review_task(
                conn, child, claimer="review-race"
            )

    threads = [threading.Thread(target=opener), threading.Thread(target=claimer)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    with kb.connect_closing(path) as conn:
        status = conn.execute(
            "SELECT status FROM tasks WHERE id=?", (child,)
        ).fetchone()[0]
        assert outcomes["open"]["state"] == "pending"
        assert (outcomes["claim"] is not None and status == "running") or (
            outcomes["claim"] is None and status == "review"
        )


def test_phase2_select_and_claim_serialize_across_connections(board):
    root, child, _ = board
    path = kb.kanban_db_path()
    with kb.connect_closing(path) as setup:
        setup.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        pc.open_decision(setup, _open(root, child))
    barrier = threading.Barrier(2)
    outcomes = {}

    def selector():
        with kb.connect_closing(path) as conn:
            barrier.wait(); outcomes["select"] = pc.select_decision(conn, _select(root))

    def claimer():
        with kb.connect_closing(path) as conn:
            barrier.wait(); outcomes["claim"] = kb.claim_task(conn, child, claimer="race")

    threads = [threading.Thread(target=selector), threading.Thread(target=claimer)]
    for thread in threads: thread.start()
    for thread in threads: thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    assert outcomes["select"]["state"] == "selected"
    # A claim that linearizes before selection is rejected; one after is valid.
    assert outcomes["claim"] is None or outcomes["claim"].status == "running"


def test_phase2_open_does_not_interrupt_running_affected_task(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='ready' WHERE id=?", (child,))
        claimed = kb.claim_task(conn, child, claimer="existing-run")
        assert claimed is not None
        before = conn.execute(
            "SELECT current_run_id,claim_lock,claim_expires FROM tasks WHERE id=?", (child,)
        ).fetchone()
        pc.open_decision(conn, _open(root, child))
        after = conn.execute(
            "SELECT status,current_run_id,claim_lock,claim_expires FROM tasks WHERE id=?", (child,)
        ).fetchone()
        assert after["status"] == "running"
        assert tuple(after[key] for key in ("current_run_id", "claim_lock", "claim_expires")) == tuple(before)
        run = conn.execute("SELECT status,ended_at FROM task_runs WHERE id=?", (before["current_run_id"],)).fetchone()
        assert tuple(run) == ("running", None)


def test_phase2_open_preserves_already_claimed_running_review(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (child,))
        claimed = kb.claim_review_task(conn, child, claimer="existing-review")
        assert claimed is not None
        before = tuple(conn.execute(
            "SELECT current_run_id,claim_lock,claim_expires FROM tasks WHERE id=?",
            (child,),
        ).fetchone())

        pc.open_decision(conn, _open(root, child))

        after = conn.execute(
            "SELECT status,current_run_id,claim_lock,claim_expires FROM tasks WHERE id=?",
            (child,),
        ).fetchone()
        assert after["status"] == "running"
        assert tuple(after[key] for key in (
            "current_run_id", "claim_lock", "claim_expires"
        )) == before
        assert tuple(conn.execute(
            "SELECT status,ended_at FROM task_runs WHERE id=?", (before[0],)
        ).fetchone()) == ("running", None)


def test_phase2_select_releases_review_task_to_review_claim_path(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (child,))
        pc.open_decision(conn, _open(root, child))

        pc.select_decision(conn, _select(root))

        assert conn.execute(
            "SELECT status FROM tasks WHERE id=?", (child,)
        ).fetchone()[0] == "review"
        claimed = kb.claim_review_task(conn, child, claimer="released-review")
        assert claimed is not None
        assert claimed.status == "running"


def test_schema_fresh_repeat_and_foreign_keys(board):
    root, child, _ = board
    kb.init_db(); kb.init_db()
    with kb.connect_closing() as conn:
        assert conn.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []
        task_cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "program_control_version" in task_cols
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert {"program_decisions", "program_decision_options",
                "program_decision_affected_nodes", "program_hints",
                "program_control_requests"} <= tables
        indexes = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='index'")}
        assert {"idx_program_decisions_node", "idx_program_hints_node_state"} <= indexes
        pc.open_decision(conn, _open(root, child))
        conn.execute("DELETE FROM tasks WHERE id=?", (root,))
        assert conn.execute("SELECT COUNT(*) FROM program_decisions").fetchone()[0] == 0


def test_request_idempotency_root_must_be_canonical_and_cascades(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO program_control_requests VALUES (?,?,?,?,?,?)",
                (child, "hint.add", "child-root", "fp", "{}", 1),
            )
        conn.execute(
            "INSERT INTO program_control_requests VALUES (?,?,?,?,?,?)",
            (root, "hint.add", "canonical-root", "fp", "{}", 1),
        )
        conn.execute("DELETE FROM tasks WHERE id=?", (root,))
        assert conn.execute("SELECT COUNT(*) FROM program_control_requests").fetchone()[0] == 0


def test_arbitrary_legacy_additive_migration_and_partial_drift_fails_closed(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(kb.SCHEMA_SQL)
    for table in ("program_decision_options", "program_decision_affected_nodes",
                  "program_hints", "program_control_requests", "program_decisions"):
        conn.execute(f"DROP TABLE {table}")
    conn.execute("DROP INDEX IF EXISTS idx_program_decisions_node")
    conn.execute("DROP INDEX IF EXISTS idx_program_affected_node")
    conn.execute("DROP INDEX IF EXISTS idx_program_hints_node_state")
    conn.execute("ALTER TABLE tasks DROP COLUMN program_control_version")
    conn.execute("INSERT INTO tasks (id,title,status,created_at) VALUES ('legacy', 'Legacy', 'done', 1)")
    conn.commit(); conn.close()
    kb.init_db(path)
    with kb.connect_closing(path) as migrated:
        assert migrated.execute("SELECT program_control_version FROM tasks WHERE id='legacy'").fetchone()[0] == 0
    drift = tmp_path / "drift.db"
    conn = sqlite3.connect(drift)
    conn.executescript(kb.SCHEMA_SQL)
    conn.execute("DROP TABLE program_hints")
    conn.execute("CREATE TABLE program_hints (root_id TEXT PRIMARY KEY)")
    conn.commit(); conn.close()
    kb._INITIALIZED_PATHS.discard(str(drift.resolve()))
    with pytest.raises(RuntimeError, match="program_hints.*schema"):
        kb.init_db(drift)


@pytest.mark.parametrize("raw", [
    b"", b"[]", b'{"a":1}{"b":2}', b'{"a":1,"a":2}', b'{"a":NaN}',
    b'{"a":Infinity}', b'{"a":true}', b"\xff", b"{} " + b"x" * (1024 * 1024),
])
def test_strict_stdin_parser_rejects_malformed_transport(raw):
    with pytest.raises(pc.ProgramControlError):
        pc.parse_request_stdin(io.BytesIO(raw))


def test_strict_stdin_parser_rejects_over_depth_before_recursive_consumers():
    depth = max(1200, __import__("sys").getrecursionlimit() + 100)
    raw = b'{"a":' + b"[" * depth + b"0" + b"]" * depth + b"}"
    with pytest.raises(pc.ProgramControlError) as exc_info:
        pc.parse_request_stdin(io.BytesIO(raw))
    assert exc_info.value.code == "invalid_request"


def test_program_mutation_over_depth_json_is_one_machine_error(monkeypatch, capsys):
    depth = max(1200, __import__("sys").getrecursionlimit() + 100)
    raw = b'{"a":' + b"[" * depth + b"0" + b"]" * depth + b"}"
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(raw), encoding="utf-8"))
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(["program", "hint", "add", "--request-json-stdin", "--json"])
    assert kc.kanban_command(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_request","ok":false}\n'


def test_open_select_hint_happy_paths_exact_shapes_and_privacy(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        opened = pc.open_decision(conn, _open(root, child))
        assert opened == {"ok": True, "checkpoint_id": "checkpoint-1", "state": "pending", "version": 1, "deduplicated": False}
        selected = pc.select_decision(conn, {"root_id": root, "checkpoint_id": "checkpoint-1",
            "option_id": "blue", "expected_version": 1, "idempotency_key": "select-1",
            "actor": "control:sergey"})
        assert selected == {"ok": True, "checkpoint_id": "checkpoint-1", "state": "selected",
                            "version": 2, "selected_option_id": "blue", "deduplicated": False}
        hinted = pc.add_hint(conn, _hint(root, child, expected_node_version=1))
        assert set(hinted) == {"ok", "hint_id", "node_id", "state", "node_version", "deduplicated"}
        assert hinted | {"hint_id": hinted["hint_id"]} == {"ok": True, "hint_id": hinted["hint_id"],
            "node_id": child, "state": "recorded", "node_version": 2, "deduplicated": False}
        events = b"\n".join(bytes(r[0] or "", "utf-8") for r in conn.execute(
            "SELECT payload FROM task_events WHERE kind LIKE 'decision_%' OR kind='hint_recorded'"))
        for secret in (b"Choose release", b"Safer", b"Use blue", b"Check the canary", b"open-1", b"hint-1"):
            assert secret not in events


def test_idempotency_precedes_cas_and_conflicts(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        first = pc.add_hint(conn, _hint(root, child))
        replay = pc.add_hint(conn, _hint(root, child, expected_node_version=999))
        assert replay == first | {"deduplicated": True}
        with pytest.raises(pc.ProgramControlError, match="idempotency_conflict"):
            pc.add_hint(conn, _hint(root, child, text="different"))
        assert conn.execute("SELECT program_control_version FROM tasks WHERE id=?", (child,)).fetchone()[0] == 1


def test_mutations_do_not_modify_caller_request_and_replay_stays_stable(board):
    root, child, _ = board
    request = _open(
        root,
        child,
        title="Cafe\u0301 rollout",
        options=[
            _open(root, child)["options"][0]
            | {"label": "Blue\u0301", "benefits": ["Cafe\u0301"]},
            _open(root, child)["options"][1],
        ],
    )
    original = copy.deepcopy(request)
    with kb.connect_closing() as conn:
        first = pc.open_decision(conn, request)
        assert request == original
        replay = pc.open_decision(conn, request)
        assert request == original
        assert replay == first | {"deduplicated": True}


def test_mutation_nests_without_committing_caller_transaction(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("BEGIN IMMEDIATE")
        conn.execute(
            "INSERT INTO task_comments(task_id,author,body,created_at) VALUES (?,?,?,?)",
            (child, "caller", "outer transaction", 1),
        )
        pc.add_hint(conn, _hint(root, child))
        assert conn.in_transaction
        conn.rollback()
        assert conn.execute(
            "SELECT COUNT(*) FROM task_comments WHERE author='caller'"
        ).fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM program_hints").fetchone()[0] == 0


@pytest.mark.parametrize("mutate,code", [
    (lambda r, root, child, other: r.update(root_id=other), "outside_root"),
    (lambda r, root, child, other: r.update(node_id=other), "outside_root"),
    (lambda r, root, child, other: r.update(expected_node_version=4), "version_conflict"),
    (lambda r, root, child, other: r.update(options=r["options"][:1]), "options"),
    (lambda r, root, child, other: r.update(recommended_option_id="missing"), "recommended"),
    (lambda r, root, child, other: r.update(extra="no"), "keys"),
])
def test_open_references_bounds_and_rollback(board, mutate, code):
    root, child, other = board
    request = _open(root, child)
    mutate(request, root, child, other)
    with kb.connect_closing() as conn, pytest.raises(pc.ProgramControlError, match=code):
        pc.open_decision(conn, request)
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM program_decisions").fetchone()[0] == 0
        assert conn.execute("SELECT program_control_version FROM tasks WHERE id=?", (child,)).fetchone()[0] == 0


def test_hint_unicode_bounds_terminal_and_independent_of_decision(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        assert pc.add_hint(conn, _hint(root, child, text="e\u0301"))["state"] == "recorded"
        stored = conn.execute("SELECT text FROM program_hints").fetchone()[0]
        assert stored == "é"
        with pytest.raises(pc.ProgramControlError, match="text"):
            pc.add_hint(conn, _hint(root, child, text="x" * 2001, idempotency_key="long"))
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (child,))
        with pytest.raises(pc.ProgramControlError, match="terminal"):
            pc.add_hint(conn, _hint(root, child, expected_node_version=1, idempotency_key="terminal"))


def test_concurrent_writers_have_one_cas_winner(board):
    root, child, _ = board
    barrier = threading.Barrier(2)
    results = []
    def writer(key):
        with kb.connect_closing() as conn:
            barrier.wait()
            try: results.append(pc.add_hint(conn, _hint(root, child, idempotency_key=key)))
            except pc.ProgramControlError as exc: results.append(exc.code)
    threads = [threading.Thread(target=writer, args=(f"key-{i}",)) for i in range(2)]
    [t.start() for t in threads]; [t.join() for t in threads]
    assert len([r for r in results if isinstance(r, dict)]) == 1
    assert "version_conflict" in results


def test_program_mutations_denied_from_slash_and_gateway_path(board):
    for command in ("program decision open --request-json-stdin --json",
                    "program decision select --request-json-stdin --json",
                    "program hint add --request-json-stdin --json"):
        output = kc.run_slash(command)
        assert "trusted direct" in output.lower()
    with kb.connect_closing() as conn:
        assert conn.execute("SELECT COUNT(*) FROM program_control_requests").fetchone()[0] == 0


def test_cli_reads_bytes_and_emits_one_compact_json_document(board, monkeypatch, capsys):
    root, child, _ = board
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(["program", "hint", "add", "--request-json-stdin", "--json"])
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(_hint(root, child)).encode())))
    assert kc.kanban_command(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == json.dumps(json.loads(captured.out), sort_keys=True, separators=(",", ":")) + "\n"


@pytest.mark.parametrize("argv", [
    ["kanban", "program", "decision"],
    ["kanban", "program", "decision", "unknown-canary-subcommand"],
    ["kanban", "program", "decision", "open"],
    ["kanban", "program", "decision", "select", "--request-json-stdin"],
    ["kanban", "program", "hint"],
    ["kanban", "program", "hint", "unknown-canary-subcommand"],
    ["kanban", "program", "hint", "add", "--json"],
    ["kanban", "program", "hint", "add", "--request-json-stdin", "--json",
     "--unknown-canary=do-not-print-me"],
])
def test_top_level_parse_failures_for_program_mutations_are_machine_only(argv, capsys):
    parser, subparsers, _ = build_top_level_parser()
    kc.build_parser(subparsers)
    with pytest.raises(SystemExit) as exc:
        parser.parse_args(argv)
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_request","ok":false}\n'


@pytest.mark.parametrize("board_arg", ["", "   "])
def test_program_mutation_empty_board_is_machine_only(board, board_arg, capsys):
    parser, subparsers, _ = build_top_level_parser()
    kc.build_parser(subparsers)
    args = parser.parse_args([
        "kanban", "--board", board_arg, "program", "hint", "add",
        "--request-json-stdin", "--json",
    ])
    assert kc.kanban_command(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_request","ok":false}\n'


def test_cli_domain_failure_is_stderr_only_and_exit_two(board, monkeypatch, capsys):
    root, child, _ = board
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(["program", "hint", "add", "--request-json-stdin", "--json"])
    invalid = _hint(root, child, text="x" * 2001)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(invalid).encode())))
    assert kc.kanban_command(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_text","ok":false}\n'


def test_open_rejects_terminal_target(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET status='done' WHERE id=?", (child,))
        with pytest.raises(pc.ProgramControlError, match="terminal"):
            pc.open_decision(conn, _open(root, child))


@pytest.mark.parametrize("policy_value", [
    None,
    "{}",
    '{"allowed_assignees":["planner","worker"]}',
    '{"orchestrator_assignees":"planner"}',
])
def test_open_requires_exact_orchestrator_policy_and_fails_closed(board, policy_value):
    root, child, _ = board
    with kb.connect_closing() as conn:
        if policy_value is not None:
            conn.execute("UPDATE tasks SET orchestration_policy=? WHERE id=?", (policy_value, root))
        else:
            conn.execute("UPDATE tasks SET orchestration_policy=NULL WHERE id=?", (root,))
        with pytest.raises(pc.ProgramControlError, match="invalid_actor"):
            pc.open_decision(conn, _open(root, child))
        assert conn.execute("SELECT COUNT(*) FROM program_decisions").fetchone()[0] == 0


def test_open_denies_ordinary_allowed_worker(board):
    root, child, _ = board
    with kb.connect_closing() as conn, pytest.raises(pc.ProgramControlError, match="invalid_actor"):
        pc.open_decision(conn, _open(root, child, actor="agent:worker"))


def test_durable_lineage_option_and_run_references_reject_direct_sql(board):
    root, child, other = board
    with kb.connect_closing() as conn:
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
                "expected_node_version,committed_node_version,state,run_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (root, "bad-node", other, "x", "control:owner", "bad-node", 0, 1,
                 "recorded", None, 1),
            )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
                "expected_node_version,committed_node_version,state,run_id,created_at) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                (root, "bad-run", child, "x", "control:owner", "bad-run", 0, 1,
                 "recorded", 999999, 1),
            )
        conn.rollback()
        conn.execute("BEGIN")
        conn.execute(
            "INSERT INTO program_decisions(root_id,checkpoint_id,node_id,state,version,title,"
            "recommended_option_id,recommendation_rationale,actor,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (root, "bad-option", child, "pending", 1, "x", "not-an-option", "x",
             "agent:planner", 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.commit()
        conn.rollback()
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_hint_run_must_belong_to_the_same_node(board):
    root, child, other = board
    with kb.connect_closing() as conn:
        child_run = conn.execute(
            "INSERT INTO task_runs(task_id,status,started_at) VALUES (?,?,?)",
            (child, "running", 1),
        ).lastrowid
        other_run = conn.execute(
            "INSERT INTO task_runs(task_id,status,started_at) VALUES (?,?,?)",
            (other, "running", 1),
        ).lastrowid
        insert = (
            "INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
            "expected_node_version,committed_node_version,state,run_id,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)"
        )
        values = [root, "same-node", child, "x", "control:owner", "same-node", 0, 1,
                  "recorded", child_run, 1]
        conn.execute(insert, values)
        for hint_id, run_id in (("cross-node", other_run), ("missing-run", 999999)):
            values[1] = values[5] = hint_id
            values[9] = run_id
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(insert, values)

        parent_index = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='index' "
            "AND name='idx_task_runs_id_task_id'"
        ).fetchone()
        assert parent_index is not None
        assert "UNIQUE INDEX" in parent_index[0]
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def _mutation_snapshot(conn, root, child):
    return (
        conn.execute("SELECT program_control_version FROM tasks WHERE id=?", (child,)).fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM program_decisions").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM program_hints").fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM task_events WHERE task_id=?", (child,)).fetchone()[0],
        conn.execute("SELECT COUNT(*) FROM program_control_requests WHERE root_id=?", (root,)).fetchone()[0],
    )


@pytest.mark.parametrize("action", ["open", "hint"])
def test_task_version_exhaustion_is_clean_and_max_minus_one_succeeds(board, action):
    root, child, _ = board
    with kb.connect_closing() as conn:
        conn.execute("UPDATE tasks SET program_control_version=? WHERE id=?",
                     (pc.MAX_SAFE_INTEGER, child))
        before = _mutation_snapshot(conn, root, child)
        request = (_open(root, child, expected_node_version=pc.MAX_SAFE_INTEGER)
                   if action == "open" else
                   _hint(root, child, expected_node_version=pc.MAX_SAFE_INTEGER))
        operation = pc.open_decision if action == "open" else pc.add_hint
        with pytest.raises(pc.ProgramControlError, match="version_exhausted"):
            operation(conn, request)
        assert _mutation_snapshot(conn, root, child) == before
        conn.execute("UPDATE tasks SET program_control_version=? WHERE id=?",
                     (pc.MAX_SAFE_INTEGER - 1, child))
        request["expected_node_version"] = pc.MAX_SAFE_INTEGER - 1
        assert operation(conn, request)["ok"] is True
        assert conn.execute("SELECT program_control_version FROM tasks WHERE id=?",
                            (child,)).fetchone()[0] == pc.MAX_SAFE_INTEGER


def test_decision_version_exhaustion_is_clean_and_max_minus_one_succeeds(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        pc.open_decision(conn, _open(root, child))
        conn.execute("UPDATE program_decisions SET version=?", (pc.MAX_SAFE_INTEGER,))
        before = _mutation_snapshot(conn, root, child)
        request = {"root_id": root, "checkpoint_id": "checkpoint-1", "option_id": "blue",
                   "expected_version": pc.MAX_SAFE_INTEGER, "idempotency_key": "select-max",
                   "actor": "control:owner"}
        with pytest.raises(pc.ProgramControlError, match="version_exhausted"):
            pc.select_decision(conn, request)
        assert _mutation_snapshot(conn, root, child) == before
        conn.execute("UPDATE program_decisions SET version=?", (pc.MAX_SAFE_INTEGER - 1,))
        request["expected_version"] = pc.MAX_SAFE_INTEGER - 1
        assert pc.select_decision(conn, request)["version"] == pc.MAX_SAFE_INTEGER


def test_unknown_board_program_mutation_is_one_machine_error_without_creation(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(["--board", "missing", "program", "hint", "add",
                              "--request-json-stdin", "--json"])
    assert kc.kanban_command(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"unknown_board","ok":false}\n'
    assert not (home / "kanban" / "boards" / "missing").exists()


@pytest.mark.parametrize(
    "argv",
    [
        ["program", "decision", "open", "--request-json-stdin", "--json"],
        ["program", "decision", "select", "--request-json-stdin", "--json"],
        ["program", "hint", "add", "--request-json-stdin", "--json"],
    ],
)
def test_program_mutation_init_exception_is_one_database_error(
    argv, monkeypatch, capsys
):
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(argv)
    monkeypatch.setattr(kb, "init_db", lambda: (_ for _ in ()).throw(
        RuntimeError("SECRET-CANARY /private/board.db")
    ))

    assert kc.kanban_command(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"database_error","ok":false}\n'
    assert "SECRET-CANARY" not in captured.err


def test_program_mutation_corrupt_db_is_one_database_error(tmp_path, monkeypatch, capsys):
    path = tmp_path / "SECRET-CANARY-board.db"
    path.write_bytes(b"not a sqlite database SECRET-CANARY")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(path))
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(
        ["program", "hint", "add", "--request-json-stdin", "--json"]
    )

    assert kc.kanban_command(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"database_error","ok":false}\n'
    assert "SECRET-CANARY" not in captured.err


@pytest.mark.parametrize(
    ("stage", "failure"),
    [
        ("connect", RuntimeError(
            "snapshot instability SECRET-CANARY /private/snapshot.db"
        )),
        ("connect", kb.KanbanDbCorruptError(
            Path("/private/SECRET-CANARY.db"), None, "corrupt"
        )),
        ("operation", Exception("arbitrary SECRET-CANARY /private/canary.db")),
    ],
)
def test_program_machine_handler_normalizes_every_non_domain_exception(
    stage, failure, monkeypatch, capsys
):
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(
        ["program", "hint", "add", "--request-json-stdin", "--json"]
    )
    monkeypatch.setattr(pc, "parse_request_stdin", lambda _stream: {})
    if stage == "connect":
        monkeypatch.setattr(kb, "connect_closing", lambda: (_ for _ in ()).throw(failure))
    else:
        @contextlib.contextmanager
        def connection():
            yield object()

        monkeypatch.setattr(kb, "connect_closing", connection)
        monkeypatch.setattr(pc, "add_hint", lambda _conn, _request: (_ for _ in ()).throw(failure))

    assert kc._cmd_program(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"database_error","ok":false}\n'
    assert "SECRET-CANARY" not in captured.err


def test_program_machine_handler_normalizes_schema_drift(monkeypatch, capsys):
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(
        ["program", "hint", "add", "--request-json-stdin", "--json"]
    )
    monkeypatch.setattr(pc, "parse_request_stdin", lambda _stream: {})
    monkeypatch.setattr(kb, "connect_closing", lambda: (_ for _ in ()).throw(
        RuntimeError("schema drift SECRET-CANARY /private/schema.db")
    ))

    assert kc._cmd_program(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"database_error","ok":false}\n'
