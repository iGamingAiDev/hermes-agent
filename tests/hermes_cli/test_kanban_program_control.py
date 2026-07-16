from __future__ import annotations

import argparse
import contextlib
import copy
import io
import json
import hashlib
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
    public_options = [
        {"option_id": "blue", "ordinal": 0, "label": "Blue release",
         "summary": "Run the blue environment before cutover",
         "benefits": ["Lower cutover risk"], "risks": ["Temporary duplicate capacity"],
         "reversibility": "reversible", "security_impact": "No material change",
         "cost_impact": "Temporary capacity increase", "operations_impact": "Operate both environments"},
        {"option_id": "green", "ordinal": 1, "label": "Green release",
         "summary": "Cut over directly to the green environment",
         "benefits": ["Simpler execution"], "risks": ["Higher cutover risk"],
         "reversibility": "partially_reversible", "security_impact": "No material change",
         "cost_impact": "No material change", "operations_impact": "Single cutover window"},
    ]
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
        "public_brief": {
            "title": "Release strategy decision",
            "recommendation_rationale": "Prefer the lower-risk staged release",
            "recommended_option_id": "blue", "options": public_options,
        },
        "affected_node_ids": [child], "expected_node_version": 0,
        "idempotency_key": "open-1", "actor": "agent:planner",
    }
    request.update(updates)
    return request


def test_a4_open_decision_atomically_persists_digest_bound_public_brief(board):
    root, child, _ = board
    request = _open(root, child)
    with kb.connect_closing() as conn:
        result = pc.open_decision(conn, request)
        brief = dict(conn.execute(
            "SELECT * FROM program_decision_public_briefs WHERE root_id=? AND checkpoint_id=?",
            (root, "checkpoint-1"),
        ).fetchone())
        options = [dict(row) for row in conn.execute(
            "SELECT * FROM program_decision_public_options WHERE root_id=? AND checkpoint_id=? ORDER BY ordinal",
            (root, "checkpoint-1"),
        )]

        canonical = {
        "kind": "hermes_program_decision_public_brief",
        "schema_version": 1, "scanner_policy_version": 1,
        "classification": "operator_visible",
        "title": request["public_brief"]["title"],
        "recommendation_rationale": request["public_brief"]["recommendation_rationale"],
        "recommended_option_id": request["public_brief"]["recommended_option_id"],
        "options": request["public_brief"]["options"],
    }
    expected_digest = hashlib.sha256(json.dumps(
        canonical, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")).hexdigest()
    assert result == {"ok": True, "checkpoint_id": "checkpoint-1", "state": "pending",
                      "version": 1, "deduplicated": False}
    assert brief["schema_version"] == 1
    assert brief["classification"] == "operator_visible"
    assert brief["content_digest"] == expected_digest
    assert brief["generated_at"] == brief["created_at"]
    assert [(row["option_id"], row["ordinal"]) for row in options] == [("blue", 0), ("green", 1)]


@pytest.mark.parametrize("forbidden", [
    "https://example.invalid/x", "urn:private:value", "mailto:owner", "owner@example.invalid",
    "/var/lib/hermes", r"C:\\Users\\operator", "~/private", "-----BEGIN PRIVATE KEY-----",
    "Bearer abcdefghijklmnop", "Basic abcdefghijklmnop", "api_key=abcdefghijklmnop",
    "password: abcdefghijklmnop", "secret token abcdefghijklmnop",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxIn0.abcdefghijklmnopqrstuvwxyz123456",
    "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv",
])
def test_a4_open_rejects_public_disclosure_classes_without_mutation(board, forbidden):
    root, child, _ = board
    request = _open(root, child)
    request["public_brief"]["options"][0]["summary"] = forbidden
    with kb.connect_closing() as conn:
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.open_decision(conn, request)
        assert exc_info.value.code == "public_brief_disclosure"
        assert forbidden not in str(exc_info.value)
        assert _mutation_snapshot(conn, root, child) == before
        assert conn.execute("SELECT COUNT(*) FROM program_decision_public_briefs").fetchone()[0] == 0


@pytest.mark.parametrize("invalid", ["Cafe\u0301", "line\nnext", "x\u202ey", "x\u2066y",
                                            "x\ue000y", "x\ud800y", "x\x00y", "x\x1fy"])
def test_a4_public_strings_require_safe_canonical_unicode(board, invalid):
    root, child, _ = board
    request = _open(root, child)
    request["public_brief"]["title"] = invalid
    with kb.connect_closing() as conn, pytest.raises(pc.ProgramControlError) as exc_info:
        pc.open_decision(conn, request)
    assert exc_info.value.code == "invalid_public_brief_text"


def test_a4_select_fails_closed_when_public_brief_is_tampered(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        pc.open_decision(conn, _open(root, child))
        conn.execute("UPDATE program_decision_public_options SET summary='tampered' WHERE option_id='blue'")
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.select_decision(conn, _select(root))
        assert exc_info.value.code == "invalid_public_brief"
        assert _mutation_snapshot(conn, root, child) == before


def test_a4_capability_read_only_rejects_corrupt_pending_public_brief(board):
    root, child, _ = board
    path = kb.kanban_db_path()
    with kb.connect_closing() as conn:
        pc.open_decision(conn, _open(root, child))
        conn.execute("UPDATE program_decision_public_briefs SET content_digest=?", ("0" * 64,))
    schema_before = sqlite3.connect(path).execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
    ).fetchall()
    images_before = {candidate.name: candidate.read_bytes() for candidate in
                     (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")) if candidate.exists()}
    with pytest.raises(RuntimeError, match="^public_brief_unavailable$"):
        kb.validate_current_program_control_schema_read_only(path)
    schema_after = sqlite3.connect(path).execute(
        "SELECT type,name,tbl_name,sql FROM sqlite_master ORDER BY type,name"
    ).fetchall()
    images_after = {candidate.name: candidate.read_bytes() for candidate in
                    (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")) if candidate.exists()}
    assert schema_after == schema_before
    assert images_after == images_before


def test_a4_open_replay_requires_untampered_durable_public_brief(board):
    root, child, _ = board
    request = _open(root, child)
    with kb.connect_closing() as conn:
        pc.open_decision(conn, request)
        conn.execute("DELETE FROM program_decision_public_options WHERE option_id='green'")
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.open_decision(conn, request)
        assert exc_info.value.code == "public_brief_unavailable"
        assert _mutation_snapshot(conn, root, child) == before


def test_a4_select_replay_requires_selected_state_and_untampered_brief(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        pc.open_decision(conn, _open(root, child))
        request = _select(root)
        pc.select_decision(conn, request)
        conn.execute("UPDATE program_decision_public_options SET summary='tampered' WHERE option_id='blue'")
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.select_decision(conn, request)
        assert exc_info.value.code == "public_brief_unavailable"
        assert _mutation_snapshot(conn, root, child) == before


def test_a4_select_is_owner_only_before_mutation(board):
    root, child, _ = board
    with kb.connect_closing() as conn:
        pc.open_decision(conn, _open(root, child))
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.select_decision(conn, _select(root, actor="control:sergey"))
        assert exc_info.value.code in {"invalid_actor", "forbidden"}
        assert _mutation_snapshot(conn, root, child) == before
        assert pc.select_decision(conn, _select(root))["state"] == "selected"


@pytest.mark.parametrize("invalid", ["x\u200by", "x\u200dy", "x\u2060y", "x\ufdd0y", "x\U0010ffffy"])
def test_a4_public_strings_reject_all_format_controls_and_noncharacters(board, invalid):
    root, child, _ = board
    request = _open(root, child)
    request["public_brief"]["title"] = invalid
    with kb.connect_closing() as conn, pytest.raises(pc.ProgramControlError) as exc_info:
        pc.open_decision(conn, request)
    assert exc_info.value.code == "invalid_public_brief_text"


def test_a4_scans_concatenated_canonical_public_envelope(board):
    root, child, _ = board
    request = _open(root, child)
    request["public_brief"]["options"][0]["benefits"] = ["Bearer"]
    request["public_brief"]["options"][0]["risks"] = ["abcdefghijklmnop"]
    with kb.connect_closing() as conn, pytest.raises(pc.ProgramControlError) as exc_info:
        pc.open_decision(conn, request)
    assert exc_info.value.code == "public_brief_disclosure"


@pytest.mark.parametrize("field,value", [
    ("checkpoint_id", "eyJabcdefgh.eyJijklmnop.abcdefghijklmnopqrstuvwxyz123456"),
    ("option_id", "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuv0123456789"),
])
def test_a4_metadata_visible_identifiers_use_disclosure_gate(board, field, value):
    root, child, _ = board
    request = _open(root, child)
    if field == "checkpoint_id":
        request[field] = value
    else:
        request["options"][0]["option_id"] = value
        request["recommended_option_id"] = value
        request["public_brief"]["options"][0]["option_id"] = value
        request["public_brief"]["recommended_option_id"] = value
    with kb.connect_closing() as conn:
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.open_decision(conn, request)
        assert exc_info.value.code == "public_brief_disclosure"
        assert value not in str(exc_info.value)
        assert _mutation_snapshot(conn, root, child) == before


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
    conn.execute("ALTER TABLE tasks DROP COLUMN program_control_version")
    conn.execute("INSERT INTO tasks (id,title,status,created_at) VALUES ('legacy', 'Legacy', 'done', 1)")
    conn.commit(); conn.close()
    kb.init_db(path)
    with kb.connect_closing(path) as migrated:
        assert migrated.execute("SELECT program_control_version FROM tasks WHERE id='legacy'").fetchone()[0] == 0
    drift = tmp_path / "drift.db"
    conn = sqlite3.connect(drift)
    conn.row_factory = sqlite3.Row
    conn.executescript(kb.SCHEMA_SQL)
    kb._create_program_control_schema(conn)
    conn.execute("DROP TABLE program_hints")
    conn.execute("CREATE TABLE program_hints (root_id TEXT PRIMARY KEY)")
    conn.commit(); conn.close()
    kb._INITIALIZED_PATHS.discard(str(drift.resolve()))
    with pytest.raises(RuntimeError, match="program_hints.*schema"):
        kb.init_db(drift)


def test_partial_a4_public_schema_never_self_repairs(tmp_path):
    path = tmp_path / "partial-a4.db"
    kb.init_db(path)
    with kb.connect_closing(path) as conn:
        conn.execute("DROP TABLE program_decision_public_options")
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))

    with pytest.raises(
        RuntimeError,
        match="^program-control public schema is incomplete or incompatible$",
    ):
        kb.init_db(path)

    with sqlite3.connect(path) as conn:
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert "program_decision_public_briefs" in tables
    assert "program_decision_public_options" not in tables


def test_exact_a3_phase2_hints_migrate_before_a4_validation(tmp_path):
    path = tmp_path / "a3-phase2-hints.db"
    kb.init_db(path)
    with kb.connect_closing(path) as conn:
        conn.execute("DROP INDEX idx_program_hints_node_state")
        conn.execute("DROP TABLE program_hints")
        conn.executescript(
            """
            CREATE TABLE program_hints (
                root_id TEXT NOT NULL,
                hint_id TEXT NOT NULL, node_id TEXT NOT NULL,
                text TEXT NOT NULL, actor TEXT NOT NULL, idempotency_key TEXT NOT NULL,
                expected_node_version INTEGER NOT NULL,
                committed_node_version INTEGER NOT NULL,
                state TEXT NOT NULL CHECK (state IN
                  ('recorded','seen','incorporated','deferred','rejected','reconcile')),
                run_id INTEGER, claim_lock TEXT, profile TEXT, created_at INTEGER NOT NULL,
                delivered_at INTEGER, terminal_at INTEGER, terminal_reason_code TEXT,
                PRIMARY KEY (root_id, hint_id), UNIQUE (root_id, idempotency_key),
                FOREIGN KEY (root_id, root_id)
                  REFERENCES tasks(id, orchestration_root_id) ON DELETE CASCADE,
                FOREIGN KEY (node_id, root_id)
                  REFERENCES tasks(id, orchestration_root_id) ON DELETE CASCADE,
                FOREIGN KEY (run_id, node_id) REFERENCES task_runs(id, task_id)
            );
            CREATE INDEX idx_program_hints_node_state
              ON program_hints(root_id, node_id, state, created_at);
            """
        )
        conn.execute("DROP TABLE program_decision_public_options")
        conn.execute("DROP TABLE program_decision_public_briefs")
        conn.execute("DROP INDEX idx_program_decision_options_ordinal")
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))

    kb.init_db(path)

    with kb.connect_closing(path) as conn:
        fresh = sqlite3.connect(":memory:")
        fresh.row_factory = sqlite3.Row
        fresh.executescript(
            "CREATE TABLE tasks(id TEXT PRIMARY KEY, orchestration_root_id TEXT);"
            "CREATE TABLE task_runs(id INTEGER PRIMARY KEY, task_id TEXT NOT NULL);"
        )
        kb._create_program_control_schema(fresh)
        assert kb._program_control_shape(conn, "program_hints") == kb._program_control_shape(
            fresh, "program_hints"
        )
        fresh.close()
        tables = {
            row[0]
            for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {"program_decision_public_briefs", "program_decision_public_options"} <= tables
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_exact_a3_predecessor_migrates_additively_to_a4(tmp_path):
    path = tmp_path / "a3.db"
    kb.init_db(path)
    with kb.connect_closing(path) as conn:
        conn.execute(
            "INSERT INTO tasks(id,title,status,created_at,orchestration_root_id) "
            "VALUES ('legacy-root','Legacy','done',1,'legacy-root')"
        )
        conn.commit()
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute(
            "INSERT INTO program_decisions VALUES "
            "('legacy-root','selected-history','legacy-root','selected',2,'Raw title','a',"
            "'Raw rationale','agent:planner','a',1,2)"
        )
        for ordinal, option_id in enumerate(("a", "b")):
            conn.execute(
                "INSERT INTO program_decision_options VALUES "
                "(?,?,?,?,?,?,?,?,?,?,?,?)",
                ("legacy-root", "selected-history", option_id, ordinal, "Raw", "Raw",
                 '["raw"]', '["raw"]', "reversible", "raw", "raw", "raw"),
            )
        conn.execute("DROP TABLE program_decision_public_options")
        conn.execute("DROP TABLE program_decision_public_briefs")
        conn.execute("DROP INDEX idx_program_decision_options_ordinal")
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))

    kb.init_db(path)

    with kb.connect_closing(path) as conn:
        tables = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )}
        assert {"program_decision_public_briefs", "program_decision_public_options"} <= tables
        assert conn.execute("SELECT COUNT(*) FROM program_decisions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM program_decision_public_briefs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM program_decision_public_options").fetchone()[0] == 0
        assert conn.execute("PRAGMA foreign_key_check").fetchall() == []


def test_exact_a3_pending_migration_refuses_before_any_a4_ddl(tmp_path):
    path = tmp_path / "a3-pending.db"
    kb.init_db(path)
    with kb.connect_closing(path) as conn:
        conn.execute("PRAGMA foreign_keys=OFF")
        conn.execute("INSERT INTO tasks(id,title,status,created_at,orchestration_root_id) "
                     "VALUES ('root','Root','todo',1,'root')")
        conn.execute("INSERT INTO program_decisions VALUES "
                     "('root','pending','root','pending',1,'Raw title','a','Raw rationale',"
                     "'agent:planner',NULL,1,NULL)")
        conn.execute("INSERT INTO program_decision_options VALUES "
                     "('root','pending','a',0,'A','Raw A','[\"x\"]','[\"y\"]','reversible','x','x','x')")
        conn.execute("INSERT INTO program_decision_options VALUES "
                     "('root','pending','b',1,'B','Raw B','[\"x\"]','[\"y\"]','reversible','x','x','x')")
        conn.execute("DROP TABLE program_decision_public_options")
        conn.execute("DROP TABLE program_decision_public_briefs")
        conn.execute("DROP INDEX idx_program_decision_options_ordinal")
        before = (conn.execute("SELECT group_concat(sql,'\n') FROM sqlite_master").fetchone()[0],
                  [tuple(row) for row in conn.execute("SELECT * FROM program_decisions")],
                  [tuple(row) for row in conn.execute(
                      "SELECT * FROM program_decision_options ORDER BY ordinal")])
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))
    with pytest.raises(RuntimeError, match="^legacy_pending_public_brief_required$"):
        kb.init_db(path)
    raw = sqlite3.connect(path)
    try:
        after = (raw.execute("SELECT group_concat(sql,'\n') FROM sqlite_master").fetchone()[0],
                 raw.execute("SELECT * FROM program_decisions").fetchall(),
                 raw.execute("SELECT * FROM program_decision_options ORDER BY ordinal").fetchall())
        assert after == before
        tables = {row[0] for row in raw.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        assert "program_decision_public_briefs" not in tables
        assert "program_decision_public_options" not in tables
        assert raw.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        raw.close()


def test_exact_a3_migration_serializes_pending_writer_with_a4_ddl(tmp_path, monkeypatch):
    path = tmp_path / "a3-race.db"
    kb.init_db(path)
    with kb.connect_closing(path) as conn:
        conn.execute("INSERT INTO tasks(id,title,status,created_at,orchestration_root_id) "
                     "VALUES ('root','Root','todo',1,'root')")
        conn.execute("DROP TABLE program_decision_public_options")
        conn.execute("DROP TABLE program_decision_public_briefs")
        conn.execute("DROP INDEX idx_program_decision_options_ordinal")
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))

    validation_done = threading.Event()
    writer_started = threading.Event()
    original_validate = kb._validate_existing_program_control_schema

    def coordinated_validate(conn):
        result = original_validate(conn)
        if conn.execute("PRAGMA database_list").fetchone()[2] == str(path):
            validation_done.set()
            assert writer_started.wait(5)
        return result

    monkeypatch.setattr(kb, "_validate_existing_program_control_schema", coordinated_validate)
    outcomes = {}

    def migrate():
        try:
            kb.init_db(path)
            outcomes["migration"] = "committed"
        except RuntimeError as exc:
            outcomes["migration"] = str(exc)

    def write_pending():
        assert validation_done.wait(5)
        conn = sqlite3.connect(path, timeout=5, isolation_level=None)
        try:
            writer_started.set()
            conn.execute("BEGIN IMMEDIATE")
            if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' "
                            "AND name='program_decision_public_briefs'").fetchone():
                conn.rollback()
                outcomes["writer"] = "a4_won"
                return
            conn.execute("INSERT INTO program_decisions VALUES "
                         "('root','pending','root','pending',1,'Raw title','a','Raw rationale',"
                         "'agent:planner',NULL,1,NULL)")
            for ordinal, option_id in enumerate(("a", "b")):
                conn.execute("INSERT INTO program_decision_options VALUES "
                             "(?,?,?,?,?,?,?,?,?,?,?,?)",
                             ("root", "pending", option_id, ordinal, option_id, "Raw",
                              '["x"]', '["y"]', "reversible", "x", "x", "x"))
            conn.commit()
            outcomes["writer"] = "committed"
        finally:
            conn.close()

    migration_thread = threading.Thread(target=migrate)
    writer_thread = threading.Thread(target=write_pending)
    migration_thread.start(); writer_thread.start()
    migration_thread.join(10); writer_thread.join(10)
    assert not migration_thread.is_alive() and not writer_thread.is_alive()
    assert outcomes in (
        {"writer": "committed", "migration": "legacy_pending_public_brief_required"},
        {"migration": "committed", "writer": "a4_won"},
    )


def test_exact_a3_migration_ddl_failure_rolls_back_all_a4_objects(tmp_path, monkeypatch):
    path = tmp_path / "a3-ddl-failure.db"
    kb.init_db(path)
    with kb.connect_closing(path) as conn:
        conn.execute("DROP TABLE program_decision_public_options")
        conn.execute("DROP TABLE program_decision_public_briefs")
        conn.execute("DROP INDEX idx_program_decision_options_ordinal")
    kb._INITIALIZED_PATHS.discard(str(path.resolve()))

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql, parameters=()):
            if "CREATE TABLE IF NOT EXISTS program_decision_public_options" in sql:
                raise sqlite3.OperationalError("injected A4 DDL failure")
            return super().execute(sql, parameters)

    def failing_connect(db_path, *, configure_busy_timeout=True):
        conn = sqlite3.connect(str(db_path), isolation_level=None, timeout=5,
                               factory=FailingConnection)
        if configure_busy_timeout:
            conn.execute("PRAGMA busy_timeout=5000")
        return conn

    monkeypatch.setattr(kb, "_sqlite_connect", failing_connect)
    with pytest.raises(sqlite3.OperationalError, match="injected A4 DDL failure"):
        kb.init_db(path)
    with sqlite3.connect(path) as conn:
        objects = {row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE name IN "
            "('program_decision_public_briefs','program_decision_public_options',"
            "'idx_program_decision_options_ordinal')"
        )}
    assert objects == set()


@pytest.mark.parametrize("mutate", [
    lambda request: request.pop("public_brief"),
    lambda request: request["public_brief"].update(recommended_option_id="green"),
    lambda request: request["public_brief"]["options"].reverse(),
    lambda request: request["public_brief"]["options"].pop(),
    lambda request: request["public_brief"]["options"].append(
        request["public_brief"]["options"][0] | {"option_id": "extra", "ordinal": 2}
    ),
])
def test_a4_invalid_or_incomplete_public_brief_rolls_back_every_surface(board, mutate):
    root, child, _ = board
    request = _open(root, child)
    mutate(request)
    with kb.connect_closing() as conn:
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError):
            pc.open_decision(conn, request)
        assert _mutation_snapshot(conn, root, child) == before
        assert conn.execute("SELECT COUNT(*) FROM program_decision_public_briefs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM program_decision_public_options").fetchone()[0] == 0


@pytest.mark.parametrize(("ordinal", "index"), [(False, 0), (True, 1)])
def test_a4_open_decision_rejects_bool_public_ordinal_without_mutation(board, ordinal, index):
    root, child, _ = board
    request = _open(root, child)
    request["public_brief"]["options"][index]["ordinal"] = ordinal
    with kb.connect_closing() as conn:
        before = _mutation_snapshot(conn, root, child)
        tables = ("program_decisions", "program_decision_options",
                  "program_decision_public_briefs", "program_decision_public_options",
                  "task_events", "program_control_requests")
        counts = {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                  for table in tables}
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.open_decision(conn, request)
        assert exc_info.value.code == "invalid_public_brief"
        assert _mutation_snapshot(conn, root, child) == before
        assert {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                for table in tables} == counts


def test_a4_idempotency_binds_public_brief(board):
    root, child, _ = board
    request = _open(root, child)
    with kb.connect_closing() as conn:
        first = pc.open_decision(conn, request)
        assert pc.open_decision(conn, request) == first | {"deduplicated": True}
        changed = copy.deepcopy(request)
        changed["public_brief"]["title"] = "Changed operator title"
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.open_decision(conn, changed)
        assert exc_info.value.code == "idempotency_conflict"


@pytest.mark.parametrize("tamper", [
    lambda conn: conn.execute("DELETE FROM program_decision_public_briefs"),
    lambda conn: conn.execute("DELETE FROM program_decision_public_options WHERE option_id='blue'"),
    lambda conn: conn.execute("UPDATE program_decision_public_briefs SET content_digest=?", ("0" * 64,)),
])
def test_a4_select_requires_complete_digest_bound_brief(board, tamper):
    root, child, _ = board
    with kb.connect_closing() as conn:
        pc.open_decision(conn, _open(root, child))
        tamper(conn)
        before = _mutation_snapshot(conn, root, child)
        with pytest.raises(pc.ProgramControlError) as exc_info:
            pc.select_decision(conn, _select(root))
        assert exc_info.value.code == "invalid_public_brief"
        assert _mutation_snapshot(conn, root, child) == before


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
            "actor": "control:owner"})
        assert selected == {"ok": True, "checkpoint_id": "checkpoint-1", "state": "selected",
                            "version": 2, "selected_option_id": "blue", "deduplicated": False}
        hinted = pc.add_hint(conn, _hint(root, child, expected_node_version=1))
        assert set(hinted) == {"ok", "hint_id", "node_id", "state", "node_version", "deduplicated"}
        assert hinted | {"hint_id": hinted["hint_id"]} == {"ok": True, "hint_id": hinted["hint_id"],
            "node_id": child, "state": "recorded", "node_version": 2, "deduplicated": False}
        events = b"\n".join(bytes(r[0] or "", "utf-8") for r in conn.execute(
            "SELECT payload FROM task_events WHERE kind LIKE 'decision_%' OR kind='hint_recorded'"))
        for secret in (b"Choose release", b"Safer", b"Use blue", b"Check the canary", b"open-1", b"hint-1",
                       b"Release strategy decision", b"Prefer the lower-risk", b"Lower cutover risk"):
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


def test_cli_select_denies_sergey_and_owner_succeeds(board, monkeypatch, capsys):
    root, child, _ = board
    with kb.connect_closing() as conn:
        pc.open_decision(conn, _open(root, child))
        before = _mutation_snapshot(conn, root, child)
    parser = kc.build_parser(argparse.ArgumentParser().add_subparsers())
    args = parser.parse_args(["program", "decision", "select", "--request-json-stdin", "--json"])
    denied = _select(root, actor="control:sergey")
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(denied).encode())))
    assert kc.kanban_command(args) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_actor","ok":false}\n'
    with kb.connect_closing() as conn:
        assert _mutation_snapshot(conn, root, child) == before
    allowed = _select(root)
    monkeypatch.setattr("sys.stdin", io.TextIOWrapper(io.BytesIO(json.dumps(allowed).encode())))
    assert kc.kanban_command(args) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "selected"


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
            "expected_node_version,committed_node_version,state,run_id,claim_lock,profile,"
            "created_at,delivered_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)"
        )
        values = [root, "same-node", child, "x", "control:owner", "same-node", 0, 1,
                  "seen", child_run, "lock", "worker", 1, 1]
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
