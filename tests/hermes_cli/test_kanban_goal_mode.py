"""Tests for kanban goal_mode — per-card Ralph-style goal loop.

Covers three layers:

1. DB: goal_mode / goal_max_turns persist through create_task + from_row,
   and a legacy DB (without the columns) migrates cleanly.
2. Spawn: _default_spawn sets the HERMES_KANBAN_GOAL_MODE env vars only
   when the card opts in.
3. Loop: goals.run_kanban_goal_loop continuation / completion / budget
   behaviour, driven entirely through injected callbacks (no live model).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import goals


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


# ---------------------------------------------------------------------------
# DB layer
# ---------------------------------------------------------------------------

def test_goal_mode_defaults_off(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain task", assignee="worker")
        task = kb.get_task(conn, tid)
    assert task.goal_mode is False
    assert task.goal_max_turns is None


def test_goal_mode_persists(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="open-ended task",
            assignee="worker",
            goal_mode=True,
            goal_max_turns=7,
        )
        task = kb.get_task(conn, tid)
    assert task.goal_mode is True
    assert task.goal_max_turns == 7


def test_goal_mode_without_max_turns(kanban_home):
    with kb.connect() as conn:
        tid = kb.create_task(
            conn, title="t", assignee="worker", goal_mode=True
        )
        task = kb.get_task(conn, tid)
    assert task.goal_mode is True
    assert task.goal_max_turns is None


def test_legacy_db_migrates_goal_columns(tmp_path, monkeypatch):
    """A tasks table created without goal columns must gain them on init."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)

    db_path = kb.kanban_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    # Minimal legacy schema: tasks table missing goal_mode / goal_max_turns.
    legacy = sqlite3.connect(db_path)
    legacy.execute(
        """
        CREATE TABLE tasks (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            body TEXT,
            assignee TEXT,
            status TEXT NOT NULL DEFAULT 'ready',
            priority INTEGER NOT NULL DEFAULT 0,
            created_by TEXT,
            created_at INTEGER NOT NULL,
            started_at INTEGER,
            completed_at INTEGER,
            workspace_kind TEXT NOT NULL DEFAULT 'scratch',
            workspace_path TEXT,
            claim_lock TEXT,
            claim_expires INTEGER
        )
        """
    )
    legacy.execute(
        "INSERT INTO tasks (id, title, status, priority, created_at, workspace_kind) "
        "VALUES ('legacy1', 'old', 'ready', 0, 1, 'scratch')"
    )
    legacy.commit()
    legacy.close()

    # init_db runs the additive migration.
    kb.init_db()
    with kb.connect() as conn:
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(tasks)")}
        assert "goal_mode" in cols
        assert "goal_max_turns" in cols
        task = kb.get_task(conn, "legacy1")
    # Existing row keeps the safe default.
    assert task.goal_mode is False
    assert task.goal_max_turns is None


# ---------------------------------------------------------------------------
# Spawn env
# ---------------------------------------------------------------------------

def test_spawn_sets_goal_env_only_when_enabled(kanban_home, monkeypatch):
    captured = {}

    class _FakeProc:
        pid = 4242

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(
            conn,
            title="goal task",
            assignee="default",
            goal_mode=True,
            goal_max_turns=5,
        )
        task = kb.get_task(conn, tid)

    kb._default_spawn(task, str(kanban_home))
    env = captured["env"]
    assert env.get("HERMES_KANBAN_GOAL_MODE") == "1"
    assert env.get("HERMES_KANBAN_GOAL_MAX_TURNS") == "5"


def test_spawn_no_goal_env_for_plain_task(kanban_home, monkeypatch):
    captured = {}

    class _FakeProc:
        pid = 4243

    def _fake_popen(cmd, **kwargs):
        captured["env"] = kwargs.get("env", {})
        return _FakeProc()

    monkeypatch.setattr("subprocess.Popen", _fake_popen)

    with kb.connect() as conn:
        tid = kb.create_task(conn, title="plain", assignee="default")
        task = kb.get_task(conn, tid)

    kb._default_spawn(task, str(kanban_home))
    env = captured["env"]
    assert "HERMES_KANBAN_GOAL_MODE" not in env
    assert "HERMES_KANBAN_GOAL_MAX_TURNS" not in env


# ---------------------------------------------------------------------------
# Goal loop logic (callback-injected, no live model)
# ---------------------------------------------------------------------------

def _patch_judge(monkeypatch, verdicts):
    """Make judge_goal return a scripted sequence of verdicts."""
    seq = list(verdicts)

    def _fake_judge(goal, response, subgoals=None, background_processes=None, **_kw):
        v = seq.pop(0) if seq else "done"
        # 4-tuple contract: (verdict, reason, parse_failed, wait_directive)
        return v, f"scripted:{v}", False, None

    monkeypatch.setattr(goals, "judge_goal", _fake_judge)


def test_loop_stops_when_worker_already_completed(monkeypatch):
    # Worker called kanban_complete on its first turn — no judging needed.
    _patch_judge(monkeypatch, ["continue"])  # should never be consulted
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t1",
        goal_text="do the thing",
        run_turn=lambda p: turns.append(p) or "x",
        task_status_fn=lambda: "done",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="done already",
    )
    assert res["outcome"] == "completed_by_worker"
    assert turns == []  # no extra turns


def test_loop_continues_then_worker_completes(monkeypatch):
    _patch_judge(monkeypatch, ["continue", "continue"])
    statuses = iter(["running", "running", "done"])
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t2",
        goal_text="ship feature",
        run_turn=lambda p: turns.append(p) or f"turn{len(turns)}",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="started",
    )
    assert res["outcome"] == "completed_by_worker"
    # Two continuation turns fed before the worker completed.
    assert len(turns) == 2
    assert all("not done yet" in p for p in turns)


def test_loop_merges_hints_into_existing_turn_without_extra_turn(monkeypatch):
    _patch_judge(monkeypatch, ["continue"])
    statuses = iter(["running", "done"])
    turns = []
    batches = [[{"hint_id": "h1", "text": "private"}]]
    acked = []

    res = goals.run_kanban_goal_loop(
        task_id="hinted",
        goal_text="ship feature",
        run_turn=lambda p: turns.append(p) or "done",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail(r),
        max_turns=2,
        first_response="started",
        prepare_turn=lambda p: (p + "\n\nPRIVATE", batches.pop(0)),
        ack_turn=lambda batch: acked.extend(batch),
    )
    assert res["turns_used"] == 2
    assert len(turns) == 1  # hint did not create a separate turn
    assert turns[0].endswith("\n\nPRIVATE")
    assert [item["hint_id"] for item in acked] == ["h1"]


def test_goal_loop_never_sends_hinted_model_prose_to_judge_emit_or_log(monkeypatch):
    canary = "SECRET-CANARY"
    judged = []
    emitted = []
    logged = []
    statuses = iter(["running", "done"])

    def judge(_goal, response, **_kwargs):
        judged.append(response)
        return "continue", "fixed", False, None

    monkeypatch.setattr(goals, "judge_goal", judge)
    result = goals.run_kanban_goal_loop(
        task_id="hint-private",
        goal_text="ship",
        run_turn=lambda _prompt: canary,
        task_status_fn=lambda: next(statuses),
        block_fn=lambda reason: pytest.fail(reason),
        first_response="public first",
        prepare_turn=lambda prompt: (prompt + "\nprivate", [{"hint_id": "h1", "text": "a"}]),
        ack_turn=lambda _batch: None,
        emit_turn=emitted.append,
        log=logged.append,
    )
    assert result["outcome"] == "completed_by_worker"
    assert judged == ["public first"]
    assert emitted == ["Operator-guided model turn completed."]
    assert canary not in repr((result, judged, emitted, logged))


@pytest.mark.parametrize(
    ("raw", "expected_reason"),
    [
        ({"response": "SECRET", "failed": True}, None),
        ({"response": "SECRET", "partial": True}, None),
        ({"response": "SECRET", "failed": True, "failure_reason": "rate_limit"}, "rate_limit"),
        ({"response": "SECRET", "partial": True, "failure_reason": "billing"}, "billing"),
    ],
)
def test_hinted_goal_turn_failure_is_fixed_unacked_unjudged_and_counted(
    monkeypatch, raw, expected_reason
):
    judged = []
    emitted = []
    logged = []
    acked = []
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda _goal, response, **_kwargs: judged.append(response)
        or ("continue", "fixed", False, None),
    )

    result = goals.run_kanban_goal_loop(
        task_id="hint-failure",
        goal_text="ship",
        run_turn=lambda _prompt: goals.KanbanGoalTurnResult(**raw),
        task_status_fn=lambda: "running",
        block_fn=lambda reason: pytest.fail(reason),
        max_turns=5,
        first_response="public first",
        prepare_turn=lambda prompt: (prompt + "\nprivate", [{"hint_id": "h1", "text": "secret"}]),
        ack_turn=lambda batch: acked.extend(batch),
        emit_turn=emitted.append,
        log=logged.append,
    )

    assert result["outcome"] == "turn_failed"
    assert result["turns_used"] == 2
    assert result["result"]["final_response"] == "Operator-guided model turn failed."
    assert result["result"].get("failure_reason") == expected_reason
    assert raw.get("failed") is result["result"].get("failed")
    assert raw.get("partial") is result["result"].get("partial")
    assert acked == [] and emitted == []
    assert judged == ["public first"]
    assert "SECRET" not in repr((result, judged, emitted, logged))


def test_goal_turn_exception_is_fixed_unacked_and_counted(monkeypatch):
    judged = []
    acked = []
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda _goal, response, **_kwargs: judged.append(response)
        or ("continue", "fixed", False, None),
    )

    def explode(_prompt):
        raise RuntimeError("SECRET provider prose")

    result = goals.run_kanban_goal_loop(
        task_id="exception",
        goal_text="ship",
        run_turn=explode,
        task_status_fn=lambda: "running",
        block_fn=lambda reason: pytest.fail(reason),
        first_response="public first",
        prepare_turn=lambda prompt: (prompt, [{"hint_id": "h1", "text": "secret"}]),
        ack_turn=lambda batch: acked.extend(batch),
    )
    assert result == {
        "outcome": "turn_failed",
        "turns_used": 2,
        "result": {
            "final_response": "Operator-guided model turn failed.",
            "error": {
                "code": "operator_guided_turn_failed",
                "message": "The operator-guided model turn failed.",
            },
            "failed": True,
        },
    }
    assert judged == ["public first"]
    assert acked == []
    assert "SECRET" not in repr(result)


def test_no_hint_goal_turn_success_and_judging_are_unchanged(monkeypatch):
    judged = []
    statuses = iter(["running", "done"])
    exact = "exact response bytes\n"
    monkeypatch.setattr(
        goals,
        "judge_goal",
        lambda _goal, response, **_kwargs: judged.append(response)
        or ("continue", "fixed", False, None),
    )
    result = goals.run_kanban_goal_loop(
        task_id="plain",
        goal_text="ship",
        run_turn=lambda _prompt: exact,
        task_status_fn=lambda: next(statuses),
        block_fn=lambda reason: pytest.fail(reason),
        first_response="first",
    )
    assert result["outcome"] == "completed_by_worker"
    assert judged == ["first"]


def test_no_hint_failed_goal_turn_is_fixed_and_stops_without_extra_turn(monkeypatch):
    calls = []
    _patch_judge(monkeypatch, ["continue"] * 3)
    result = goals.run_kanban_goal_loop(
        task_id="plain-failure",
        goal_text="ship",
        run_turn=lambda prompt: calls.append(prompt) or goals.KanbanGoalTurnResult(
            response="SECRET", failed=True, failure_reason=None
        ),
        task_status_fn=lambda: "running",
        block_fn=lambda reason: pytest.fail(reason),
        max_turns=5,
        first_response="first",
    )
    assert len(calls) == 1
    assert result["outcome"] == "turn_failed"
    assert result["turns_used"] == 2
    assert result["result"]["failed"] is True
    assert "SECRET" not in repr(result)


@pytest.mark.parametrize(
    ("later", "expected_exit"),
    [
        ({"failed": True}, 1),
        ({"partial": True}, 1),
        ({"failed": True, "failure_reason": "rate_limit"}, 75),
        ({"partial": True, "failure_reason": "billing"}, 75),
    ],
)
def test_main_goal_outcome_replaces_initial_success_and_controls_exit(monkeypatch, later, expected_exit):
    from cli import _apply_goal_loop_outcome, _kanban_worker_exit_code

    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    fixed = goals._fixed_goal_turn_failure(goals.KanbanGoalTurnResult(**later))
    result, response = _apply_goal_loop_outcome(
        {"final_response": "initial success"},
        "initial success",
        {"outcome": "turn_failed", "turns_used": 2, "result": fixed},
    )
    assert result is fixed
    assert response == "Operator-guided model turn failed."
    assert _kanban_worker_exit_code(result) == expected_exit
    assert "initial success" not in repr((result, response))


def test_apply_goal_loop_outcome_accepts_any_typed_failure_outcome():
    from cli import _apply_goal_loop_outcome

    fixed = goals._fixed_goal_turn_failure()
    result, response = _apply_goal_loop_outcome(
        {"final_response": "initial success"},
        "initial success",
        {"outcome": "operational_failed", "turns_used": 1, "result": fixed},
    )
    assert result is fixed
    assert response == "Operator-guided model turn failed."


def test_quiet_goal_output_is_buffered_until_success_and_keeps_exact_order():
    from cli import _publish_goal_loop_output

    emitted = []
    result, response = _publish_goal_loop_output(
        {"final_response": "initial\nbytes"},
        "initial\nbytes",
        {"outcome": "completed_by_worker", "turns_used": 3},
        ["continuation one\n", "continuation two"],
        emit=emitted.append,
    )
    assert (result, response) == ({"final_response": "initial\nbytes"}, "initial\nbytes")
    assert emitted == ["initial\nbytes", "continuation one\n", "continuation two"]


@pytest.mark.parametrize(
    "later",
    [
        {"failed": True},
        {"partial": True},
        {"failed": True, "failure_reason": "rate_limit"},
    ],
)
def test_quiet_goal_output_discards_initial_and_buffered_success_on_failure(later):
    from cli import _kanban_worker_exit_code, _publish_goal_loop_output

    fixed = goals._fixed_goal_turn_failure(goals.KanbanGoalTurnResult(**later))
    emitted = []
    result, response = _publish_goal_loop_output(
        {"final_response": "INITIAL SUCCESS CANARY"},
        "INITIAL SUCCESS CANARY",
        {"outcome": "operational_failed", "turns_used": 2, "result": fixed},
        ["BUFFERED HINTED SUCCESS CANARY"],
        emit=emitted.append,
    )
    assert result is fixed
    assert response == "Operator-guided model turn failed."
    assert emitted == ["Operator-guided model turn failed."]
    assert "SUCCESS CANARY" not in repr(emitted)
    assert _kanban_worker_exit_code(result) in (1, 75)


def test_loop_task_status_exception_is_fixed_operational_failure(monkeypatch):
    from cli import _kanban_worker_exit_code

    def explode():
        raise RuntimeError("SECRET status path")

    result = goals.run_kanban_goal_loop(
        task_id="status-exception", goal_text="ship", run_turn=lambda _: "unused",
        task_status_fn=explode, block_fn=lambda _: None, first_response="success",
    )
    assert result["outcome"] == "operational_failed"
    assert result["result"] == goals._fixed_goal_turn_failure()
    assert _kanban_worker_exit_code(result["result"]) == 1
    assert "SECRET" not in repr(result)


@pytest.mark.parametrize("verdicts,max_turns", [(["done", "done"], 10), (["continue"], 1)])
def test_loop_block_exception_is_fixed_operational_failure(monkeypatch, verdicts, max_turns):
    from cli import _kanban_worker_exit_code

    _patch_judge(monkeypatch, verdicts)

    def explode(_reason):
        raise RuntimeError("SECRET persistence path")

    result = goals.run_kanban_goal_loop(
        task_id="block-exception", goal_text="ship", run_turn=lambda _: "continued",
        task_status_fn=lambda: "running", block_fn=explode,
        max_turns=max_turns, first_response="success",
    )
    assert result["outcome"] == "operational_failed"
    assert result["result"] == goals._fixed_goal_turn_failure()
    assert _kanban_worker_exit_code(result["result"]) == 1
    assert "SECRET" not in repr(result)


def test_loop_judge_exception_is_fixed_operational_failure(monkeypatch):
    from cli import _kanban_worker_exit_code

    def explode(*_args, **_kwargs):
        raise RuntimeError("SECRET judge path")

    monkeypatch.setattr(goals, "judge_goal", explode)
    result = goals.run_kanban_goal_loop(
        task_id="judge-exception", goal_text="ship", run_turn=lambda _: "unused",
        task_status_fn=lambda: "running", block_fn=lambda _: None,
        first_response="success",
    )
    assert result["outcome"] == "operational_failed"
    assert result["result"] == goals._fixed_goal_turn_failure()
    assert _kanban_worker_exit_code(result["result"]) == 1
    assert "SECRET" not in repr(result)


def test_loop_judge_parse_failure_is_fixed_operational_failure(monkeypatch):
    monkeypatch.setattr(
        goals, "judge_goal", lambda *_a, **_kw: ("continue", "PRIVATE PARSE", True, None)
    )
    result = goals.run_kanban_goal_loop(
        task_id="parse", goal_text="ship", run_turn=lambda _: pytest.fail("no turn"),
        task_status_fn=lambda: "running", block_fn=lambda _: pytest.fail("no block"),
        first_response="success",
    )
    assert result["outcome"] == "operational_failed"
    assert result["result"] == goals._fixed_goal_turn_failure()
    assert "PRIVATE" not in repr(result)


@pytest.mark.parametrize("status", ["ready", "archived", "cancelled", "unexpected"])
def test_loop_unexpected_or_reclaimed_status_is_operational_failure(status):
    result = goals.run_kanban_goal_loop(
        task_id="stale", goal_text="ship", run_turn=lambda _: pytest.fail("no turn"),
        task_status_fn=lambda: status, block_fn=lambda _: pytest.fail("no block"),
        first_response="initial success",
    )
    assert result["outcome"] == "operational_failed"
    assert result["result"] == goals._fixed_goal_turn_failure()


def test_quiet_goal_setup_missing_task_id_is_typed_operational_failure(monkeypatch):
    from cli import _run_kanban_goal_loop_q
    monkeypatch.delenv("HERMES_KANBAN_TASK", raising=False)
    result = _run_kanban_goal_loop_q(object(), "initial success")
    assert result["outcome"] == "operational_failed"
    assert result["result"] == goals._fixed_goal_turn_failure()


@pytest.mark.parametrize("setup", ["missing_task", "empty_goal"])
def test_quiet_goal_setup_invalid_card_is_typed_operational_failure(tmp_path, monkeypatch, setup):
    from cli import _run_kanban_goal_loop_q
    from hermes_cli import kanban_db as kb
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    kb.init_db()
    with kb.connect_closing() as conn:
        if setup == "missing_task":
            task_id = "t_missing"
        else:
            task_id = kb.create_task(conn, title="temporary")
            conn.execute("UPDATE tasks SET title='',body='' WHERE id=?", (task_id,))
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    result = _run_kanban_goal_loop_q(object(), "initial success")
    assert result["outcome"] == "operational_failed"
    assert result["result"] == goals._fixed_goal_turn_failure()


@pytest.mark.parametrize("raw,reason,exit_code", [
    ({"failed": True, "final_response": "PRIVATE", "error": "provider"}, None, 1),
    ({"partial": True, "final_response": "PRIVATE", "provider": "x"}, None, 1),
    ({"failed": True, "failure_reason": "rate_limit", "final_response": "PRIVATE"}, "rate_limit", 75),
    ({"partial": True, "failure_reason": "billing", "final_response": "PRIVATE"}, "billing", 75),
])
def test_initial_goal_failure_without_hints_is_fixed(monkeypatch, raw, reason, exit_code):
    from cli import _kanban_worker_exit_code, _run_kanban_worker_turns
    monkeypatch.setenv("HERMES_KANBAN_TASK", "task")
    class Agent:
        def run_conversation(self, **_kwargs):
            return raw
    class CLI:
        agent = Agent()
        conversation_history = []
    class Boundary:
        def prepare(self, prompt):
            return prompt, []
        def ack(self, _batch):
            pytest.fail("empty batch must not ack")
    result, response = _run_kanban_worker_turns(CLI(), "prompt", boundary=Boundary(), goal_mode=True)
    assert response == "Operator-guided model turn failed."
    assert result.get("failure_reason") == reason
    assert "PRIVATE" not in repr(result)
    assert _kanban_worker_exit_code(result) == exit_code


def test_loop_blocks_on_budget_exhaustion(monkeypatch):
    _patch_judge(monkeypatch, ["continue"] * 10)
    blocked = {}

    def _block(reason):
        blocked["reason"] = reason

    res = goals.run_kanban_goal_loop(
        task_id="t3",
        goal_text="endless task",
        run_turn=lambda p: "still going",
        task_status_fn=lambda: "running",
        block_fn=_block,
        max_turns=3,
        first_response="turn1",
    )
    assert res["outcome"] == "blocked_budget"
    assert res["turns_used"] == 3
    assert "turn budget" in blocked["reason"].lower()


def test_loop_finalize_nudge_when_judge_done_but_open(monkeypatch):
    # Judge says done, but worker never terminated → one finalize nudge,
    # then worker completes.
    _patch_judge(monkeypatch, ["done", "done"])
    statuses = iter(["running", "done"])
    turns = []

    res = goals.run_kanban_goal_loop(
        task_id="t4",
        goal_text="task",
        run_turn=lambda p: turns.append(p) or "ok",
        task_status_fn=lambda: next(statuses),
        block_fn=lambda r: pytest.fail("should not block"),
        max_turns=10,
        first_response="looks done",
    )
    assert res["outcome"] == "completed_by_worker"
    assert len(turns) == 1
    assert "still open" in turns[0]


def test_loop_blocks_when_judge_done_but_never_finalizes(monkeypatch):
    # Judge keeps saying done, worker never calls kanban_complete → block
    # after the single finalize nudge.
    _patch_judge(monkeypatch, ["done", "done"])
    blocked = {}

    res = goals.run_kanban_goal_loop(
        task_id="t5",
        goal_text="task",
        run_turn=lambda p: "still not finalizing",
        task_status_fn=lambda: "running",
        block_fn=lambda r: blocked.update(reason=r),
        max_turns=10,
        first_response="looks done",
    )
    assert res["outcome"] == "blocked_budget"
    assert "finalize" in blocked["reason"].lower()


def test_loop_stops_if_task_reclaimed(monkeypatch):
    _patch_judge(monkeypatch, ["continue"])
    res = goals.run_kanban_goal_loop(
        task_id="t6",
        goal_text="task",
        run_turn=lambda p: pytest.fail("should not run a turn"),
        task_status_fn=lambda: "archived",
        block_fn=lambda r: pytest.fail("should not block"),
        first_response="x",
    )
    assert res["outcome"] == "operational_failed"
