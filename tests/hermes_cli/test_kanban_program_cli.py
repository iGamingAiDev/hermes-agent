import json
import os
import shlex
import subprocess
import sys
from io import StringIO
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


@pytest.fixture
def kanban_home(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    return home


def _argv(extra=""):
    return (
        "program create --title 'Ship it' --assignee Planner "
        "--allowed-assignee planner --allowed-assignee worker "
        "--orchestrator planner --max-depth 2 --max-tasks 8 "
        "--max-concurrency 2 --max-runtime-seconds 60 "
        "--max-wall-clock-seconds 300 --goal-max-turns 5 --json " + extra
    )


VALID_CGROUP = b"0::/system.slice/vigo-mc-command-0123456789abcdef01.service\n"

_WORKTREE = Path(__file__).resolve().parents[2]


def _run_real_cli(home, *args):
    env = dict(os.environ)
    env["HERMES_HOME"] = str(home)
    env["PYTHONPATH"] = str(_WORKTREE)
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", "kanban", *args],
        cwd=_WORKTREE,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.fixture
def mission_control_cgroup(monkeypatch):
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: VALID_CGROUP)


def test_real_cli_propagates_program_create_denial_without_creating_db(tmp_path):
    home = tmp_path / ".hermes"
    result = _run_real_cli(home, *shlex.split(_argv()))

    assert result.returncode == 2
    assert "mission control" in result.stderr.lower()
    assert not (home / "kanban.db").exists()


def test_real_cli_propagates_success_for_harmless_kanban_command(tmp_path):
    home = tmp_path / ".hermes"
    result = _run_real_cli(home, "boards", "list", "--json")

    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)[0]["slug"] == "default"
    assert not (home / "kanban.db").exists()


def test_program_create_broker_emits_deterministic_safe_json_and_canonical_root(
    kanban_home, mission_control_cgroup,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    assert kc.kanban_command(args) == 0
    with kb.connect() as conn:
        task = kb.list_tasks(conn)[0]
    payload = {
        "id": task.id,
        "status": task.status,
        "orchestration_root_id": task.orchestration_root_id,
        "policy": json.loads(task.orchestration_policy.to_json()),
    }
    assert set(payload) == {"id", "status", "orchestration_root_id", "policy"}
    assert payload["id"] == payload["orchestration_root_id"]
    assert payload["status"] == "ready"
    assert payload["policy"]["allowed_assignees"] == ["planner", "worker"]
    assert payload["policy"]["goal_max_turns"] == 5
    with kb.connect() as conn:
        tasks = kb.list_tasks(conn)
    assert len(tasks) == 1
    assert tasks[0].orchestration_depth == 0
    assert tasks[0].goal_max_turns == 5


@pytest.mark.parametrize("extra", [
    "--max-concurrency 0",
    "--orchestrator outsider",
    "--allowed-assignee planner",
    "--assignee worker",
    "--project missing-project",
])
def test_program_create_cli_invalid_input_leaves_no_root(
    kanban_home, mission_control_cgroup, extra,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv(extra)))
    assert kc.kanban_command(args) == 2
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_program_create_is_rejected_by_slash_without_creating_row(kanban_home):
    output = kc.run_slash(_argv())
    assert "trusted" in output.lower() and "direct" in output.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_program_create_is_rejected_by_direct_argv_without_creating_row(
    kanban_home, capsys, monkeypatch,
):
    """A terminal-capable model must not turn direct argv into root authority."""
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: b"0::/user.slice/test.service\n")
    monkeypatch.setattr(kb, "init_db", lambda *a, **k: pytest.fail("DB initialized"))
    assert kc.kanban_command(args) == 2
    assert "mission control" in capsys.readouterr().err.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_program_create_direct_argv_rejects_forged_namespace_marker(
    kanban_home, capsys, monkeypatch,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    args.trusted = True
    args.broker = "mission-control"
    args._trusted_program_create = True
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: b"0::/user.slice/test.service\n")
    assert kc.kanban_command(args) == 2
    assert "mission control" in capsys.readouterr().err.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("cgroup", [
    b"0::/system.slice/vigo-mc-command-0123456789abcdef0.service\n",
    b"0::/system.slice/hermes-mc-command-0123456789abcdef01.service\n",
    b"0::/system.slice/vigo-mc-command-0123456789ABCDEf01.service\n",
    b"0::/system.slice/vigo-mc-command-0123456789abcdef01.service.extra\n",
    b"0::/system.slice/../system.slice/vigo-mc-command-0123456789abcdef01.service\n",
    VALID_CGROUP + b"1:name=systemd:/system.slice/other.service\n",
    b"not:a:cgroup:line\n",
    b"",
    b"x" * (kb._CGROUP_READ_LIMIT + 1),
])
def test_program_create_rejects_non_exact_cgroup(kanban_home, monkeypatch, cgroup):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: cgroup)
    assert kc.kanban_command(args) == 2
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


@pytest.mark.parametrize("error", [PermissionError("denied"), FileNotFoundError()])
def test_program_create_rejects_unreadable_or_missing_cgroup(
    kanban_home, monkeypatch, error,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))

    def unreadable():
        raise error

    monkeypatch.setattr(kb, "_read_self_cgroup", unreadable)
    assert kc.kanban_command(args) == 2
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def test_program_create_accepts_exact_v1_systemd_cgroup(
    kanban_home, monkeypatch,
):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    args = parser.parse_args(shlex.split(_argv()))
    monkeypatch.setattr(
        kb, "_read_self_cgroup",
        lambda: b"7:cpu,cpuacct:/ordinary\n1:name=systemd:/system.slice/"
        b"vigo-mc-command-fedcba9876543210ab.service\n",
    )
    assert kc.kanban_command(args) == 0


def test_interactive_kanban_dispatch_uses_fail_closed_slash_path(kanban_home, capsys):
    from hermes_cli.cli_commands_mixin import CLICommandsMixin

    CLICommandsMixin._handle_kanban_command(object(), "/kanban " + _argv())
    assert "trusted direct" in capsys.readouterr().out.lower()
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] == 0


def _parse_program(argv):
    parser = kc.build_parser(__import__("argparse").ArgumentParser().add_subparsers())
    return parser.parse_args(shlex.split(argv))


def _create_program_root():
    policy = kb.OrchestrationPolicy(
        allowed_assignees=("planner", "worker"),
        orchestrator_assignees=("planner",),
        max_depth=2,
        max_tasks=8,
        max_runtime_seconds=60,
        max_concurrency=2,
        max_wall_clock_seconds=300,
        goal_max_turns=5,
    )
    with kb.connect() as conn:
        return kb.create_task(
            conn,
            title="program",
            assignee="planner",
            orchestration_policy=policy,
            created_by="mission-control",
            idempotency_key="cli-root",
        )


def test_program_extend_deadline_cli_emits_one_json_document(
    kanban_home, mission_control_cgroup, capsys, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    root_id = _create_program_root()
    args = _parse_program(
        f"program extend-deadline {root_id} --new-deadline {now + 600} "
        "--actor operator --idempotency-key extend-cli --json"
    )
    assert kc.kanban_command(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out)["effective_deadline"] == now + 600
    assert captured.out.count("\n") == 1


def test_program_change_prepare_apply_cli_reads_strict_json_stdin(
    kanban_home, mission_control_cgroup, capsys, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    root_id = _create_program_root()
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO(json.dumps({"operation": "extend_deadline", "new_deadline": now + 600})),
    )
    prepare_args = _parse_program(
        f"program change prepare {root_id} --actor operator "
        "--idempotency-key prepare-cli --json"
    )
    assert kc.kanban_command(prepare_args) == 0
    prepared = json.loads(capsys.readouterr().out)

    apply_args = _parse_program(
        f"program change apply {root_id} {prepared['request_id']} --actor approver "
        "--idempotency-key apply-cli --json"
    )
    assert kc.kanban_command(apply_args) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["status"] == "applied"
    assert applied["effective_deadline"] == now + 600


def test_program_change_prepare_cli_rejects_non_object_json(
    kanban_home, mission_control_cgroup, capsys, monkeypatch
):
    root_id = _create_program_root()
    monkeypatch.setattr(sys, "stdin", StringIO("[]"))
    args = _parse_program(
        f"program change prepare {root_id} --actor operator "
        "--idempotency-key invalid-cli --json"
    )
    assert kc.kanban_command(args) != 0
    assert capsys.readouterr().out == ""
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM program_change_requests").fetchone()[0] == 0


def test_real_cli_rejects_untrusted_program_change_without_creating_db(tmp_path):
    home = tmp_path / ".hermes"
    result = _run_real_cli(
        home,
        "program",
        "change",
        "prepare",
        "root-id",
        "--actor",
        "operator",
        "--idempotency-key",
        "prepare-cli",
        "--json",
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert not (home / "kanban.db").exists()


@pytest.mark.parametrize(
    "raw",
    [
        '{"operation":"bogus","operation":"archive"}',
        '{"operation":"archive","meta":{"x":1,"x":2}}',
        '{"operation":"archive","reason":NaN}',
        '{"operation":"archive","reason":Infinity}',
        '{"operation":"archive","reason":-Infinity}',
        '{"operation":"archive"} {"operation":"archive"}',
    ],
)
def test_program_change_prepare_cli_rejects_non_strict_json_without_row(
    kanban_home, mission_control_cgroup, capsys, monkeypatch, raw
):
    root_id = _create_program_root()
    monkeypatch.setattr(sys, "stdin", StringIO(raw))
    args = _parse_program(
        f"program change prepare {root_id} --actor operator "
        "--idempotency-key strict-invalid --json"
    )
    assert kc.kanban_command(args) != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM program_change_requests").fetchone()[0] == 0


@pytest.mark.parametrize("extra_bytes", [0, 1])
def test_strict_request_parser_enforces_exact_utf8_byte_limit(extra_bytes):
    prefix = '{"operation":"archive","reason":"'
    suffix = '"}'
    padding = "x" * (16 * 1024 + extra_bytes - len((prefix + suffix).encode("utf-8")))
    raw = prefix + padding + suffix
    assert len(raw.encode("utf-8")) == 16 * 1024 + extra_bytes
    if extra_bytes:
        with pytest.raises(ValueError, match="exceeds 16384 bytes"):
            kc._read_strict_json_object(StringIO(raw))
    else:
        assert kc._read_strict_json_object(StringIO(raw))["operation"] == "archive"


def test_strict_request_parser_rejects_invalid_utf8_bytes():
    class BinaryStdin:
        def __init__(self):
            from io import BytesIO

            self.buffer = BytesIO(b'{"operation":"archive","reason":"\xff"}')

    with pytest.raises(ValueError, match="UTF-8"):
        kc._read_strict_json_object(BinaryStdin())


def test_program_change_rejection_precedes_database_initialization(
    tmp_path, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(kb, "_read_self_cgroup", lambda: b"0::/user.slice/test.service\n")
    monkeypatch.setattr(kb, "init_db", lambda *a, **k: pytest.fail("DB initialized"))
    monkeypatch.setattr(sys, "stdin", StringIO('{"operation":"archive"}'))
    args = _parse_program(
        "program change prepare root-id --actor operator "
        "--idempotency-key denied --json"
    )
    assert kc.kanban_command(args) == 2
    assert capsys.readouterr().out == ""


def test_program_capabilities_emits_exact_mission_contract_without_db_init(
    tmp_path, mission_control_cgroup, monkeypatch, capsys
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(kb, "init_db", lambda *a, **k: pytest.fail("DB initialized"))
    args = _parse_program("--board default program capabilities --json")

    assert kc.kanban_command(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out == (
        '{"contract":"hermes-program-control","board":"default","version":2,'
        '"schema":2,"operator_visible_decision_briefs":1,"cli":1,'
        '"dispatcher_gate":1,"classic_worker_boundary":1,'
        '"goal_loop_boundary":1,"ack_writer":1,"events":1}\n'
    )
    assert not (tmp_path / ".hermes" / "kanban.db").exists()


def _program_request(argv, document, monkeypatch):
    monkeypatch.setattr(sys, "stdin", StringIO(json.dumps(document, separators=(",", ":"))))
    return _parse_program(argv)


def test_program_hint_records_versioned_root_bound_hint_and_replays(
    kanban_home, mission_control_cgroup, monkeypatch, capsys
):
    root_id = _create_program_root()
    document = {
        "program_id": root_id,
        "node_id": root_id,
        "text": "operator context",
        "expected_node_version": 0,
        "idempotency_key": "abcdefghijklmnop",
    }
    argv = "program hint --request-json-stdin --json"
    assert kc.kanban_command(_program_request(argv, document, monkeypatch)) == 0
    first = json.loads(capsys.readouterr().out)
    assert first == {
        "ok": True,
        "hint_id": first["hint_id"],
        "node_id": root_id,
        "state": "recorded",
        "node_version": 1,
        "deduplicated": False,
    }

    assert kc.kanban_command(_program_request(argv, document, monkeypatch)) == 0
    replay = json.loads(capsys.readouterr().out)
    assert replay == {**first, "deduplicated": True}
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM program_operator_hints").fetchone()[0] == 1


def test_program_hint_stale_or_idempotency_mismatch_has_no_mutation(
    kanban_home, mission_control_cgroup, monkeypatch, capsys
):
    root_id = _create_program_root()
    base = {
        "program_id": root_id,
        "node_id": root_id,
        "text": "first",
        "expected_node_version": 0,
        "idempotency_key": "abcdefghijklmnop",
    }
    argv = "program hint --request-json-stdin --json"
    assert kc.kanban_command(_program_request(argv, base, monkeypatch)) == 0
    capsys.readouterr()
    assert kc.kanban_command(
        _program_request(argv, {**base, "text": "different"}, monkeypatch)
    ) != 0
    assert kc.kanban_command(
        _program_request(
            argv,
            {**base, "idempotency_key": "ponmlkjihgfedcba"},
            monkeypatch,
        )
    ) != 0
    with kb.connect() as conn:
        assert conn.execute("SELECT COUNT(*) FROM program_operator_hints").fetchone()[0] == 1


def test_program_decision_select_is_cas_option_bound_and_idempotent(
    kanban_home, mission_control_cgroup, monkeypatch, capsys
):
    root_id = _create_program_root()
    with kb.connect() as conn:
        kb.create_program_decision_checkpoint(
            conn,
            root_id,
            "decision-1",
            node_id=root_id,
            recommended_option_id="option-1",
            title="Choose delivery path",
            recommendation_rationale="Option one minimizes operational risk.",
            options=(
                {
                    "option_id": "option-1", "label": "Path one", "summary": "First path",
                    "benefits": ["Lower risk"], "risks": ["Slower"],
                    "reversibility": "reversible", "security_impact": "None",
                    "cost_impact": "Low", "operations_impact": "Low",
                },
                {
                    "option_id": "option-2", "label": "Path two", "summary": "Second path",
                    "benefits": ["Faster"], "risks": ["Higher risk"],
                    "reversibility": "partially_reversible", "security_impact": "Review",
                    "cost_impact": "Medium", "operations_impact": "Medium",
                },
            ),
        )
    document = {
        "root_id": root_id,
        "checkpoint_id": "decision-1",
        "option_id": "option-2",
        "expected_version": 1,
        "idempotency_key": "abcdefghijklmnop",
        "actor": "control:owner",
    }
    argv = "program decision select --request-json-stdin --json"
    assert kc.kanban_command(_program_request(argv, document, monkeypatch)) == 0
    first = json.loads(capsys.readouterr().out)
    assert first == {
        "ok": True,
        "checkpoint_id": "decision-1",
        "state": "selected",
        "version": 2,
        "selected_option_id": "option-2",
        "deduplicated": False,
    }
    assert kc.kanban_command(_program_request(argv, document, monkeypatch)) == 0
    assert json.loads(capsys.readouterr().out) == {**first, "deduplicated": True}


def test_program_hint_state_rollback_cannot_authorize_duplicate_transition(
    kanban_home, mission_control_cgroup
):
    root_id = _create_program_root()
    first = {
        "program_id": root_id,
        "node_id": root_id,
        "text": "first",
        "expected_node_version": 0,
        "idempotency_key": "abcdefghijklmnop",
    }
    with kb.connect() as conn:
        kb.record_program_operator_hint(conn, first)
        conn.execute(
            "UPDATE tasks SET program_control_version=0 WHERE id=?", (root_id,)
        )
        with pytest.raises(ValueError, match="integrity"):
            kb.record_program_operator_hint(
                conn,
                {**first, "text": "second", "idempotency_key": "ponmlkjihgfedcba"},
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM program_operator_hints"
        ).fetchone()[0] == 1


def test_program_hint_history_cap_rejects_new_transition_but_preserves_replay(
    kanban_home, mission_control_cgroup
):
    root_id = _create_program_root()
    with kb.connect() as conn:
        conn.executemany(
            "INSERT INTO program_operator_hints "
            "(hint_id,root_id,node_id,text,expected_version,resulting_version,idempotency_key,"
            "request_fingerprint,created_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                (
                    f"synthetic-{index}", root_id, root_id, "seed", index, index + 1,
                    f"synthetic-key-{index}", f"sha256:{index:064x}", 1_700_000_000,
                )
                for index in range(9_999)
            ),
        )
        conn.execute(
            "UPDATE tasks SET program_control_version=9999 WHERE id=?", (root_id,)
        )
        final = {
            "program_id": root_id,
            "node_id": root_id,
            "text": "last supported hint",
            "expected_node_version": 9_999,
            "idempotency_key": "abcdefghijklmnop",
        }
        first = kb.record_program_operator_hint(conn, final)
        assert first["node_version"] == 10_000
        replay = kb.record_program_operator_hint(conn, final)
        assert replay == {**first, "deduplicated": True}
        with pytest.raises(ValueError, match="capacity"):
            kb.record_program_operator_hint(
                conn,
                {
                    **final,
                    "expected_node_version": 10_000,
                    "idempotency_key": "ponmlkjihgfedcba",
                },
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM program_operator_hints"
        ).fetchone()[0] == 10_000


def _decision_options():
    return (
        {
            "option_id": "option-1", "label": "Path one", "summary": "First path",
            "benefits": ["Lower risk"], "risks": ["Slower"],
            "reversibility": "reversible", "security_impact": "None",
            "cost_impact": "Low", "operations_impact": "Low",
        },
        {
            "option_id": "option-2", "label": "Path two", "summary": "Second path",
            "benefits": ["Faster"], "risks": ["Higher risk"],
            "reversibility": "partially_reversible", "security_impact": "Review",
            "cost_impact": "Medium", "operations_impact": "Medium",
        },
    )


def _create_decision(conn, root_id, checkpoint_id="decision-1", **overrides):
    values = {
        "node_id": root_id,
        "recommended_option_id": "option-1",
        "title": "Choose delivery path",
        "recommendation_rationale": "Option one minimizes operational risk.",
        "options": _decision_options(),
    }
    values.update(overrides)
    kb.create_program_decision_checkpoint(conn, root_id, checkpoint_id, **values)


def test_program_decision_state_rollback_cannot_authorize_second_selection(
    kanban_home, mission_control_cgroup
):
    root_id = _create_program_root()
    first = {
        "root_id": root_id,
        "checkpoint_id": "decision-1",
        "option_id": "option-1",
        "expected_version": 1,
        "idempotency_key": "abcdefghijklmnop",
        "actor": "control:owner",
    }
    with kb.connect() as conn:
        _create_decision(conn, root_id)
        kb.select_program_decision(conn, first)
        conn.execute(
            "UPDATE program_decisions SET state='pending',version=1,selected_option_id=NULL "
            "WHERE root_id=? AND checkpoint_id='decision-1'",
            (root_id,),
        )
        with pytest.raises(ValueError, match="integrity"):
            kb.select_program_decision(
                conn,
                {**first, "option_id": "option-2", "idempotency_key": "ponmlkjihgfedcba"},
            )
        assert conn.execute(
            "SELECT COUNT(*) FROM program_decision_events"
        ).fetchone()[0] == 1


def test_checkpoint_producer_matches_mission_public_projection_bounds(
    kanban_home, mission_control_cgroup
):
    root_id = _create_program_root()
    base = list(_decision_options())
    invalid = [
        {"options": base[:1]},
        {"options": base + [{**base[0], "option_id": f"option-{index}"} for index in range(3, 6)]},
        {"title": "e\u0301"},
        {"options": [{**base[0], "benefits": ["same", "same"]}, base[1]]},
        {"options": [{**base[0], "risks": [str(index) for index in range(11)]}, base[1]]},
        {"options": [{**base[0], "benefits": ["e\u0301"]}, base[1]]},
    ]
    with kb.connect() as conn:
        for index, overrides in enumerate(invalid):
            with pytest.raises(ValueError):
                _create_decision(conn, root_id, f"invalid-{index}", **overrides)
        assert conn.execute("SELECT COUNT(*) FROM program_decisions").fetchone()[0] == 0


def test_program_schema_preflight_rejects_owned_namespace_debris(
    kanban_home, mission_control_cgroup, monkeypatch
):
    with kb.connect() as conn:
        conn.execute("CREATE TABLE program_phase2_debris(value TEXT)")
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    with pytest.raises(ValueError, match="unsupported program lifecycle schema"):
        kb.connect()


@pytest.mark.parametrize(
    "argv,document",
    [
        (
            "program hint --request-json-stdin --json",
            {"program_id": "p", "node_id": "n", "text": "x", "expected_node_version": 0,
             "idempotency_key": "abcdefghijklmnop", "extra": True},
        ),
        (
            "program decision select --request-json-stdin --json",
            {"root_id": "p", "checkpoint_id": "c", "option_id": "o", "expected_version": 0,
             "idempotency_key": "abcdefghijklmnop", "actor": "owner"},
        ),
    ],
)
def test_program_control_rejects_strict_request_before_db_initialization(
    tmp_path, mission_control_cgroup, monkeypatch, capsys, argv, document
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    monkeypatch.setattr(kb, "init_db", lambda *a, **k: pytest.fail("DB initialized"))
    assert kc.kanban_command(_program_request(argv, document, monkeypatch)) == 2
    assert capsys.readouterr().out == ""
