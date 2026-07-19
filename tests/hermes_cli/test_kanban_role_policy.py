from __future__ import annotations

import json
import sqlite3
from typing import Any

import pytest

from hermes_cli import kanban_db as kb


ROLES = (
    "orchestrator",
    "writer",
    "integrator",
    "reviewer",
    "security_reviewer",
    "qa_verifier",
)


@pytest.fixture(autouse=True)
def trusted_mission_control(monkeypatch):
    monkeypatch.setattr(kb, "is_mission_control_command_cgroup", lambda: True)


def _policy(**overrides: Any) -> kb.OrchestrationPolicy:
    values: dict[str, Any] = {
        "version": 3,
        "role_bindings": (
            ("planner", "orchestrator"),
            ("writer-a", "writer"),
            ("integrator-a", "integrator"),
            ("reviewer-a", "reviewer"),
            ("security-a", "security_reviewer"),
            ("qa-a", "qa_verifier"),
        ),
        "max_active_by_role": (
            ("orchestrator", 1),
            ("writer", 1),
            ("integrator", 1),
            ("reviewer", 1),
            ("security_reviewer", 1),
            ("qa_verifier", 1),
        ),
        "max_depth": 3,
        "max_tasks": 12,
        "max_runtime_seconds": 60,
        "max_concurrency": 6,
        "max_wall_clock_seconds": 300,
        "goal_max_turns": 5,
    }
    values.update(overrides)
    return kb.OrchestrationPolicy(**values)


def test_v3_policy_is_exact_canonical_and_digest_bound():
    policy = _policy()
    payload = json.loads(policy.to_json())
    assert payload == {
        "version": 3,
        "role_bindings": [
            {"profile": "planner", "role": "orchestrator"},
            {"profile": "writer-a", "role": "writer"},
            {"profile": "integrator-a", "role": "integrator"},
            {"profile": "reviewer-a", "role": "reviewer"},
            {"profile": "security-a", "role": "security_reviewer"},
            {"profile": "qa-a", "role": "qa_verifier"},
        ],
        "max_active_by_role": {role: 1 for role in ROLES},
        "max_depth": 3,
        "max_tasks": 12,
        "max_runtime_seconds": 60,
        "max_concurrency": 6,
        "max_wall_clock_seconds": 300,
        "goal_max_turns": 5,
    }
    assert kb.OrchestrationPolicy.from_json(policy.to_json()) == policy
    assert policy.policy_digest == kb._sha256_json(payload)[1]
    assert policy.role_for_profile("writer-a") == "writer"


def test_v3_policy_rejects_duplicate_profiles_unknown_roles_and_widening_limits():
    for overrides, match in (
        (
            {"role_bindings": _policy().role_bindings + (("writer-a", "reviewer"),)},
            "duplicate",
        ),
        (
            {
                "role_bindings": _policy().role_bindings[:-1]
                + (("qa-a", "future_role"),)
            },
            "role",
        ),
        (
            {
                "role_bindings": tuple(
                    binding
                    for binding in _policy().role_bindings
                    if binding[1] != "integrator"
                )
            },
            "integrator",
        ),
        (
            {
                "max_active_by_role": tuple(
                    (role, 2 if role == "integrator" else limit)
                    for role, limit in _policy().max_active_by_role
                )
            },
            "integrator",
        ),
        ({"max_concurrency": 5}, "sum"),
    ):
        with pytest.raises(ValueError, match=match):
            _policy(**overrides)


def test_v3_parser_rejects_unknown_or_missing_keys():
    payload = json.loads(_policy().to_json())
    for mutation in (
        lambda value: value.update(extra=True),
        lambda value: value.pop("role_bindings"),
    ):
        malformed = dict(payload)
        mutation(malformed)
        with pytest.raises(ValueError, match="malformed orchestration policy"):
            kb.OrchestrationPolicy.from_json(json.dumps(malformed))


def test_v2_has_no_inferred_role_authority():
    legacy = kb.OrchestrationPolicy(
        allowed_assignees=("planner", "worker"),
        orchestrator_assignees=("planner",),
        max_depth=2,
        max_tasks=4,
        max_runtime_seconds=60,
    )
    assert json.loads(legacy.to_json())["version"] == 2
    with pytest.raises(ValueError, match="role_policy_required"):
        legacy.role_for_profile("planner")


def _authority(claimed):
    return kb.CreationAuthority(
        task_id=claimed.id,
        run_id=claimed.current_run_id,
        claim_lock=claimed.claim_lock,
        actor_profile=claimed.assignee,
    )


def _root(conn, policy=None):
    return kb.create_task(
        conn,
        title="root",
        assignee="planner",
        orchestration_policy=policy or _policy(),
    )


def _child(conn, authority, *, title, assignee, role):
    return kb.create_task(
        conn,
        title=title,
        assignee=assignee,
        program_role=role,
        current_orchestrator_task_id=authority.task_id,
        creation_authority=authority,
    )


def test_v3_schema_and_root_run_copy_role_digest_and_generation(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        root = kb.get_task(conn, root_id)
        assert root is not None
        assert root.orchestration_policy is not None
        assert root.program_role == "orchestrator"
        assert (
            root.orchestration_policy_digest == root.orchestration_policy.policy_digest
        )
        claimed = kb.claim_task(conn, root_id)
        run = kb.latest_run(conn, root_id)
        assert claimed is not None
        assert run is not None
        assert run.program_role == "orchestrator"
        assert run.orchestration_policy_digest == root.orchestration_policy_digest
        assert run.evidence_generation == 0
        assert (
            conn.execute(
                "SELECT evidence_generation FROM program_lifecycle WHERE root_id=?",
                (root_id,),
            ).fetchone()[0]
            == 0
        )
    finally:
        conn.close()


def test_v3_children_require_active_orchestrator_and_exact_requested_role(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        root_claim = kb.claim_task(conn, root_id)
        assert root_claim is not None
        root_auth = _authority(root_claim)
        writer_id = _child(
            conn, root_auth, title="write", assignee="writer-a", role="writer"
        )
        with pytest.raises(ValueError, match="role"):
            _child(
                conn,
                root_auth,
                title="wrong binding",
                assignee="reviewer-a",
                role="writer",
            )
        writer_claim = kb.claim_task(conn, writer_id)
        assert writer_claim is not None
        with pytest.raises(ValueError, match="orchestrator"):
            _child(
                conn,
                _authority(writer_claim),
                title="unauthorized child",
                assignee="writer-a",
                role="writer",
            )
    finally:
        conn.close()


def test_v3_reassignment_is_same_role_only(tmp_path):
    policy = _policy(
        role_bindings=_policy().role_bindings + (("writer-b", "writer"),),
        max_active_by_role=tuple(
            (role, 2 if role == "writer" else limit)
            for role, limit in _policy().max_active_by_role
        ),
        max_concurrency=7,
    )
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, policy)
        root_claim = kb.claim_task(conn, root_id)
        assert root_claim is not None
        child_id = _child(
            conn,
            _authority(root_claim),
            title="write",
            assignee="writer-a",
            role="writer",
        )
        assert kb.assign_task(conn, child_id, "writer-b")
        with pytest.raises(ValueError, match="same role"):
            kb.assign_task(conn, child_id, "reviewer-a")
        child = kb.get_task(conn, child_id)
        assert child is not None
        assert child.assignee == "writer-b"
    finally:
        conn.close()


def test_v3_run_admission_rejects_role_profile_and_digest_mismatch(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        conn.execute("UPDATE tasks SET assignee='writer-a' WHERE id=?", (root_id,))
        assert kb.claim_task(conn, root_id) is None
        conn.execute(
            "UPDATE tasks SET assignee='planner', orchestration_policy_digest=? WHERE id=?",
            ("sha256:" + "0" * 64, root_id),
        )
        assert kb.claim_task(conn, root_id) is None
        assert conn.execute("SELECT COUNT(*) FROM task_runs").fetchone()[0] == 0
    finally:
        conn.close()


def test_v3_per_role_concurrency_and_integrator_serialization(tmp_path):
    policy = _policy(
        role_bindings=_policy().role_bindings + (("writer-b", "writer"),),
        max_concurrency=6,
    )
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, policy)
        root_claim = kb.claim_task(conn, root_id)
        assert root_claim is not None
        authority = _authority(root_claim)
        writer_a = _child(
            conn, authority, title="wa", assignee="writer-a", role="writer"
        )
        writer_b = _child(
            conn, authority, title="wb", assignee="writer-b", role="writer"
        )
        integrator_a = _child(
            conn, authority, title="ia", assignee="integrator-a", role="integrator"
        )
        integrator_b = _child(
            conn, authority, title="ib", assignee="integrator-a", role="integrator"
        )
        assert kb.claim_task(conn, writer_a) is not None
        assert kb.claim_task(conn, writer_b) is None
        assert kb.claim_task(conn, integrator_a) is not None
        assert kb.claim_task(conn, integrator_b) is None
    finally:
        conn.close()


def test_v3_child_policy_downgrade_cannot_bypass_integrator_serialization(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        root_claim = kb.claim_task(conn, root_id)
        assert root_claim is not None
        authority = _authority(root_claim)
        first = _child(
            conn, authority, title="i1", assignee="integrator-a", role="integrator"
        )
        second = _child(
            conn, authority, title="i2", assignee="integrator-a", role="integrator"
        )
        assert kb.claim_task(conn, first) is not None
        legacy = kb.OrchestrationPolicy(
            allowed_assignees=("integrator-a",),
            orchestrator_assignees=("integrator-a",),
            max_depth=2,
            max_tasks=4,
            max_runtime_seconds=60,
        )
        conn.execute(
            "UPDATE tasks SET orchestration_policy=? WHERE id=?",
            (legacy.to_json(), second),
        )
        assert kb.claim_task(conn, second) is None
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM task_runs r "
                "JOIN program_run_admission_evidence e ON e.run_id=r.id "
                "WHERE e.root_id=? AND e.program_role='integrator' "
                "AND r.status='running' AND r.ended_at IS NULL",
                (root_id,),
            ).fetchone()[0]
            == 1
        )
    finally:
        conn.close()


def test_change_request_capacity_allows_replay_but_rejects_new_work(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    monkeypatch.setattr(kb, "PROGRAM_MAX_CHANGE_REQUESTS", 2)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        first = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator",
            idempotency_key="capacity-one",
        )
        kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator",
            idempotency_key="capacity-two",
        )
        replay = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator",
            idempotency_key="capacity-one",
        )
        assert replay["request_id"] == first["request_id"]
        assert replay["replayed"] is True
        with pytest.raises(ValueError, match="history exceeds limit"):
            kb.prepare_program_change_request(
                conn,
                root_id,
                change={"operation": "extend_deadline", "new_deadline": now + 600},
                actor="operator",
                idempotency_key="capacity-three",
            )
        assert (
            conn.execute(
                "SELECT COUNT(*) FROM program_change_requests WHERE root_id=?",
                (root_id,),
            ).fetchone()[0]
            == 2
        )
    finally:
        conn.close()


def test_accepted_scope_change_increments_evidence_generation_once(
    tmp_path, monkeypatch
):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        prepared = kb.prepare_program_change_request(
            conn,
            root_id,
            change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator",
            idempotency_key="prepare-generation",
        )
        kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-generation",
        )
        assert (
            conn.execute(
                "SELECT evidence_generation FROM program_lifecycle WHERE root_id=?",
                (root_id,),
            ).fetchone()[0]
            == 1
        )
        kb.apply_program_change_request(
            conn,
            root_id,
            request_id=prepared["request_id"],
            actor="approver",
            idempotency_key="apply-generation",
        )
        assert (
            conn.execute(
                "SELECT evidence_generation FROM program_lifecycle WHERE root_id=?",
                (root_id,),
            ).fetchone()[0]
            == 1
        )
        conn.execute(
            "UPDATE program_lifecycle SET evidence_generation=0 WHERE root_id=?",
            (root_id,),
        )
        assert kb.claim_task(conn, root_id) is None
    finally:
        conn.close()


def _generation(conn, root_id):
    return conn.execute(
        "SELECT evidence_generation FROM program_lifecycle WHERE root_id=?", (root_id,)
    ).fetchone()[0]


def test_direct_lifecycle_helper_anchors_generation_once_on_replay(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        assert kb.extend_program_deadline(
            conn,
            root_id,
            new_deadline=now + 600,
            actor="operator",
            idempotency_key="direct",
        )["replayed"] is False
        assert _generation(conn, root_id) == 1
        assert kb.extend_program_deadline(
            conn,
            root_id,
            new_deadline=now + 600,
            actor="operator",
            idempotency_key="direct",
        )["replayed"] is True
        assert _generation(conn, root_id) == 1
    finally:
        conn.close()


def test_task_and_run_role_rewrites_fail_closed_against_immutable_evidence(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        root_claim = kb.claim_task(conn, root_id)
        assert root_claim is not None
        authority = _authority(root_claim)
        writer = _child(conn, authority, title="writer", assignee="writer-a", role="writer")
        first = _child(conn, authority, title="i1", assignee="integrator-a", role="integrator")
        second = _child(conn, authority, title="i2", assignee="integrator-a", role="integrator")
        conn.execute(
            "UPDATE tasks SET assignee='integrator-a', program_role='integrator' WHERE id=?",
            (writer,),
        )
        assert kb.claim_task(conn, writer) is None
        claimed = kb.claim_task(conn, first)
        assert claimed is not None
        conn.execute("UPDATE task_runs SET program_role='writer' WHERE id=?", (claimed.current_run_id,))
        assert kb.claim_task(conn, second) is None
    finally:
        conn.close()


def test_same_role_reassignment_appends_versioned_evidence_atomically(tmp_path):
    policy = _policy(
        role_bindings=_policy().role_bindings + (("writer-b", "writer"),),
        max_active_by_role=tuple(
            (role, 2 if role == "writer" else limit)
            for role, limit in _policy().max_active_by_role
        ),
        max_concurrency=7,
    )
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn, policy)
        root_claim = kb.claim_task(conn, root_id)
        child = _child(conn, _authority(root_claim), title="writer", assignee="writer-a", role="writer")
        assert kb.assign_task(conn, child, "writer-b")
        rows = conn.execute(
            "SELECT assignment_version, profile, program_role FROM program_task_role_evidence "
            "WHERE task_id=? ORDER BY assignment_version", (child,),
        ).fetchall()
        assert [tuple(row) for row in rows] == [(1, "writer-a", "writer"), (2, "writer-b", "writer")]
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            conn.execute(
                "UPDATE program_task_role_evidence SET profile='writer-a' WHERE task_id=?", (child,)
            )
    finally:
        conn.close()


def test_stale_orchestrator_generation_and_applied_replay_fail_closed(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        root_claim = kb.claim_task(conn, root_id)
        prepared = kb.prepare_program_change_request(
            conn, root_id, change={"operation": "extend_deadline", "new_deadline": now + 600},
            actor="operator", idempotency_key="stale-prepare",
        )
        kb.apply_program_change_request(
            conn, root_id, request_id=prepared["request_id"], actor="approver",
            idempotency_key="stale-apply",
        )
        with pytest.raises(ValueError, match="generation"):
            _child(conn, _authority(root_claim), title="stale", assignee="writer-a", role="writer")
        for invalid in (0, 2):
            conn.execute(
                "UPDATE program_lifecycle SET evidence_generation=? WHERE root_id=?", (invalid, root_id)
            )
            with pytest.raises(ValueError, match="generation"):
                kb.apply_program_change_request(
                    conn, root_id, request_id=prepared["request_id"], actor="approver",
                    idempotency_key="stale-apply",
                )
    finally:
        conn.close()


def test_cached_connect_rejects_separate_connection_owned_namespace_debris(tmp_path):
    db_path = tmp_path / "kanban.db"
    kb.connect(db_path).close()
    raw = sqlite3.connect(db_path)
    raw.execute("CREATE TABLE program_role_debris(value TEXT)")
    raw.close()
    with pytest.raises(ValueError, match="unsupported program lifecycle schema"):
        kb.connect(db_path)


def test_exact_p2a_program_schema_backfills_existing_applied_generation(tmp_path, monkeypatch):
    now = 1_700_000_000
    monkeypatch.setattr(kb.time, "time", lambda: now)
    db_path = tmp_path / "kanban.db"
    legacy = kb.OrchestrationPolicy(
        allowed_assignees=("planner",),
        orchestrator_assignees=("planner",),
        max_depth=2,
        max_tasks=4,
        max_runtime_seconds=60,
        max_wall_clock_seconds=300,
    )
    conn = kb.connect(db_path)
    root_id = _root(conn, legacy)
    prepared = kb.prepare_program_change_request(
        conn,
        root_id,
        change={"operation": "extend_deadline", "new_deadline": now + 600},
        actor="operator",
        idempotency_key="predecessor-prepare",
    )
    kb.apply_program_change_request(
        conn,
        root_id,
        request_id=prepared["request_id"],
        actor="approver",
        idempotency_key="predecessor-apply",
    )
    conn.close()

    raw = sqlite3.connect(db_path)
    # Reconstruct the genuine P2A predecessor: remove all later delivery and
    # Director successor objects rather than accepting an impossible hybrid.
    raw.execute("DROP TRIGGER trg_task_events_completed_no_update")
    raw.execute("DROP TRIGGER trg_task_events_completed_no_delete")
    for table in (
        "program_delivery_evidence",
        "program_delivery_operations",
        "program_deliveries",
        "program_director_authority",
    ):
        raw.execute(f"DROP TABLE {table}")
    for trigger in (
        "trg_program_task_role_evidence_no_update",
        "trg_program_task_role_evidence_no_delete",
        "trg_program_run_admission_evidence_no_update",
        "trg_program_run_admission_evidence_no_delete",
    ):
        raw.execute(f"DROP TRIGGER {trigger}")
    raw.execute("DROP TABLE program_task_role_evidence")
    raw.execute("DROP TABLE program_run_admission_evidence")
    raw.execute("ALTER TABLE program_lifecycle RENAME TO program_lifecycle_current")
    raw.execute(
        "CREATE TABLE program_lifecycle ("
        "root_id TEXT PRIMARY KEY, root_fingerprint TEXT NOT NULL, "
        "initial_deadline INTEGER NOT NULL, effective_deadline INTEGER NOT NULL, "
        "archived_at INTEGER, archive_actor TEXT, created_at INTEGER NOT NULL, "
        "updated_at INTEGER NOT NULL)"
    )
    raw.execute(
        "INSERT INTO program_lifecycle SELECT root_id, root_fingerprint, initial_deadline, "
        "effective_deadline, archived_at, archive_actor, created_at, updated_at "
        "FROM program_lifecycle_current"
    )
    raw.execute("DROP TABLE program_lifecycle_current")
    raw.commit()
    raw.close()

    upgraded = kb.connect(db_path)
    try:
        assert _generation(upgraded, root_id) == 1
    finally:
        upgraded.close()


def test_orphan_applied_event_with_matching_counter_blocks_v3_admission(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        root_id = _root(conn)
        conn.execute(
            "INSERT INTO program_change_request_events ("
            "request_id, root_id, event_type, actor, idempotency_key, "
            "event_fingerprint, payload_json, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "cr_orphan",
                root_id,
                "applied",
                "attacker",
                "orphan-applied-event",
                "sha256:" + "0" * 64,
                "{}",
                1_700_000_000,
            ),
        )
        conn.execute(
            "UPDATE program_lifecycle SET evidence_generation=1 WHERE root_id=?",
            (root_id,),
        )
        assert kb.claim_task(conn, root_id) is None
    finally:
        conn.close()
