from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from hermes_cli import kanban_db as kb


@pytest.fixture(autouse=True)
def trusted_mission_control(monkeypatch):
    monkeypatch.setattr(kb, "is_mission_control_command_cgroup", lambda: True)


def _policy() -> kb.OrchestrationPolicy:
    return kb.OrchestrationPolicy(
        version=3,
        role_bindings=(("planner", "orchestrator"), ("writer-a", "writer"),
                       ("integrator-a", "integrator"), ("reviewer-a", "reviewer"),
                       ("qa-a", "qa_verifier")),
        max_active_by_role=(("orchestrator", 1), ("writer", 1), ("integrator", 1),
                            ("reviewer", 1), ("security_reviewer", 0), ("qa_verifier", 1)),
        max_depth=2, max_tasks=6, max_runtime_seconds=60,
        max_concurrency=5, max_wall_clock_seconds=300, goal_max_turns=5,
    )


def _authority(task):
    return kb.CreationAuthority(task_id=task.id, run_id=task.current_run_id,
                                claim_lock=task.claim_lock, actor_profile=task.assignee)


def _setup(
    conn: sqlite3.Connection,
    *,
    policy: kb.OrchestrationPolicy | None = None,
    second_integrator: bool = False,
) -> dict[str, Any]:
    root_id = kb.create_task(conn, title="root", assignee="planner",
                             orchestration_policy=policy or _policy())
    root = kb.claim_task(conn, root_id)
    assert root is not None
    writer_id = kb.create_task(
        conn, title="write", assignee="writer-a", program_role="writer",
        current_orchestrator_task_id=root_id, creation_authority=_authority(root),
    )
    integrator_id = kb.create_task(
        conn, title="integrate", assignee="integrator-a", program_role="integrator",
        current_orchestrator_task_id=root_id, creation_authority=_authority(root),
    )
    reviewer_id = kb.create_task(
        conn, title="review", assignee="reviewer-a", program_role="reviewer",
        current_orchestrator_task_id=root_id, creation_authority=_authority(root),
    )
    qa_id = kb.create_task(
        conn, title="qa", assignee="qa-a", program_role="qa_verifier",
        current_orchestrator_task_id=root_id, creation_authority=_authority(root),
    )
    second_integrator_id = None
    if second_integrator:
        second_integrator_id = kb.create_task(
            conn, title="integrate again", assignee="integrator-a",
            program_role="integrator", current_orchestrator_task_id=root_id,
            creation_authority=_authority(root),
        )
    # Delivery role stages, rather than task dependency completion of the root,
    # govern these sibling workers.
    worker_ids = [writer_id, integrator_id, reviewer_id, qa_id]
    if second_integrator_id is not None:
        worker_ids.append(second_integrator_id)
    conn.execute(
        f"DELETE FROM task_links WHERE child_id IN ({','.join('?' for _ in worker_ids)})",
        worker_ids,
    )
    kb.recompute_ready(conn)
    policy_digest = root.orchestration_policy_digest
    assert policy_digest is not None
    director_digest = "d" * 64
    bound = kb.bind_program_director_authority(
        conn, root_id=root_id, policy_digest=policy_digest,
        authority_evidence_generation=0,
        director_authority_digest=director_digest,
    )
    assert not bound["deduplicated"]
    prepared = kb.prepare_program_delivery(conn, {
        "delivery_id": "delivery_" + "1" * 64, "root_id": root_id,
        "project_id": "project-1", "project_binding_digest": "sha256:" + "2" * 64,
        "repository_identity_digest": "sha256:" + "3" * 64, "delivery_mode": "pr",
        "source_control_adapter": "github-v1", "ci_adapter": "github-actions-v1",
        "nonproduction_adapter": "none", "actor_task_id": root_id,
        "actor_run_id": root.current_run_id, "actor_profile": "planner",
        "actor_role": "orchestrator", "policy_digest": policy_digest,
        "authority_evidence_generation": 0, "idempotency_key": "delivery-prepare",
        "director_authority_digest": director_digest,
    })
    return {"root_id": root_id, "root": root, "writer_id": writer_id,
            "integrator_id": integrator_id, "reviewer_id": reviewer_id, "qa_id": qa_id,
            "second_integrator_id": second_integrator_id,
            "director": director_digest,
            "digest": policy_digest, "delivery": prepared}


def _actor(setup: dict[str, Any], task, role: str, expected: int) -> dict[str, Any]:
    return {"delivery_id": setup["delivery"]["delivery_id"], "expected_version": expected,
            "actor_task_id": task.id, "actor_run_id": task.current_run_id,
            "actor_profile": task.assignee, "actor_role": role,
            "policy_digest": setup["digest"], "authority_evidence_generation": 0,
            "director_authority_digest": setup["director"]}


def _to_review(conn: sqlite3.Connection, setup: dict[str, Any]):
    root = setup["root"]
    kb.transition_program_delivery(conn, {**_actor(setup, root, "orchestrator", 1),
        "from_state": "plan", "to_state": "writers", "idempotency_key": "to-writers"})
    writer = kb.claim_task(conn, setup["writer_id"])
    assert writer is not None
    assert kb.complete_task(conn, setup["writer_id"], result="written",
                            expected_run_id=writer.current_run_id)
    kb.transition_program_delivery(conn, {**_actor(setup, root, "orchestrator", 2),
        "from_state": "writers", "to_state": "integration", "idempotency_key": "to-integration"})
    integrator = kb.claim_task(conn, setup["integrator_id"])
    assert integrator is not None
    result = kb.transition_program_delivery(conn, {**_actor(setup, integrator, "integrator", 3),
        "from_state": "integration", "to_state": "review",
        "integration_task_id": setup["integrator_id"], "base_sha": "a" * 40,
        "candidate_sha": "b" * 40, "candidate_digest": "sha256:" + "4" * 64,
        "required_check_policy_digest": "sha256:" + "5" * 64,
        "idempotency_key": "candidate-bind"})
    return integrator, result


def _new_role_task(conn, setup, profile, role, title):
    root_id = setup["root_id"]
    # The root run was closed after delegation; use the immutable root task as
    # parent through a fresh admitted orchestrator run.
    root = kb.get_task(conn, root_id)
    assert root is not None
    conn.execute("UPDATE tasks SET status='ready', current_run_id=NULL WHERE id=?", (root_id,))
    root = kb.claim_task(conn, root_id)
    assert root is not None
    task_id = kb.create_task(conn, title=title, assignee=profile, program_role=role,
        current_orchestrator_task_id=root_id, creation_authority=_authority(root))
    assert kb.complete_task(conn, root_id, result="delegated", expected_run_id=root.current_run_id)
    task = kb.claim_task(conn, task_id)
    assert task is not None
    return task


def test_director_binding_is_opaque_immutable_replay_safe_and_generation_bound(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn)
        replay = kb.bind_program_director_authority(
            conn, root_id=setup["root_id"], policy_digest=setup["digest"],
            authority_evidence_generation=0,
            director_authority_digest=setup["director"])
        assert replay["deduplicated"]
        for digest in ("Director", "A" * 64, "e" * 63):
            with pytest.raises(ValueError, match="director"):
                kb.bind_program_director_authority(
                    conn, root_id=setup["root_id"], policy_digest=setup["digest"],
                    authority_evidence_generation=0, director_authority_digest=digest)
        with pytest.raises(ValueError, match="mismatch|drift"):
            kb.bind_program_director_authority(
                conn, root_id=setup["root_id"], policy_digest=setup["digest"],
                authority_evidence_generation=0, director_authority_digest="e" * 64)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("UPDATE program_director_authority SET director_authority_digest=?",
                         ("e" * 64,))
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("DELETE FROM program_director_authority")
    finally:
        conn.close()


def test_every_delivery_mutator_rejects_missing_or_wrong_director(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn)
        request = {**_actor(setup, setup["root"], "orchestrator", 1),
                   "from_state": "plan", "to_state": "writers", "idempotency_key": "x"}
        request.pop("director_authority_digest")
        with pytest.raises(ValueError, match="director"):
            kb.transition_program_delivery(conn, request)
        with pytest.raises(ValueError, match="director"):
            kb.transition_program_delivery(conn, {**request, "director_authority_digest": "e" * 64})
    finally:
        conn.close()


def test_managed_completion_requires_active_exact_admitted_run_and_root_candidate(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn)
        with pytest.raises(ValueError, match="expected_run_id"):
            kb.complete_task(conn, setup["writer_id"], result="manual")
        with pytest.raises(ValueError, match="candidate"):
            kb.complete_task(conn, setup["root_id"], result="early",
                             expected_run_id=setup["root"].current_run_id)
        # Unmanaged compatibility remains intentionally permissive.
        plain = kb.create_task(conn, title="plain", assignee="legacy")
        assert kb.complete_task(conn, plain, result="manual")
    finally:
        conn.close()


def test_generic_review_claim_fails_closed_for_v3_but_legacy_is_unchanged(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn)
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (setup["writer_id"],))
        assert kb.claim_review_task(conn, setup["writer_id"]) is None
        plain = kb.create_task(conn, title="legacy", assignee="legacy")
        conn.execute("UPDATE tasks SET status='review' WHERE id=?", (plain,))
        claimed = kb.claim_review_task(conn, plain)
        assert claimed is not None and claimed.assignee == "legacy"
    finally:
        conn.close()


def _prepare_operation(conn, setup, integrator, expected=4, *, approve=True, execute=True):
    if approve:
        reviewer = kb.claim_task(conn, setup["reviewer_id"])
        assert reviewer is not None
        kb.record_program_delivery_review(conn, {
            **_actor(setup, reviewer, "reviewer", expected), "approval_code": "APPROVED",
            "approval_digest": "sha256:" + "9" * 64, "idempotency_key": "auto-review"})
        expected += 1
    adapter_request = {
        "candidate_sha": "b" * 40,
        "candidate_digest": "sha256:" + "4" * 64,
        "project_id": "project-1",
        "project_binding_digest": "sha256:" + "2" * 64,
    }
    request = {**_actor(setup, integrator, "integrator", expected), "phase": "pr",
        "attempt_no": 1, "operation_id": "op_" + "6" * 64, "adapter_id": "github-v1",
        "method": "createPullRequest", "request": adapter_request,
        "approval_binding_digest": "sha256:" + "8" * 64,
        "idempotency_key": "prepare-pr"}
    request["request_digest"] = kb._sha256_json({
        "delivery_id": request["delivery_id"],
        "candidate_generation": 1,
        "candidate_sha": adapter_request["candidate_sha"],
        "candidate_digest": adapter_request["candidate_digest"],
        "project_id": adapter_request["project_id"],
        "project_binding_digest": adapter_request["project_binding_digest"],
        "phase": request["phase"],
        "attempt_no": request["attempt_no"],
        "adapter_id": request["adapter_id"],
        "method": request["method"],
        "approval_binding_digest": request["approval_binding_digest"],
        "adapter_request": adapter_request,
    })[1]
    return request, (kb.prepare_program_delivery_operation(conn, request) if execute else None)


def test_review_to_pr_claim_and_confirm_is_one_cas_evidence_chain(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn)
        integrator, review = _to_review(conn, setup)
        assert review["state"] == "review" and review["candidate_generation"] == 1
        op_request, prepared = _prepare_operation(conn, setup, integrator)
        claimed = kb.claim_program_delivery_operation(conn, {
            **_actor(setup, integrator, "integrator", 6), "operation_id": op_request["operation_id"],
            "request_digest": op_request["request_digest"], "idempotency_key": "claim-pr"})
        started = kb.mark_program_delivery_dispatch_started(conn, {
            **_actor(setup, integrator, "integrator", 7), "operation_id": op_request["operation_id"],
            "request_digest": op_request["request_digest"], "idempotency_key": "dispatch-pr"})
        remote_result = {"number": 17}
        settled_request = {**_actor(setup, integrator, "integrator", 8),
            "operation_id": op_request["operation_id"], "request_digest": op_request["request_digest"],
            "outcome": "confirmed", "outcome_code": "PR_CREATED", "result": remote_result,
            "result_digest": kb._sha256_json(remote_result)[1], "idempotency_key": "settle-pr"}
        settled = kb.settle_program_delivery_operation(conn, settled_request)
        assert [prepared["version"], claimed["version"], started["version"], settled["version"]] == [6, 7, 8, 9]
        assert settled["state"] == "pr" and settled["operation_status"] == "confirmed"
        evidence = conn.execute("SELECT resulting_version FROM program_delivery_evidence WHERE delivery_id=? ORDER BY id",
                                (setup["delivery"]["delivery_id"],)).fetchall()
        assert [row[0] for row in evidence] == list(range(1, 10))
    finally:
        conn.close()


def test_same_operation_replay_does_not_advance_version_or_evidence(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, first = _prepare_operation(conn, setup, integrator)
        before = conn.execute("SELECT COUNT(*) FROM program_delivery_evidence").fetchone()[0]
        replay = kb.prepare_program_delivery_operation(conn, request)
        assert replay == {**first, "deduplicated": True}
        assert conn.execute("SELECT COUNT(*) FROM program_delivery_evidence").fetchone()[0] == before
    finally:
        conn.close()


def test_policy_or_generation_drift_rejects_before_claim(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        conn.execute("UPDATE program_lifecycle SET evidence_generation=1 WHERE root_id=?", (setup["root_id"],))
        before = conn.execute("SELECT version FROM program_deliveries").fetchone()[0]
        with pytest.raises(ValueError, match="generation"):
            kb.claim_program_delivery_operation(conn, {**_actor(setup, integrator, "integrator", 6),
                "operation_id": request["operation_id"], "request_digest": request["request_digest"],
                "idempotency_key": "drift-claim"})
        assert conn.execute("SELECT version FROM program_deliveries").fetchone()[0] == before
        assert conn.execute("SELECT status FROM program_delivery_operations").fetchone()[0] == "prepared"
    finally:
        conn.close()


def test_operation_request_mismatch_rejects_without_mutation(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        with pytest.raises(ValueError, match="request.*match"):
            kb.claim_program_delivery_operation(conn, {**_actor(setup, integrator, "integrator", 6),
                "operation_id": request["operation_id"], "request_digest": "sha256:" + "0" * 64,
                "idempotency_key": "bad-claim"})
    finally:
        conn.close()


def test_operation_candidate_sha_mismatch_rejects_before_prepare(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        reviewer = kb.claim_task(conn, setup["reviewer_id"]); assert reviewer is not None
        kb.record_program_delivery_review(conn, {
            **_actor(setup, reviewer, "reviewer", 4), "approval_code": "APPROVED",
            "approval_digest": "sha256:" + "9" * 64,
            "idempotency_key": "candidate-test-review"})
        adapter_request = {"candidate_sha": "c" * 40}
        with pytest.raises(ValueError, match="candidate SHA.*match"):
            kb.prepare_program_delivery_operation(conn, {
                **_actor(setup, integrator, "integrator", 5), "phase": "pr", "attempt_no": 1,
                "operation_id": "op_" + "c" * 64, "adapter_id": "github-v1",
                "method": "createPullRequest", "request": adapter_request,
                "request_digest": kb._sha256_json(adapter_request)[1],
                "approval_binding_digest": "sha256:" + "8" * 64,
                "idempotency_key": "candidate-mismatch"})
        assert conn.execute("SELECT COUNT(*) FROM program_delivery_operations").fetchone()[0] == 0
    finally:
        conn.close()


def test_adapter_request_requires_entire_candidate_project_tuple(tmp_path):
    conn = kb.connect(tmp_path / "missing-adapter-binding.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        reviewer = kb.claim_task(conn, setup["reviewer_id"]); assert reviewer is not None
        kb.record_program_delivery_review(conn, {
            **_actor(setup, reviewer, "reviewer", 4), "approval_code": "APPROVED",
            "approval_digest": "sha256:" + "9" * 64,
            "idempotency_key": "missing-tuple-review"})
        adapter_request = {"candidate_sha": "b" * 40}
        request = {**_actor(setup, integrator, "integrator", 5), "phase": "pr",
            "attempt_no": 1, "operation_id": "op_" + "d" * 64,
            "adapter_id": "github-v1", "method": "createPullRequest",
            "request": adapter_request, "request_digest": "sha256:" + "0" * 64,
            "approval_binding_digest": "sha256:" + "8" * 64,
            "idempotency_key": "missing-tuple"}
        with pytest.raises(ValueError, match="candidate_digest.*match"):
            kb.prepare_program_delivery_operation(conn, request)
        assert conn.execute("SELECT COUNT(*) FROM program_delivery_operations").fetchone()[0] == 0
    finally:
        conn.close()


def test_operation_payload_rejects_nested_raw_body(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        reviewer = kb.claim_task(conn, setup["reviewer_id"]); assert reviewer is not None
        kb.record_program_delivery_review(conn, {
            **_actor(setup, reviewer, "reviewer", 4), "approval_code": "APPROVED",
            "approval_digest": "sha256:" + "9" * 64,
            "idempotency_key": "payload-test-review"})
        adapter_request = {"candidate_sha": "b" * 40, "metadata": {"body": "secret"}}
        with pytest.raises(ValueError, match="forbidden raw material"):
            kb.prepare_program_delivery_operation(conn, {
                **_actor(setup, integrator, "integrator", 5), "phase": "pr", "attempt_no": 1,
                "operation_id": "op_" + "a" * 64, "adapter_id": "github-v1",
                "method": "createPullRequest", "request": adapter_request,
                "request_digest": kb._sha256_json(adapter_request)[1],
                "approval_binding_digest": "sha256:" + "8" * 64,
                "idempotency_key": "nested-raw"})
    finally:
        conn.close()


@pytest.mark.parametrize("bad_key", [
    "repo_path", "repositoryPath", "rawOutput", "response-body",
    "callback.url", "nestedRepoURL",
])
def test_operation_payload_rejects_canonicalized_raw_material_aliases(tmp_path, bad_key):
    conn = kb.connect(tmp_path / f"raw-{bad_key.replace('.', '-')}.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        reviewer = kb.claim_task(conn, setup["reviewer_id"]); assert reviewer is not None
        kb.record_program_delivery_review(conn, {
            **_actor(setup, reviewer, "reviewer", 4), "approval_code": "APPROVED",
            "approval_digest": "sha256:" + "9" * 64,
            "idempotency_key": "raw-alias-review"})
        adapter_request = {
            "candidate_sha": "b" * 40,
            "candidate_digest": "sha256:" + "4" * 64,
            "project_id": "project-1",
            "project_binding_digest": "sha256:" + "2" * 64,
            "metadata": {"nested": [{bad_key: "SECRET"}]},
        }
        request = {**_actor(setup, integrator, "integrator", 5), "phase": "pr",
            "attempt_no": 1, "operation_id": "op_" + "e" * 64,
            "adapter_id": "github-v1", "method": "createPullRequest",
            "request": adapter_request,
            "approval_binding_digest": "sha256:" + "8" * 64,
            "idempotency_key": "raw-alias", "request_digest": "sha256:" + "0" * 64}
        with pytest.raises(ValueError, match="forbidden raw material"):
            kb.prepare_program_delivery_operation(conn, request)
        assert conn.execute("SELECT COUNT(*) FROM program_delivery_operations").fetchone()[0] == 0
    finally:
        conn.close()


def test_sealed_digest_keys_are_allowed_in_bounded_operation_json(tmp_path):
    conn = kb.connect(tmp_path / "sealed-digests.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        _, prepared = _prepare_operation(conn, setup, integrator)
        assert prepared["operation_status"] == "prepared"
        stored = conn.execute("SELECT request_json FROM program_delivery_operations").fetchone()[0]
        assert len(stored.encode()) <= 4096
        assert "project_binding_digest" in stored
    finally:
        conn.close()


def test_dispatch_and_settle_require_exact_claimed_integrator_identity(tmp_path):
    conn = kb.connect(tmp_path / "actor-binding.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        claim_request = {**_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "claim-bound"}
        kb.claim_program_delivery_operation(conn, claim_request)
        reviewer = kb.get_task(conn, setup["reviewer_id"]); assert reviewer is not None
        wrong_dispatch = {**_actor(setup, reviewer, "reviewer", 7),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "wrong-dispatch"}
        with pytest.raises(ValueError, match="claimed integrator identity"):
            kb.mark_program_delivery_dispatch_started(conn, wrong_dispatch)
        assert conn.execute("SELECT status FROM program_delivery_operations").fetchone()[0] == "claimed"
        dispatch = kb.mark_program_delivery_dispatch_started(conn, {
            **claim_request, "expected_version": 7, "idempotency_key": "dispatch-bound"})
        assert dispatch["operation_status"] == "dispatch_started"
        remote = {"number": 21}
        wrong_settle = {**_actor(setup, reviewer, "reviewer", 8),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "outcome": "confirmed", "outcome_code": "PR_CREATED", "result": remote,
            "result_digest": kb._sha256_json(remote)[1], "idempotency_key": "wrong-settle"}
        with pytest.raises(ValueError, match="claimed integrator identity"):
            kb.settle_program_delivery_operation(conn, wrong_settle)
        assert conn.execute("SELECT status FROM program_delivery_operations").fetchone()[0] == "dispatch_started"
    finally:
        conn.close()


def test_qa_cannot_dispatch_claimed_integrator_operation(tmp_path):
    conn = kb.connect(tmp_path / "qa-dispatch.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        kb.claim_program_delivery_operation(conn, {
            **_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "qa-race-claim"})
        qa = kb.claim_task(conn, setup["qa_id"])
        assert qa is not None
        with pytest.raises(ValueError, match="claimed integrator identity"):
            kb.mark_program_delivery_dispatch_started(conn, {
                **_actor(setup, qa, "qa_verifier", 7),
                "operation_id": request["operation_id"],
                "request_digest": request["request_digest"],
                "idempotency_key": "qa-dispatch"})
        assert conn.execute("SELECT status FROM program_delivery_operations").fetchone()[0] == "claimed"
    finally:
        conn.close()


def test_second_integrator_cannot_settle_first_integrators_operation(tmp_path):
    conn = kb.connect(tmp_path / "second-integrator.db")
    try:
        setup = _setup(conn, second_integrator=True)
        integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        kb.claim_program_delivery_operation(conn, {
            **_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "first-integrator-claim"})
        kb.mark_program_delivery_dispatch_started(conn, {
            **_actor(setup, integrator, "integrator", 7),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "first-integrator-dispatch"})
        assert kb.complete_task(
            conn, integrator.id, result="handoff forbidden",
            expected_run_id=integrator.current_run_id,
        )
        second_id = setup["second_integrator_id"]
        assert second_id is not None
        second = kb.claim_task(conn, second_id)
        assert second is not None
        result = {"number": 23}
        with pytest.raises(ValueError, match="claimed integrator identity"):
            kb.settle_program_delivery_operation(conn, {
                **_actor(setup, second, "integrator", 8),
                "operation_id": request["operation_id"],
                "request_digest": request["request_digest"],
                "outcome": "confirmed", "outcome_code": "PR_CREATED", "result": result,
                "result_digest": kb._sha256_json(result)[1],
                "idempotency_key": "second-integrator-settle"})
        assert conn.execute("SELECT status FROM program_delivery_operations").fetchone()[0] == "dispatch_started"
    finally:
        conn.close()


def test_claim_replay_is_bound_and_racing_wrong_role_cannot_steal(tmp_path):
    conn = kb.connect(tmp_path / "claim-race.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        claim = {**_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "claim-race"}
        first = kb.claim_program_delivery_operation(conn, claim)
        replay = kb.claim_program_delivery_operation(conn, claim)
        assert replay == {**first, "deduplicated": True}
        reviewer = kb.get_task(conn, setup["reviewer_id"]); assert reviewer is not None
        with pytest.raises(ValueError, match="integrator|idempotency"):
            kb.claim_program_delivery_operation(conn, {
                **claim, **_actor(setup, reviewer, "reviewer", 7)})
        assert conn.execute(
            "SELECT COUNT(*) FROM program_delivery_evidence WHERE event_type='operation_claimed'"
        ).fetchone()[0] == 1
    finally:
        conn.close()


def test_operation_digest_binds_exact_delivery_and_adapter_tuple(tmp_path):
    conn = kb.connect(tmp_path / "operation-binding.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        operation = conn.execute("SELECT * FROM program_delivery_operations").fetchone()
        assert operation["request_digest"] == request["request_digest"]
        assert operation["request_digest"] != kb._sha256_json(request["request"])[1]
        mismatched = dict(request)
        mismatched["operation_id"] = "op_" + "f" * 64
        mismatched["attempt_no"] = 2
        mismatched["expected_version"] = 6
        mismatched["idempotency_key"] = "tuple-mismatch"
        mismatched["request"] = {**request["request"], "project_id": "other-project"}
        with pytest.raises(ValueError, match="project_id.*match"):
            kb.prepare_program_delivery_operation(conn, mismatched)
        assert conn.execute("SELECT COUNT(*) FROM program_delivery_operations").fetchone()[0] == 1
    finally:
        conn.close()


@pytest.mark.parametrize("field,value", [
    ("status", "clean_failed"), ("claimed_by_task_id", "attacker"),
    ("claimed_by_run_id", 999999), ("claimed_by_profile", "attacker"),
    ("claimed_by_role", "reviewer"), ("outcome_code", "FORGED"),
    ("result_digest", "sha256:" + "0" * 64), ("claimed_at", 1),
    ("dispatch_started_at", 1), ("settled_at", 1),
])
def test_operation_projection_tamper_fails_before_replay_or_mutation(tmp_path, field, value):
    conn = kb.connect(tmp_path / f"tamper-{field}.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        claim = {**_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "tamper-claim"}
        kb.claim_program_delivery_operation(conn, claim)
        conn.execute(f"UPDATE program_delivery_operations SET {field}=?", (value,))
        before = conn.execute("SELECT version FROM program_deliveries").fetchone()[0]
        with pytest.raises(ValueError, match="operation history integrity"):
            kb.claim_program_delivery_operation(conn, claim)
        assert conn.execute("SELECT version FROM program_deliveries").fetchone()[0] == before
    finally:
        conn.close()


def test_operation_evidence_snapshots_are_bounded_and_exclude_raw_result(tmp_path):
    conn = kb.connect(tmp_path / "snapshots.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        kb.claim_program_delivery_operation(conn, {**_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "snapshot-claim"})
        kb.mark_program_delivery_dispatch_started(conn, {**_actor(setup, integrator, "integrator", 7),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "idempotency_key": "snapshot-dispatch"})
        result = {"number": 42, "result_digest": "safe-id"}
        kb.settle_program_delivery_operation(conn, {**_actor(setup, integrator, "integrator", 8),
            "operation_id": request["operation_id"], "request_digest": request["request_digest"],
            "outcome": "confirmed", "outcome_code": "PR_CREATED", "result": result,
            "result_digest": kb._sha256_json(result)[1], "idempotency_key": "snapshot-settle"})
        payloads = [json.loads(row[0]) for row in conn.execute(
            "SELECT payload_json FROM program_delivery_evidence WHERE operation_id=? ORDER BY id",
            (request["operation_id"],))]
        assert [item["operation"]["status"] for item in payloads] == [
            "prepared", "claimed", "dispatch_started", "confirmed"]
        assert all(len(kb._sha256_json(item)[0].encode()) <= 4096 for item in payloads)
        assert all("result_json" not in item["operation"] for item in payloads)
        assert all(item["operation"].get("result") is None for item in payloads)
    finally:
        conn.close()


def test_operation_state_race_and_append_only_triggers(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        with pytest.raises(sqlite3.IntegrityError, match="status"):
            conn.execute("UPDATE program_delivery_operations SET status='confirmed' WHERE operation_id=?", (request["operation_id"],))
        for sql in ("UPDATE program_delivery_evidence SET payload_json='{}'", "DELETE FROM program_delivery_evidence",
                    "DELETE FROM program_delivery_operations"):
            with pytest.raises(sqlite3.IntegrityError):
                conn.execute(sql)
    finally:
        conn.close()


def test_completed_task_events_are_immutable_but_appends_remain_allowed(tmp_path):
    conn = kb.connect(tmp_path / "completed-events.db")
    try:
        task_id = kb.create_task(conn, title="event", assignee="legacy")
        conn.execute(
            "INSERT INTO task_events(task_id,run_id,kind,payload,created_at) VALUES (?,?,?,?,?)",
            (task_id, None, "completed", "{}", 1),
        )
        completed_id = conn.execute(
            "SELECT MAX(id) FROM task_events WHERE kind='completed'"
        ).fetchone()[0]
        for sql in (
            "UPDATE task_events SET kind='note' WHERE id=?",
            "UPDATE task_events SET payload='forged' WHERE id=?",
            "DELETE FROM task_events WHERE id=?",
        ):
            with pytest.raises(sqlite3.IntegrityError, match="completed task event evidence"):
                conn.execute(sql, (completed_id,))
        note_id = conn.execute(
            "INSERT INTO task_events(task_id,run_id,kind,payload,created_at) VALUES (?,?,?,?,?)",
            (task_id, None, "note", None, 2),
        ).lastrowid
        with pytest.raises(sqlite3.IntegrityError, match="completed task event evidence"):
            conn.execute("UPDATE task_events SET kind='completed' WHERE id=?", (note_id,))
        assert conn.execute("SELECT COUNT(*) FROM task_events").fetchone()[0] >= 2
    finally:
        conn.close()


@pytest.mark.parametrize("mode", ["missing", "weakened"])
def test_reconnect_rejects_missing_or_weakened_completed_event_trigger(tmp_path, mode):
    db_path = tmp_path / f"event-trigger-{mode}.db"
    kb.connect(db_path).close()
    raw = sqlite3.connect(db_path)
    raw.execute("DROP TRIGGER trg_task_events_completed_no_update")
    if mode == "weakened":
        raw.execute(
            "CREATE TRIGGER trg_task_events_completed_no_update BEFORE UPDATE ON task_events "
            "BEGIN SELECT 1; END"
        )
    raw.commit(); raw.close()
    with pytest.raises(ValueError, match="unsupported program lifecycle schema"):
        kb.connect(db_path)


def test_exact_pre_trigger_schema_reconnect_adds_both_completed_event_guards(tmp_path):
    db_path = tmp_path / "pre-completed-trigger.db"
    kb.connect(db_path).close()
    raw = sqlite3.connect(db_path)
    raw.execute("DROP TRIGGER trg_task_events_completed_no_update")
    raw.execute("DROP TRIGGER trg_task_events_completed_no_delete")
    raw.commit(); raw.close()
    upgraded = kb.connect(db_path)
    try:
        names = {row[0] for row in upgraded.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger' AND name GLOB 'trg_task_events_completed_*'"
        )}
        assert names == {
            "trg_task_events_completed_no_update",
            "trg_task_events_completed_no_delete",
        }
    finally:
        upgraded.close()


def _p2e_schema(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.executescript(kb.SCHEMA_SQL)
        conn.execute("DROP TABLE program_director_authority")
        conn.execute("DROP TRIGGER trg_task_events_completed_no_update")
        conn.execute("DROP TRIGGER trg_task_events_completed_no_delete")
        conn.commit()
    finally:
        conn.close()


def test_exact_p2e_predecessor_reconnect_migrates_and_near_shape_rejects(tmp_path):
    predecessor = tmp_path / "p2e.db"
    _p2e_schema(predecessor)
    conn = kb.connect(predecessor); conn.close()
    conn = sqlite3.connect(predecessor)
    assert conn.execute("SELECT name FROM sqlite_master WHERE name='program_deliveries'").fetchone()
    conn.close()

    near = tmp_path / "near.db"; _p2e_schema(near)
    conn = sqlite3.connect(near)
    conn.execute("DROP TRIGGER trg_program_task_role_evidence_no_update")
    conn.execute("CREATE TRIGGER trg_program_task_role_evidence_no_update BEFORE UPDATE ON program_task_role_evidence BEGIN SELECT 1; END")
    conn.commit(); conn.close()
    with pytest.raises(ValueError, match="unsupported program lifecycle schema"):
        kb.connect(near)


def test_impossible_p2a_delivery_hybrid_is_rejected(tmp_path):
    hybrid = tmp_path / "p2a-delivery-hybrid.db"
    conn = sqlite3.connect(hybrid)
    conn.executescript(kb.SCHEMA_SQL)
    conn.execute("DROP TABLE program_director_authority")
    for trigger in (
        "trg_program_task_role_evidence_no_update",
        "trg_program_task_role_evidence_no_delete",
        "trg_program_run_admission_evidence_no_update",
        "trg_program_run_admission_evidence_no_delete",
        "trg_task_events_completed_no_update",
        "trg_task_events_completed_no_delete",
    ):
        conn.execute(f"DROP TRIGGER {trigger}")
    conn.execute("DROP TABLE program_task_role_evidence")
    conn.execute("DROP TABLE program_run_admission_evidence")
    conn.execute("ALTER TABLE program_lifecycle RENAME TO program_lifecycle_current")
    conn.execute(
        "CREATE TABLE program_lifecycle (root_id TEXT PRIMARY KEY, "
        "root_fingerprint TEXT NOT NULL, initial_deadline INTEGER NOT NULL, "
        "effective_deadline INTEGER NOT NULL, archived_at INTEGER, "
        "archive_actor TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL)"
    )
    conn.execute("DROP TABLE program_lifecycle_current")
    conn.commit(); conn.close()
    with pytest.raises(ValueError, match="unsupported program lifecycle schema"):
        kb.connect(hybrid)


def test_v2_and_unmanaged_roots_have_no_delivery_authority(tmp_path):
    conn = kb.connect(tmp_path / "kanban.db")
    try:
        unmanaged = kb.create_task(conn, title="plain", assignee="planner")
        legacy = kb.create_task(conn, title="v2", assignee="planner", orchestration_policy=kb.OrchestrationPolicy(
            allowed_assignees=("planner",), orchestrator_assignees=("planner",), max_depth=1,
            max_tasks=2, max_runtime_seconds=60))
        for root_id in (unmanaged, legacy):
            with pytest.raises(ValueError, match="role_policy_required"):
                kb.prepare_program_delivery(conn, {"root_id": root_id})
        assert conn.execute("SELECT COUNT(*) FROM program_deliveries").fetchone()[0] == 0
    finally:
        conn.close()


def test_writers_to_integration_rejects_missing_completed_writer_evidence(tmp_path):
    conn = kb.connect(tmp_path / "writer-gate.db")
    try:
        setup = _setup(conn); root = setup["root"]
        kb.transition_program_delivery(conn, {**_actor(setup, root, "orchestrator", 1),
            "from_state": "plan", "to_state": "writers", "idempotency_key": "writers"})
        with pytest.raises(ValueError, match="writer.*completed"):
            kb.transition_program_delivery(conn, {**_actor(setup, root, "orchestrator", 2),
                "from_state": "writers", "to_state": "integration", "idempotency_key": "skip"})
    finally:
        conn.close()


def test_review_approval_and_exact_ci_qa_chain(tmp_path):
    conn = kb.connect(tmp_path / "chain.db")
    try:
        setup = _setup(conn); integrator, _ = _to_review(conn, setup)
        with pytest.raises(ValueError, match="review approval"):
            _prepare_operation(conn, setup, integrator, approve=False)
        reviewer = kb.claim_task(conn, setup["reviewer_id"]); assert reviewer is not None
        approved = kb.record_program_delivery_review(conn, {
            **_actor(setup, reviewer, "reviewer", 4), "approval_code": "APPROVED",
            "approval_digest": "sha256:" + "9" * 64, "idempotency_key": "approved"})
        assert approved["version"] == 5 and approved["state"] == "review"
        op_request, _ = _prepare_operation(conn, setup, integrator, expected=5, approve=False)
        kb.claim_program_delivery_operation(conn, {**_actor(setup, integrator, "integrator", 6),
            "operation_id": op_request["operation_id"], "request_digest": op_request["request_digest"],
            "idempotency_key": "claim"})
        kb.mark_program_delivery_dispatch_started(conn, {**_actor(setup, integrator, "integrator", 7),
            "operation_id": op_request["operation_id"], "request_digest": op_request["request_digest"],
            "idempotency_key": "dispatch"})
        remote = {"number": 17}
        kb.settle_program_delivery_operation(conn, {**_actor(setup, integrator, "integrator", 8),
            "operation_id": op_request["operation_id"], "request_digest": op_request["request_digest"],
            "outcome": "confirmed", "outcome_code": "PR_CREATED", "result": remote,
            "result_digest": kb._sha256_json(remote)[1], "idempotency_key": "settle"})
        kb.transition_program_delivery(conn, {**_actor(setup, integrator, "integrator", 9),
            "from_state": "pr", "to_state": "ci", "evidence_code": "CI_VERIFIED",
            "stage_evidence_digest": "sha256:" + "a" * 64, "idempotency_key": "ci"})
        qa = kb.claim_task(conn, setup["qa_id"]); assert qa is not None
        for expected, source, target, code, digit in (
            (10, "ci", "nonprod", "NONPROD_STARTED", "b"),
            (11, "nonprod", "e2e", "E2E_PASSED", "c"),
            (12, "e2e", "candidate", "CANDIDATE", "d"),
        ):
            final = kb.transition_program_delivery(conn, {**_actor(setup, qa, "qa_verifier", expected),
                "from_state": source, "to_state": target, "evidence_code": code,
                "stage_evidence_digest": "sha256:" + digit * 64, "idempotency_key": target})
        assert final["state"] == "candidate"
        assert len({integrator.assignee, reviewer.assignee, qa.assignee, "writer-a"}) == 4
    finally:
        conn.close()


def test_managed_completion_rechecks_admitted_run_inside_write_transaction(
    tmp_path, monkeypatch,
):
    db_path = tmp_path / "completion-race.db"
    conn = kb.connect(db_path)
    try:
        setup = _setup(conn)
        root = setup["root"]
        kb.transition_program_delivery(conn, {
            **_actor(setup, root, "orchestrator", 1),
            "from_state": "plan", "to_state": "writers",
            "idempotency_key": "completion-race-writers",
        })
        writer = kb.claim_task(conn, setup["writer_id"])
        assert writer is not None and writer.current_run_id is not None

        original_write_txn = kb.write_txn
        raced = False

        @contextmanager
        def racing_write_txn(target):
            nonlocal raced
            if not raced:
                raced = True
                other = sqlite3.connect(db_path)
                try:
                    other.execute(
                        "UPDATE task_runs SET status='reclaimed', outcome='reclaimed', "
                        "ended_at=123456 WHERE id=?",
                        (writer.current_run_id,),
                    )
                    other.commit()
                finally:
                    other.close()
            with original_write_txn(target):
                yield

        monkeypatch.setattr(kb, "write_txn", racing_write_txn)
        with pytest.raises(ValueError, match="active|admitted|run"):
            kb.complete_task(
                conn, setup["writer_id"], result="must not complete",
                expected_run_id=writer.current_run_id,
            )
        task = kb.get_task(conn, setup["writer_id"])
        assert task is not None and task.status == "running"
        assert conn.execute(
            "SELECT COUNT(*) FROM task_events WHERE task_id=? AND kind='completed'",
            (setup["writer_id"],),
        ).fetchone()[0] == 0
    finally:
        conn.close()


def test_ambiguous_operation_blocks_new_attempt_until_explicit_reconcile(tmp_path):
    conn = kb.connect(tmp_path / "ambiguous-retry.db")
    try:
        setup = _setup(conn)
        integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator)
        kb.claim_program_delivery_operation(conn, {
            **_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"],
            "request_digest": request["request_digest"],
            "idempotency_key": "ambiguous-claim",
        })
        kb.mark_program_delivery_dispatch_started(conn, {
            **_actor(setup, integrator, "integrator", 7),
            "operation_id": request["operation_id"],
            "request_digest": request["request_digest"],
            "idempotency_key": "ambiguous-dispatch",
        })
        result = {"observation": "unknown"}
        ambiguous = kb.settle_program_delivery_operation(conn, {
            **_actor(setup, integrator, "integrator", 8),
            "operation_id": request["operation_id"],
            "request_digest": request["request_digest"],
            "outcome": "ambiguous", "outcome_code": "REMOTE_UNKNOWN",
            "result": result, "result_digest": kb._sha256_json(result)[1],
            "idempotency_key": "ambiguous-settle",
        })
        reconcile_phase = conn.execute(
            "SELECT reconcile_phase FROM program_deliveries WHERE delivery_id=?",
            (setup["delivery"]["delivery_id"],),
        ).fetchone()[0]
        assert reconcile_phase == "pr"

        retry = dict(request)
        retry.update({
            "operation_id": "op_" + "7" * 64,
            "attempt_no": 2,
            "expected_version": ambiguous["version"],
            "idempotency_key": "blind-retry",
        })
        retry["request_digest"] = kb._sha256_json({
            "delivery_id": retry["delivery_id"],
            "candidate_generation": 1,
            "candidate_sha": retry["request"]["candidate_sha"],
            "candidate_digest": retry["request"]["candidate_digest"],
            "project_id": retry["request"]["project_id"],
            "project_binding_digest": retry["request"]["project_binding_digest"],
            "phase": retry["phase"], "attempt_no": 2,
            "adapter_id": retry["adapter_id"], "method": retry["method"],
            "approval_binding_digest": retry["approval_binding_digest"],
            "adapter_request": retry["request"],
        })[1]
        before = conn.execute(
            "SELECT COUNT(*) FROM program_delivery_operations"
        ).fetchone()[0]
        with pytest.raises(ValueError, match="reconcile"):
            kb.prepare_program_delivery_operation(conn, retry)
        assert conn.execute(
            "SELECT COUNT(*) FROM program_delivery_operations"
        ).fetchone()[0] == before

        # The mutable gate is authenticated by the append-only settlement
        # evidence; direct SQL clearing must fail before retry admission.
        conn.execute(
            "UPDATE program_deliveries SET reconcile_phase=NULL,reconcile_code=NULL "
            "WHERE delivery_id=?",
            (setup["delivery"]["delivery_id"],),
        )
        with pytest.raises(ValueError, match="history integrity"):
            kb.prepare_program_delivery_operation(conn, retry)
        assert conn.execute(
            "SELECT COUNT(*) FROM program_delivery_operations"
        ).fetchone()[0] == before
        conn.execute(
            "UPDATE program_deliveries SET reconcile_phase='pr',reconcile_code='REMOTE_UNKNOWN' "
            "WHERE delivery_id=?",
            (setup["delivery"]["delivery_id"],),
        )

        observation = {"remote_effect": "absent"}
        reconcile_request = {
            **_actor(setup, integrator, "integrator", ambiguous["version"]),
            "operation_id": request["operation_id"],
            "request_digest": request["request_digest"],
            "observed_outcome": "clean_failed",
            "observed_code": "REMOTE_ABSENT",
            "observation_digest": kb._sha256_json(observation)[1],
            "idempotency_key": "explicit-reconcile",
        }
        unexpected_request = dict(reconcile_request)
        unexpected_request["unexpected"] = {"nested": "value"}
        with pytest.raises(ValueError, match="schema"):
            kb.reconcile_program_delivery_operation(conn, unexpected_request)
        reconciled = kb.reconcile_program_delivery_operation(conn, reconcile_request)
        assert reconciled["state"] == "review"
        assert reconciled["version"] == ambiguous["version"] + 1
        assert tuple(conn.execute(
            "SELECT reconcile_phase,reconcile_code FROM program_deliveries WHERE delivery_id=?",
            (setup["delivery"]["delivery_id"],),
        ).fetchone()) == (None, None)
        evidence_before_replay = conn.execute(
            "SELECT COUNT(*) FROM program_delivery_evidence WHERE delivery_id=?",
            (setup["delivery"]["delivery_id"],),
        ).fetchone()[0]
        replayed = kb.reconcile_program_delivery_operation(conn, reconcile_request)
        assert replayed["deduplicated"] is True
        assert conn.execute(
            "SELECT COUNT(*) FROM program_delivery_evidence WHERE delivery_id=?",
            (setup["delivery"]["delivery_id"],),
        ).fetchone()[0] == evidence_before_replay

        retry["expected_version"] = reconciled["version"]
        prepared_retry = kb.prepare_program_delivery_operation(conn, retry)
        assert prepared_retry["version"] == reconciled["version"] + 1
        assert conn.execute(
            "SELECT COUNT(*) FROM program_delivery_operations"
        ).fetchone()[0] == before + 1

        kb.claim_program_delivery_operation(conn, {
            **_actor(setup, integrator, "integrator", prepared_retry["version"]),
            "operation_id": retry["operation_id"],
            "request_digest": retry["request_digest"],
            "idempotency_key": "retry-claim",
        })
        kb.mark_program_delivery_dispatch_started(conn, {
            **_actor(setup, integrator, "integrator", prepared_retry["version"] + 1),
            "operation_id": retry["operation_id"],
            "request_digest": retry["request_digest"],
            "idempotency_key": "retry-dispatch",
        })
        second_result = {"observation": "still-unknown"}
        second_ambiguous = kb.settle_program_delivery_operation(conn, {
            **_actor(setup, integrator, "integrator", prepared_retry["version"] + 2),
            "operation_id": retry["operation_id"],
            "request_digest": retry["request_digest"],
            "outcome": "ambiguous",
            "outcome_code": "REMOTE_UNKNOWN_SECOND",
            "result": second_result,
            "result_digest": kb._sha256_json(second_result)[1],
            "idempotency_key": "retry-ambiguous",
        })
        confirmed_observation = {"remote_effect": "present"}
        confirmed_request = {
            **_actor(setup, integrator, "integrator", second_ambiguous["version"]),
            "operation_id": retry["operation_id"],
            "request_digest": retry["request_digest"],
            "observed_outcome": "confirmed",
            "observed_code": "REMOTE_PRESENT",
            "observation_digest": kb._sha256_json(confirmed_observation)[1],
            "idempotency_key": "confirmed-reconcile",
        }
        confirmed = kb.reconcile_program_delivery_operation(conn, confirmed_request)
        assert confirmed["state"] == "pr"
        assert confirmed["version"] == second_ambiguous["version"] + 1
        evidence_row = conn.execute(
            "SELECT id,payload_json FROM program_delivery_evidence "
            "WHERE event_type='reconcile_observed' ORDER BY id DESC LIMIT 1"
        ).fetchone()
        payload = json.loads(evidence_row["payload_json"])
        payload["unexpected_top_level"] = "value"
        payload_json, evidence_digest = kb._sha256_json(payload)
        conn.execute("DROP TRIGGER trg_program_delivery_evidence_no_update")
        conn.execute(
            "UPDATE program_delivery_evidence SET payload_json=?,evidence_digest=? WHERE id=?",
            (payload_json, evidence_digest, evidence_row["id"]),
        )
        with pytest.raises(ValueError, match="history integrity"):
            kb.reconcile_program_delivery_operation(conn, confirmed_request)
    finally:
        conn.close()


@pytest.mark.parametrize(
    "stage", ("prepare", "claim", "settle_operation", "settle_delivery"),
)
def test_delivery_operation_zero_row_cas_fails_closed(tmp_path, stage):
    conn = kb.connect(tmp_path / f"zero-row-{stage}.db")
    try:
        setup = _setup(conn)
        integrator, _ = _to_review(conn, setup)
        request, _ = _prepare_operation(conn, setup, integrator, execute=False)
        delivery_id = setup["delivery"]["delivery_id"]

        if stage == "prepare":
            conn.execute(
                "CREATE TEMP TRIGGER ignore_delivery_update BEFORE UPDATE ON program_deliveries "
                "BEGIN SELECT RAISE(IGNORE); END"
            )
            with pytest.raises(ValueError, match="CAS"):
                kb.prepare_program_delivery_operation(conn, request)
            assert conn.execute(
                "SELECT COUNT(*) FROM program_delivery_operations"
            ).fetchone()[0] == 0
            return

        kb.prepare_program_delivery_operation(conn, request)
        claim_request = {
            **_actor(setup, integrator, "integrator", 6),
            "operation_id": request["operation_id"],
            "request_digest": request["request_digest"],
            "idempotency_key": "cas-claim",
        }
        if stage == "claim":
            conn.execute(
                "CREATE TEMP TRIGGER ignore_delivery_update BEFORE UPDATE ON program_deliveries "
                "BEGIN SELECT RAISE(IGNORE); END"
            )
            with pytest.raises(ValueError, match="CAS"):
                kb.claim_program_delivery_operation(conn, claim_request)
            assert conn.execute(
                "SELECT status FROM program_delivery_operations"
            ).fetchone()[0] == "prepared"
            return

        kb.claim_program_delivery_operation(conn, claim_request)
        kb.mark_program_delivery_dispatch_started(conn, {
            **_actor(setup, integrator, "integrator", 7),
            "operation_id": request["operation_id"],
            "request_digest": request["request_digest"],
            "idempotency_key": "cas-dispatch",
        })
        if stage == "settle_operation":
            conn.execute(
                "CREATE TEMP TRIGGER ignore_operation_update BEFORE UPDATE ON "
                "program_delivery_operations BEGIN SELECT RAISE(IGNORE); END"
            )
        else:
            conn.execute(
                "CREATE TEMP TRIGGER ignore_delivery_update BEFORE UPDATE ON program_deliveries "
                "BEGIN SELECT RAISE(IGNORE); END"
            )
        result = {"number": 17}
        with pytest.raises(ValueError, match="CAS"):
            kb.settle_program_delivery_operation(conn, {
                **_actor(setup, integrator, "integrator", 8),
                "operation_id": request["operation_id"],
                "request_digest": request["request_digest"],
                "outcome": "confirmed", "outcome_code": "PR_CREATED",
                "result": result, "result_digest": kb._sha256_json(result)[1],
                "idempotency_key": f"cas-{stage}",
            })
        assert conn.execute(
            "SELECT status FROM program_delivery_operations"
        ).fetchone()[0] == "dispatch_started"
        assert conn.execute(
            "SELECT state FROM program_deliveries WHERE delivery_id=?", (delivery_id,),
        ).fetchone()[0] == "review"
    finally:
        conn.close()
