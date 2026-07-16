"""Typed Phase-1 program-control validation and durable mutations."""

from __future__ import annotations

import copy
import hashlib
import json
import re
import sqlite3
import time
import unicodedata
from typing import Any, BinaryIO, Callable


MAX_TRANSPORT_BYTES = 1024 * 1024
MAX_AGGREGATE_BYTES = 256 * 1024
MAX_JSON_DEPTH = 64
MAX_SAFE_INTEGER = 2**53 - 1
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_TERMINAL = {"done", "archived"}


class ProgramControlError(ValueError):
    def __init__(self, code: str, detail: str | None = None):
        self.code = code
        super().__init__(code if detail is None else f"{code}: {detail}")


def _reject_constant(value: str) -> None:
    raise ProgramControlError("invalid_json", value)


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ProgramControlError("duplicate_key", key)
        result[key] = value
    return result


def parse_request_stdin(stream: BinaryIO) -> dict[str, Any]:
    raw = stream.read(MAX_TRANSPORT_BYTES + 1)
    if len(raw) > MAX_TRANSPORT_BYTES:
        raise ProgramControlError("transport_too_large")
    try:
        text = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as exc:
        raise ProgramControlError("invalid_utf8") from exc
    decoder = json.JSONDecoder(object_pairs_hook=_pairs, parse_constant=_reject_constant)
    try:
        value, end = decoder.raw_decode(text)
    except ProgramControlError:
        raise
    except (RecursionError, OverflowError) as exc:
        raise ProgramControlError("invalid_request") from exc
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise ProgramControlError("invalid_json") from exc
    if text[end:].strip():
        raise ProgramControlError("extra_json_document")
    if not isinstance(value, dict):
        raise ProgramControlError("request_not_object")
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if depth > MAX_JSON_DEPTH:
            raise ProgramControlError("invalid_request")
        if isinstance(item, bool):
            raise ProgramControlError("bool_as_int")
        if isinstance(item, dict):
            pending.extend((nested, depth + 1) for nested in item.values())
        elif isinstance(item, list):
            pending.extend((nested, depth + 1) for nested in item)
    return value


def _exact(request: dict[str, Any], keys: set[str]) -> None:
    if set(request) != keys:
        raise ProgramControlError("invalid_keys")


def _string(value: Any, field: str, *, minimum: int = 1, maximum: int = 128,
            identifier: bool = False) -> str:
    if not isinstance(value, str):
        raise ProgramControlError(f"invalid_{field}")
    value = unicodedata.normalize("NFC", value)
    if not minimum <= len(value) <= maximum:
        raise ProgramControlError(f"invalid_{field}")
    if identifier and not _ID_RE.fullmatch(value):
        raise ProgramControlError(f"invalid_{field}")
    return value


def _integer(value: Any, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_SAFE_INTEGER:
        raise ProgramControlError(f"invalid_{field}")
    return value


def _actor(value: Any, *, agent: bool) -> str:
    actor = _string(value, "actor", maximum=128)
    if agent:
        if not actor.startswith("agent:") or not _ID_RE.fullmatch(actor[6:]):
            raise ProgramControlError("invalid_actor")
    elif actor not in {"control:owner", "control:sergey"}:
        raise ProgramControlError("invalid_actor")
    return actor


def _bounded_list(value: Any, field: str, *, maximum: int, item_max: int,
                  unique: bool = False, identifiers: bool = False) -> list[str]:
    if not isinstance(value, list) or len(value) > maximum:
        raise ProgramControlError(f"invalid_{field}")
    result = [_string(item, field, maximum=item_max, identifier=identifiers) for item in value]
    if unique and len(set(result)) != len(result):
        raise ProgramControlError(f"invalid_{field}")
    return result


def _aggregate(request: dict[str, Any]) -> None:
    encoded = json.dumps(request, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
    if len(encoded) > MAX_AGGREGATE_BYTES:
        raise ProgramControlError("request_too_large")


def _canonical(request: dict[str, Any], excluded: set[str]) -> str:
    semantic = {key: value for key, value in request.items() if key not in excluded}
    raw = json.dumps(semantic, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                     allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _lineage(conn: sqlite3.Connection, root_id: str, node_ids: list[str]) -> dict[str, sqlite3.Row]:
    ids = list(dict.fromkeys([root_id, *node_ids]))
    rows = conn.execute(
        f"SELECT id, status, orchestration_root_id, orchestration_policy, program_control_version "
        f"FROM tasks WHERE id IN ({','.join('?' for _ in ids)})", ids,
    ).fetchall()
    by_id = {row["id"]: row for row in rows}
    root = by_id.get(root_id)
    if root is None or root["orchestration_root_id"] != root_id:
        raise ProgramControlError("outside_root")
    for node_id in node_ids:
        row = by_id.get(node_id)
        if row is None or row["orchestration_root_id"] != root_id:
            raise ProgramControlError("outside_root")
        version = row["program_control_version"]
        if isinstance(version, bool) or not isinstance(version, int) or not 0 <= version <= MAX_SAFE_INTEGER:
            raise ProgramControlError("invalid_stored_version")
    return by_id


def _replay(conn: sqlite3.Connection, root_id: str, action: str, key: str,
            fingerprint: str) -> dict[str, Any] | None:
    row = conn.execute(
        "SELECT fingerprint, result_json FROM program_control_requests "
        "WHERE root_id=? AND action=? AND idempotency_key=?", (root_id, action, key),
    ).fetchone()
    if row is None:
        return None
    if row["fingerprint"] != fingerprint:
        raise ProgramControlError("idempotency_conflict")
    result = json.loads(row["result_json"])
    result["deduplicated"] = True
    return result


def _store(conn: sqlite3.Connection, root: str, action: str, key: str,
           fingerprint: str, result: dict[str, Any], now: int) -> None:
    conn.execute(
        "INSERT INTO program_control_requests VALUES (?,?,?,?,?,?)",
        (root, action, key, fingerprint,
         json.dumps(result, ensure_ascii=False, sort_keys=True, separators=(",", ":")), now),
    )


def _mutation(conn: sqlite3.Connection, request: dict[str, Any], action: str,
              expected_field: str, operation: Callable[[int], dict[str, Any]]) -> dict[str, Any]:
    fingerprint = _canonical(request, {"idempotency_key", expected_field})
    nested = conn.in_transaction
    if nested:
        conn.execute("SAVEPOINT program_control_mutation")
    else:
        conn.execute("BEGIN IMMEDIATE")
    try:
        replay = _replay(conn, request["root_id"], action, request["idempotency_key"], fingerprint)
        if replay is not None:
            if nested:
                conn.execute("RELEASE SAVEPOINT program_control_mutation")
            else:
                conn.commit()
            return replay
        result = operation(int(time.time()))
        _store(conn, request["root_id"], action, request["idempotency_key"], fingerprint,
               result, int(time.time()))
        if nested:
            conn.execute("RELEASE SAVEPOINT program_control_mutation")
        else:
            conn.commit()
        return result
    except Exception:
        if nested:
            conn.execute("ROLLBACK TO SAVEPOINT program_control_mutation")
            conn.execute("RELEASE SAVEPOINT program_control_mutation")
        else:
            conn.rollback()
        raise


_OPTION_KEYS = {"option_id", "label", "summary", "benefits", "risks", "reversibility",
                "security_impact", "cost_impact", "operations_impact"}
_OPEN_KEYS = {"root_id", "checkpoint_id", "node_id", "title", "options",
              "recommended_option_id", "recommendation_rationale", "affected_node_ids",
              "expected_node_version", "idempotency_key", "actor"}


def open_decision(conn: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(request)
    _exact(request, _OPEN_KEYS); _aggregate(request)
    for field in ("root_id", "checkpoint_id", "node_id", "recommended_option_id", "idempotency_key"):
        request[field] = _string(request[field], field, identifier=True)
    request["title"] = _string(request["title"], "title", maximum=200)
    request["recommendation_rationale"] = _string(request["recommendation_rationale"], "recommendation_rationale", maximum=1000)
    request["expected_node_version"] = _integer(request["expected_node_version"], "expected_node_version")
    request["actor"] = _actor(request["actor"], agent=True)
    request["affected_node_ids"] = _bounded_list(request["affected_node_ids"], "affected_node_ids",
                                                  maximum=100, item_max=128, unique=True, identifiers=True)
    if not isinstance(request["options"], list) or not 2 <= len(request["options"]) <= 4:
        raise ProgramControlError("invalid_options")
    option_ids: set[str] = set()
    for option in request["options"]:
        if not isinstance(option, dict): raise ProgramControlError("invalid_options")
        _exact(option, _OPTION_KEYS)
        option["option_id"] = _string(option["option_id"], "option_id", identifier=True)
        if option["option_id"] in option_ids: raise ProgramControlError("invalid_options")
        option_ids.add(option["option_id"])
        option["label"] = _string(option["label"], "label", maximum=200)
        for field in ("summary", "security_impact", "cost_impact", "operations_impact"):
            option[field] = _string(option[field], field, maximum=1000)
        option["benefits"] = _bounded_list(option["benefits"], "benefits", maximum=10, item_max=500)
        option["risks"] = _bounded_list(option["risks"], "risks", maximum=10, item_max=500)
        if option["reversibility"] not in {"reversible", "partially_reversible", "irreversible"}:
            raise ProgramControlError("invalid_reversibility")
    if request["recommended_option_id"] not in option_ids:
        raise ProgramControlError("invalid_recommended_option_id")

    def operation(now: int) -> dict[str, Any]:
        rows = _lineage(conn, request["root_id"], [request["node_id"], *request["affected_node_ids"]])
        node = rows[request["node_id"]]
        if node["status"] in _TERMINAL:
            raise ProgramControlError("terminal_target")
        try:
            from hermes_cli.kanban_db import OrchestrationPolicy
            policy = OrchestrationPolicy.from_json(
                rows[request["root_id"]]["orchestration_policy"]
            )
            actor_profile = request["actor"][6:]
        except (TypeError, ValueError):
            raise ProgramControlError("invalid_actor") from None
        if actor_profile not in policy.orchestrator_assignees:
            raise ProgramControlError("invalid_actor")
        if node["program_control_version"] == MAX_SAFE_INTEGER:
            raise ProgramControlError("version_exhausted")
        if node["program_control_version"] != request["expected_node_version"]:
            raise ProgramControlError("version_conflict")
        if conn.execute(
            "SELECT 1 FROM program_decisions WHERE root_id=? AND checkpoint_id=?",
            (request["root_id"], request["checkpoint_id"]),
        ).fetchone() is not None:
            raise ProgramControlError("checkpoint_exists")
        changed = conn.execute("UPDATE tasks SET program_control_version=program_control_version+1 "
                               "WHERE id=? AND program_control_version=?",
                               (request["node_id"], request["expected_node_version"]))
        if changed.rowcount != 1: raise ProgramControlError("version_conflict")
        conn.execute("INSERT INTO program_decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                     (request["root_id"], request["checkpoint_id"], request["node_id"], "pending", 1,
                      request["title"], request["recommended_option_id"], request["recommendation_rationale"],
                      request["actor"], None, now, None))
        for ordinal, option in enumerate(request["options"]):
            conn.execute("INSERT INTO program_decision_options VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (request["root_id"], request["checkpoint_id"], option["option_id"], ordinal,
                          option["label"], option["summary"], json.dumps(option["benefits"], ensure_ascii=False),
                          json.dumps(option["risks"], ensure_ascii=False), option["reversibility"],
                          option["security_impact"], option["cost_impact"], option["operations_impact"]))
        for affected in request["affected_node_ids"]:
            conn.execute("INSERT INTO program_decision_affected_nodes VALUES (?,?,?)",
                         (request["root_id"], request["checkpoint_id"], affected))
        payload = {"checkpoint_id": request["checkpoint_id"], "node_id": request["node_id"],
                   "version": 1, "option_count": len(request["options"]),
                   "affected_node_count": len(request["affected_node_ids"]), "state": "pending"}
        conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES (?,?,?,?)",
                     (request["node_id"], "decision_checkpoint_opened", json.dumps(payload, separators=(",", ":")), now))
        return {"ok": True, "checkpoint_id": request["checkpoint_id"], "state": "pending",
                "version": 1, "deduplicated": False}
    return _mutation(conn, request, "decision_open", "expected_node_version", operation)


def select_decision(conn: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(request)
    _exact(request, {"root_id", "checkpoint_id", "option_id", "expected_version", "idempotency_key", "actor"}); _aggregate(request)
    for field in ("root_id", "checkpoint_id", "option_id", "idempotency_key"):
        request[field] = _string(request[field], field, identifier=True)
    request["expected_version"] = _integer(request["expected_version"], "expected_version")
    request["actor"] = _actor(request["actor"], agent=False)
    def operation(now: int) -> dict[str, Any]:
        _lineage(conn, request["root_id"], [])
        row = conn.execute("SELECT node_id, version, state FROM program_decisions WHERE root_id=? AND checkpoint_id=?",
                           (request["root_id"], request["checkpoint_id"])).fetchone()
        if row is None: raise ProgramControlError("unknown_checkpoint")
        if row["version"] != request["expected_version"] or row["state"] != "pending":
            raise ProgramControlError("version_conflict")
        if row["version"] == MAX_SAFE_INTEGER:
            raise ProgramControlError("version_exhausted")
        if conn.execute("SELECT 1 FROM program_decision_options WHERE root_id=? AND checkpoint_id=? AND option_id=?",
                        (request["root_id"], request["checkpoint_id"], request["option_id"])).fetchone() is None:
            raise ProgramControlError("unknown_option")
        changed = conn.execute("UPDATE program_decisions SET state='selected', version=version+1, selected_option_id=?, selected_at=? "
                               "WHERE root_id=? AND checkpoint_id=? AND state='pending' AND version=?",
                               (request["option_id"], now, request["root_id"], request["checkpoint_id"], request["expected_version"]))
        if changed.rowcount != 1: raise ProgramControlError("version_conflict")
        affected = [item["node_id"] for item in conn.execute(
            "SELECT node_id FROM program_decision_affected_nodes "
            "WHERE root_id=? AND checkpoint_id=?",
            (request["root_id"], request["checkpoint_id"]),
        ).fetchall()]
        from hermes_cli.kanban_db import _recompute_ready_locked
        _recompute_ready_locked(conn, task_ids=affected)
        result_version = request["expected_version"] + 1
        payload = {"checkpoint_id": request["checkpoint_id"], "version": result_version,
                   "selected_option_id": request["option_id"], "state": "selected"}
        conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES (?,?,?,?)",
                     (row["node_id"], "decision_checkpoint_selected", json.dumps(payload, separators=(",", ":")), now))
        return {"ok": True, "checkpoint_id": request["checkpoint_id"], "state": "selected",
                "version": result_version, "selected_option_id": request["option_id"], "deduplicated": False}
    return _mutation(conn, request, "decision_select", "expected_version", operation)


def add_hint(conn: sqlite3.Connection, request: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(request)
    _exact(request, {"root_id", "node_id", "text", "expected_node_version", "idempotency_key", "actor"}); _aggregate(request)
    for field in ("root_id", "node_id", "idempotency_key"):
        request[field] = _string(request[field], field, identifier=True)
    request["text"] = _string(request["text"], "text", maximum=2000)
    request["expected_node_version"] = _integer(request["expected_node_version"], "expected_node_version")
    request["actor"] = _actor(request["actor"], agent=False)
    def operation(now: int) -> dict[str, Any]:
        node = _lineage(conn, request["root_id"], [request["node_id"]])[request["node_id"]]
        if node["status"] in _TERMINAL: raise ProgramControlError("terminal_target")
        if node["program_control_version"] != request["expected_node_version"]:
            raise ProgramControlError("version_conflict")
        if node["program_control_version"] == MAX_SAFE_INTEGER:
            raise ProgramControlError("version_exhausted")
        changed = conn.execute("UPDATE tasks SET program_control_version=program_control_version+1 "
                               "WHERE id=? AND program_control_version=?",
                               (request["node_id"], request["expected_node_version"]))
        if changed.rowcount != 1: raise ProgramControlError("version_conflict")
        committed = request["expected_node_version"] + 1
        hint_id = "h_" + hashlib.sha256(
            f"{request['root_id']}\0{request['idempotency_key']}".encode()).hexdigest()[:24]
        conn.execute("INSERT INTO program_hints(root_id,hint_id,node_id,text,actor,idempotency_key,"
                     "expected_node_version,committed_node_version,state,created_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
                     (request["root_id"], hint_id, request["node_id"], request["text"], request["actor"],
                      request["idempotency_key"], request["expected_node_version"], committed, "recorded", now))
        payload = {"hint_id": hint_id, "node_id": request["node_id"], "node_version": committed, "state": "recorded"}
        conn.execute("INSERT INTO task_events(task_id,kind,payload,created_at) VALUES (?,?,?,?)",
                     (request["node_id"], "hint_recorded", json.dumps(payload, separators=(",", ":")), now))
        return {"ok": True, "hint_id": hint_id, "node_id": request["node_id"], "state": "recorded",
                "node_version": committed, "deduplicated": False}
    return _mutation(conn, request, "hint_add", "expected_node_version", operation)
