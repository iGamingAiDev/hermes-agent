import argparse
import errno
import json
import os
import sqlite3
import tempfile
from pathlib import Path

import pytest

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb


SUCCESS = (
    '{"contract":"hermes-program-control","board":"alpha","version":1,'
    '"schema":1,"cli":1,"dispatcher_gate":1,"classic_worker_boundary":1,'
    '"goal_loop_boundary":1,"ack_writer":1,"events":1}\n'
)


@pytest.fixture
def current_board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    kb.create_board("alpha")
    with kb.connect_closing(board="alpha") as conn:
        conn.execute(
            "INSERT INTO tasks(id,title,status,created_at) VALUES (?,?,?,?)",
            ("wal-canary", "visible only through WAL", "todo", 1),
        )
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    return home


def _parser():
    top = argparse.ArgumentParser()
    sub = top.add_subparsers(dest="command")
    kc.build_parser(sub)
    return top


def _run(argv, capsys):
    args = _parser().parse_args(argv)
    code = kc.kanban_command(args)
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _images(path: Path):
    result = {}
    for candidate in (path, Path(str(path) + "-wal"), Path(str(path) + "-shm")):
        if candidate.exists():
            fd = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOATIME", 0))
            try:
                contents = os.read(fd, candidate.stat().st_size)
            finally:
                os.close(fd)
            stat = candidate.stat()
            result[candidate.name] = (
                contents,
                stat.st_mode,
                stat.st_uid,
                stat.st_gid,
                stat.st_size,
                stat.st_atime_ns,
                stat.st_mtime_ns,
                stat.st_ctime_ns,
            )
    return result


def _ancestor_stats(path: Path):
    current = Path(path.anchor)
    ancestors = [current]
    for part in path.parent.parts[1:]:
        current /= part
        ancestors.append(current)
    return {ancestor: ancestor.stat() for ancestor in ancestors}


def test_capabilities_exact_repeatable_read_only_success(current_board, capsys):
    path = kb.kanban_db_path(board="alpha")
    before = _images(path)
    ancestor_before = _ancestor_stats(path)
    argv = ["kanban", "--board", "ALPHA", "program", "capabilities", "--json"]
    assert _run(argv, capsys) == (0, SUCCESS, "")
    assert _run(argv, capsys) == (0, SUCCESS, "")
    assert _images(path) == before
    assert _ancestor_stats(path) == ancestor_before
    payload = json.loads(SUCCESS)
    assert all(type(value) is int for value in list(payload.values())[2:])


@pytest.mark.parametrize(
    "tail",
    [
        ["capabilities"],
        ["capabilities", "--json", "extra"],
        ["capabilities", "extra", "--json"],
        ["capabilities", "child", "--json"],
    ],
)
def test_capabilities_parse_failures_are_machine_only(tail, capsys):
    with pytest.raises(SystemExit) as exc:
        _parser().parse_args(["kanban", "--board", "alpha", "program", *tail])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == '{"error":"invalid_request","ok":false}\n'


@pytest.mark.parametrize("board", ["", "   ", "../escape", "bad/slash"])
def test_capabilities_bad_board_is_fixed_and_creates_nothing(tmp_path, monkeypatch, board, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _run(
        ["kanban", "--board", board, "program", "capabilities", "--json"], capsys
    ) == (2, "", '{"error":"invalid_request","ok":false}\n')
    assert list(home.iterdir()) == []


def test_capabilities_unknown_board_is_fixed_and_creates_nothing(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    assert _run(
        ["kanban", "--board", "missing", "program", "capabilities", "--json"], capsys
    ) == (2, "", '{"error":"unknown_board","ok":false}\n')
    assert list(home.iterdir()) == []


def test_capabilities_default_without_database_is_unknown_board(tmp_path, monkeypatch, capsys):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    before = _tree_image(home)
    assert _run(
        ["kanban", "--board", "default", "program", "capabilities", "--json"], capsys
    ) == (2, "", '{"error":"unknown_board","ok":false}\n')
    assert _tree_image(home) == before


def _tree_image(root: Path):
    return {
        str(path.relative_to(root)): (path.read_bytes(), path.stat())
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }


def test_capabilities_explicit_board_ignores_valid_foreign_db_override(
    tmp_path, monkeypatch, capsys
):
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    kb.create_board("foreign")
    foreign = kb.kanban_db_path(board="foreign")
    monkeypatch.setenv("HERMES_KANBAN_DB", str(foreign))
    (kb.board_dir("drifted") / "board.json").parent.mkdir(parents=True)
    (kb.board_dir("drifted") / "board.json").write_text('{}\n')
    before = _tree_image(home)
    assert _run(
        ["kanban", "--board", "drifted", "program", "capabilities", "--json"], capsys
    ) == (2, "", '{"error":"unknown_board","ok":false}\n')
    assert _tree_image(home) == before


def test_capabilities_explicit_board_ignores_invalid_db_and_routing_overrides(
    current_board, monkeypatch, capsys
):
    requested = kb.kanban_db_path(board="alpha")
    before = _images(requested)
    monkeypatch.setenv("HERMES_KANBAN_DB", str(current_board / "invalid.db"))
    monkeypatch.setenv("HERMES_KANBAN_BOARD", "foreign")
    with kb.scoped_current_board("foreign"):
        assert _run(
            ["kanban", "--board", "ALPHA", "program", "capabilities", "--json"], capsys
        ) == (0, SUCCESS, "")
    assert _images(requested) == before


def test_capabilities_does_not_read_stdin(current_board, monkeypatch, capsys):
    class ExplodingStdin:
        @property
        def buffer(self):
            raise AssertionError("stdin was read")

    monkeypatch.setattr("sys.stdin", ExplodingStdin())
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (0, SUCCESS, "")


def test_capabilities_schema_drift_and_corruption_are_fixed(current_board, monkeypatch, capsys):
    path = kb.kanban_db_path(board="alpha")
    path.write_bytes(b"not sqlite SECRET /private/path")
    before = _images(path)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    assert _images(path) == before


@pytest.mark.parametrize("drift", ["table", "index", "tasks_column", "phase2_hints"])
def test_capabilities_rejects_predecessor_and_partial_a3(
    current_board, monkeypatch, capsys, drift
):
    path = kb.kanban_db_path(board="alpha")
    connection = sqlite3.connect(path)
    try:
        if drift == "table":
            connection.execute("DROP TABLE program_control_requests")
        elif drift == "index":
            connection.execute("DROP INDEX idx_program_decisions_node")
        elif drift == "tasks_column":
            connection.execute(
                "ALTER TABLE tasks RENAME COLUMN program_control_version TO old_version"
            )
        else:
            connection.execute("PRAGMA foreign_keys=OFF")
            connection.execute("DROP TABLE program_hints")
            connection.executescript(kb._PHASE2_PROGRAM_HINTS_SCHEMA_SQL)
            connection.execute(
                "CREATE INDEX idx_program_hints_node_state "
                "ON program_hints(root_id,node_id,state,created_at)"
            )
        connection.commit()
    finally:
        connection.close()
    monkeypatch.setattr(kb, "_INITIALIZED_PATHS", set())
    before = _images(path)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    assert _images(path) == before


def test_capabilities_rejects_symlink_without_touching_target(current_board, capsys):
    real = kb.kanban_db_path(board="alpha")
    link = real.with_name("linked.db")
    link.symlink_to(real)
    before = _images(real)
    with pytest.raises(RuntimeError):
        kb.validate_current_program_control_schema_read_only(link)
    assert _images(real) == before


def test_capabilities_snapshot_instability_is_fixed(
    current_board, monkeypatch, capsys
):
    def unstable(_path):
        raise RuntimeError("SECRET /private/snapshot")

    monkeypatch.setattr(kb, "_strict_capability_db_snapshot", unstable)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')


def test_capability_snapshot_fails_closed_when_noatime_is_unavailable(
    current_board, monkeypatch, capsys
):
    path = kb.kanban_db_path(board="alpha")
    before = _images(path)
    monkeypatch.delattr(os, "O_NOATIME", raising=False)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    monkeypatch.undo()
    assert _images(path) == before


@pytest.mark.parametrize("flag_name", ["O_PATH", "O_DIRECTORY", "O_NOFOLLOW"])
def test_capability_snapshot_fails_closed_when_required_chain_flag_is_unavailable(
    current_board, monkeypatch, capsys, flag_name
):
    path = kb.kanban_db_path(board="alpha")
    source_before = _images(path)
    ancestor_before = _ancestor_stats(path)
    monkeypatch.delattr(os, flag_name, raising=False)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    monkeypatch.undo()
    assert _images(path) == source_before
    assert _ancestor_stats(path) == ancestor_before


def test_capability_snapshot_fails_closed_without_openat_support(
    current_board, monkeypatch, capsys
):
    path = kb.kanban_db_path(board="alpha")
    source_before = _images(path)
    ancestor_before = _ancestor_stats(path)
    monkeypatch.setattr(kb, "_OPEN_SUPPORTS_DIR_FD", False)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    assert _images(path) == source_before
    assert _ancestor_stats(path) == ancestor_before


@pytest.mark.parametrize(
    "error_number", [errno.EPERM, errno.EACCES, errno.EINVAL, errno.EOPNOTSUPP]
)
def test_capability_directory_chain_open_errors_fail_closed_without_source_read(
    current_board, monkeypatch, capsys, error_number
):
    path = kb.kanban_db_path(board="alpha")
    source_before = _images(path)
    ancestor_before = _ancestor_stats(path)
    real_open = os.open
    source_names = {path.name, path.name + "-wal", path.name + "-shm"}
    source_opened = False

    def reject_chain(name, flags, *args, **kwargs):
        nonlocal source_opened
        if name in source_names:
            source_opened = True
        if flags & os.O_DIRECTORY:
            raise OSError(error_number, "directory-chain open rejected")
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", reject_chain)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    assert not source_opened
    monkeypatch.undo()
    assert _images(path) == source_before
    assert _ancestor_stats(path) == ancestor_before


def test_capability_uses_opath_noatime_chain_and_read_only_source_flags(
    current_board, monkeypatch, capsys
):
    path = kb.kanban_db_path(board="alpha")
    real_open = os.open
    calls = []

    def record_open(name, flags, *args, **kwargs):
        calls.append((name, flags, kwargs.get("dir_fd")))
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", record_open)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (0, SUCCESS, "")

    chain = [(name, flags) for name, flags, _dir_fd in calls if flags & os.O_DIRECTORY]
    assert chain
    required_chain = os.O_PATH | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_NOATIME
    assert all(flags == required_chain for _name, flags in chain)
    source_names = {path.name, path.name + "-wal", path.name + "-shm"}
    sources = [(name, flags) for name, flags, _dir_fd in calls if name in source_names]
    assert sources
    required_source = os.O_RDONLY | os.O_NOATIME | os.O_NOFOLLOW
    assert all(flags == required_source for _name, flags in sources)


def test_capability_snapshot_noatime_eperm_does_not_fallback(
    current_board, monkeypatch, capsys
):
    path = kb.kanban_db_path(board="alpha")
    before = _images(path)
    real_open = os.open

    def denied(name, flags, *args, **kwargs):
        if flags & os.O_NOATIME:
            raise PermissionError(errno.EPERM, "noatime denied")
        return real_open(name, flags, *args, **kwargs)

    import errno
    monkeypatch.setattr(os, "open", denied)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    monkeypatch.undo()
    assert _images(path) == before


@pytest.mark.parametrize("suffix", ["", "-wal", "-shm"])
def test_capability_rejects_source_swapped_to_symlink_before_open(
    current_board, monkeypatch, capsys, suffix
):
    path = kb.kanban_db_path(board="alpha")
    source = Path(str(path) + suffix)
    if not source.exists():
        source.write_bytes(b"sidecar race fixture")
    arbitrary = current_board / f"arbitrary{suffix or '-main'}"
    arbitrary.write_bytes(b"ARBITRARY SECRET BYTES")
    arbitrary_before = _images(arbitrary)
    saved = source.with_name(source.name + ".saved")
    real_open = os.open
    swapped = False

    def swap_then_open(name, flags, *args, **kwargs):
        nonlocal swapped
        if kwargs.get("dir_fd") is not None and name == source.name and not swapped:
            swapped = True
            source.rename(saved)
            source.symlink_to(arbitrary)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(os, "open", swap_then_open)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')
    monkeypatch.undo()
    assert _images(arbitrary) == arbitrary_before


@pytest.mark.parametrize("ancestor_depth", [0, 1])
def test_capability_detects_parent_or_ancestor_directory_replacement(
    current_board, monkeypatch, capsys, ancestor_depth
):
    path = kb.kanban_db_path(board="alpha")
    changed_dir = path.parent if ancestor_depth == 0 else path.parent.parent
    displaced = changed_dir.with_name(changed_dir.name + "-displaced")
    real_verify = kb._capability_verify_chain

    def replace_then_verify(chain_path, signatures):
        changed_dir.rename(displaced)
        changed_dir.mkdir()
        return real_verify(chain_path, signatures)

    monkeypatch.setattr(kb, "_capability_verify_chain", replace_then_verify)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')


def test_capability_detects_content_mutation_during_copy(
    current_board, monkeypatch, capsys
):
    path = kb.kanban_db_path(board="alpha")
    real_copy = kb._copy_capability_fd
    def copy_then_mutate(fd, target):
        real_copy(fd, target)
        if target.name == path.name:
            with path.open("ab") as handle:
                handle.write(b"race")

    monkeypatch.setattr(kb, "_copy_capability_fd", copy_then_mutate)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')


def test_capability_success_leaks_no_descriptors_or_temporary_directories(
    current_board, capsys
):
    fd_root = Path("/proc/self/fd")
    if not fd_root.exists():
        pytest.skip("descriptor accounting requires procfs")
    before_fds = len(list(fd_root.iterdir()))
    before_temps = set(Path(tempfile.gettempdir()).glob("hermes-kanban-capability-*"))
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (0, SUCCESS, "")
    assert len(list(fd_root.iterdir())) == before_fds
    assert set(Path(tempfile.gettempdir()).glob("hermes-kanban-capability-*")) == before_temps


def test_capability_success_with_live_wal_preserves_all_source_images(
    current_board, capsys
):
    path = kb.kanban_db_path(board="alpha")
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute(
            "INSERT INTO tasks(id,title,status,created_at) VALUES (?,?,?,?)",
            ("live-wal", "must be visible", "todo", 2),
        )
        connection.commit()
        before = _images(path)
        assert {path.name, path.name + "-wal", path.name + "-shm"} <= set(before)
        assert _run(
            ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
        ) == (0, SUCCESS, "")
        assert _images(path) == before
    finally:
        connection.close()


def test_capabilities_builder_absence_fails_closed(current_board, monkeypatch, capsys):
    monkeypatch.delattr(kc, "_PROGRAM_CAPABILITIES", raising=False)
    assert _run(
        ["kanban", "--board", "alpha", "program", "capabilities", "--json"], capsys
    ) == (1, "", '{"error":"database_error","ok":false}\n')


def test_slash_allows_capabilities_but_keeps_mutations_denied(current_board):
    assert kc.run_slash("--board alpha program capabilities --json") == SUCCESS.rstrip()
    denied = kc.run_slash(
        "--board alpha program hint add --request-json-stdin --json"
    )
    assert "trusted direct" in denied.lower()
