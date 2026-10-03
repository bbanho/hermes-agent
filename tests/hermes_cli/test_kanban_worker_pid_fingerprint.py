"""A recycled worker PID is never mistaken for our worker.

``tasks.worker_pid`` survives a reboot; the number can then belong to an unrelated process. Every
liveness decision (extend/defer the claim) and every kill (SIGTERM/SIGKILL on timeout or reclaim)
must require the spawn-time start fingerprint to match, never bare PID existence.
"""

import os
import signal
import time
from typing import Optional

import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from hermes_cli import kanban_db_dispatch as kbd


@pytest.fixture
def board(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("HERMES_KANBAN_CRASH_GRACE_SECONDS", "0")
    conn = kbc.connect(tmp_path / "kanban.db")
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture
def boot_epoch(monkeypatch):
    """Pin the instantiation epoch so no assertion depends on this host's real boot.

    Patched on ``gateway.drain_control`` — the single source both the fingerprint
    (``_process_fingerprint``) and the stale-reclaim classification read — and,
    defensively, on ``kanban_db`` in case the helper binds the name at module
    scope. Tests set ``state["value"]``; the lru_cache on the real witness is
    cleared first so a later reader cannot pick up a boot captured by another
    test.
    """
    from gateway import drain_control

    drain_control.current_instantiation_epoch.cache_clear()
    state = {"value": "liveboot:1"}

    def _fake_epoch():
        return state["value"]

    monkeypatch.setattr(drain_control, "current_instantiation_epoch", _fake_epoch)
    monkeypatch.setattr(kb, "current_instantiation_epoch", _fake_epoch, raising=False)
    return state


DEAD_PID = 12345  # never signalled: _pid_alive is stubbed dead in these tests


def _claimed_running(conn, *, pid: int, started_at, max_runtime=None) -> str:
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=max_runtime)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, pid)
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ?, started_at = ?, claim_expires = ? WHERE id = ?",
                     (started_at, old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))
    return tid


def test_recycled_pid_is_reclaimed_without_being_signalled(board):
    """Our own live PID with a foreign fingerprint models a post-reboot recycle: the claim is released
    (dead worker), no signal is sent, and max-runtime enforcement does not SIGTERM the stranger either."""
    conn = board
    killed = []
    stranger_fingerprint = 1  # no live process started at tick 1
    tid = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint, max_runtime=1)

    assert kbd._worker_alive(os.getpid(), stranger_fingerprint) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    tid2 = _claimed_running(conn, pid=os.getpid(), started_at=stranger_fingerprint)
    assert kb.release_stale_claims(conn, signal_fn=lambda pid, sig: killed.append((pid, sig))) == 1
    assert killed == []
    assert kb.get_task(conn, tid2).status == "ready"


def test_matching_fingerprint_keeps_the_live_worker(board):
    """The same PID with ITS OWN fingerprint (recorded at spawn) is our worker: the expired claim is
    extended rather than reclaimed, and the timeout path signals it."""
    from gateway.status import get_process_start_time

    conn = board
    killed = []
    tid = _claimed_running(conn, pid=os.getpid(), started_at=get_process_start_time(os.getpid()))
    assert kbd._worker_alive(os.getpid(), get_process_start_time(os.getpid())) is True
    assert kb.release_stale_claims(conn) == 0
    assert kb.get_task(conn, tid).status == "running"
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert "claim_extended" in kinds

    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET max_runtime_seconds = 1 WHERE id = ?", (tid,))
    kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed and killed[0] == (os.getpid(), signal.SIGTERM)


def test_same_pid_and_start_tick_on_another_boot_is_foreign(board, monkeypatch):
    """A row that survived a reboot: the PID AND the boot-relative start tick both match a process on
    this boot (the Linux start time is clock ticks since boot, so that recurs), but the persisted
    instantiation epoch does not. The worker is foreign: claim released, zero signals."""
    from gateway import drain_control

    conn = board
    killed = []
    live_fingerprint = kbd._process_fingerprint(os.getpid())
    assert live_fingerprint is not None and live_fingerprint.split("|", 1)[1] == str(
        __import__("gateway.status", fromlist=["x"]).get_process_start_time(os.getpid()))
    tid = _claimed_running(conn, pid=os.getpid(), started_at=live_fingerprint, max_runtime=1)
    assert kbd._worker_alive(os.getpid(), live_fingerprint) is True

    # Same PID, same start tick, different boot identity.
    other_boot = "deadbeef-boot:1|" + live_fingerprint.split("|", 1)[1]
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_started_at = ? WHERE id = ?", (other_boot, tid))
    assert kbd._worker_alive(os.getpid(), other_boot) is False
    assert tid in kbd.enforce_max_runtime(conn, signal_fn=lambda pid, sig: killed.append((pid, sig)))
    assert killed == []
    task = kb.get_task(conn, tid)
    assert task.status == "ready" and task.worker_pid is None

    # The same value re-derived on THIS boot still identifies our worker (the witness is stable
    # within a boot, unlike the recorded epoch of a previous one).
    drain_control.current_instantiation_epoch.cache_clear()
    assert kbd._process_fingerprint(os.getpid()) == live_fingerprint


def test_unverified_fingerprint_capture_never_authorizes_a_signal(board, monkeypatch):
    """Fingerprint capture fails for a new spawn: the row is NOT a legacy NULL row. A live PID under
    it is never SIGTERM/SIGKILLed by any reclaim/timeout path, and the claim is held (not released
    beside the live process); once the PID is gone the claim is reclaimed normally."""
    import gateway.status as status

    conn = board
    killed = []
    monkeypatch.setattr(status, "_get_process_start_time", lambda pid: None)
    tid = kb.create_task(conn, title="job", assignee="worker", max_runtime_seconds=1)
    kb.claim_task(conn, tid)
    kbd._set_worker_pid(conn, tid, os.getpid())
    row = conn.execute("SELECT worker_started_at FROM tasks WHERE id = ?", (tid,)).fetchone()
    assert row["worker_started_at"] == kbd.UNVERIFIED_WORKER_FINGERPRINT
    monkeypatch.undo()
    old = int(time.time()) - 3600
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET started_at = ?, claim_expires = ? WHERE id = ?", (old, old, tid))
        conn.execute("UPDATE task_runs SET started_at = ? WHERE id = (SELECT current_run_id FROM tasks WHERE id = ?)",
                     (old, tid))

    sig = lambda pid, s: killed.append((pid, s))  # noqa: E731
    assert kbd.enforce_max_runtime(conn, signal_fn=sig) == []
    assert kb.release_stale_claims(conn, signal_fn=sig) == 0
    assert killed == []
    assert kb.get_task(conn, tid).status == "running"
    # An explicit operator reclaim releases the claim (human override) but still sends nothing.
    assert kb.reclaim_task(conn, tid, reason="operator", signal_fn=sig) is True
    assert killed == []

    # The process is gone (a dead PID): the row is reclaimed like any dead worker, still no signal.
    tid2 = kb.create_task(conn, title="job2", assignee="worker")
    kb.claim_task(conn, tid2)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
                     (os.getpid(), kbd.UNVERIFIED_WORKER_FINGERPRINT, old, tid2))
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert killed == [] and kb.get_task(conn, tid2).status == "ready"


# ---------------------------------------------------------------------------
# Reboot vs crash: a claim that died with the host is not a card failure.
#
# After a reboot every in-flight claim is stale by definition — the workers are
# gone because the machine restarted, not because the card failed. Charged as a
# failure, DEFAULT_FAILURE_LIMIT reclaims (two) block a card whose text will
# never heal on retry. ``tasks.worker_started_at`` carries the
# ``"<instantiation epoch>|<start>"`` fingerprint from ``_set_worker_pid``, and
# that epoch changes on every reboot, so the recorded value distinguishes a
# reboot from a genuine crash without touching anything else.
#
# The reclaim itself is unchanged either way (lock cleared, events recorded);
# only the counter charge is at stake. "gave_up-shaped recording" is asserted
# where it is real: the counted path below still emits ``gave_up`` and blocks.
# On the suppressed path the card must NOT block, so no ``gave_up`` may exist —
# the non-success outcome is still booked (``last_failure_error``) and each
# reclaim is still in the event log.
# ---------------------------------------------------------------------------


def _fingerprint(epoch: str, start: int = 534886) -> str:
    """``_process_fingerprint``'s form for a worker spawned on ``epoch``."""
    return f"{epoch}|{start}"


def _dead_worker_row(conn, *, fingerprint, pid: int = DEAD_PID) -> str:
    """Claimed + running, claim expired an hour ago, owned by a dead PID."""
    tid = kb.create_task(conn, title="interrupted", assignee="worker")
    kb.claim_task(conn, tid)
    with kb.write_txn(conn):
        conn.execute(
            "UPDATE tasks SET worker_pid = ?, worker_started_at = ?, claim_expires = ? WHERE id = ?",
            (pid, fingerprint, int(time.time()) - 3600, tid),
        )
    return tid


def _expire_again(conn, tid: str) -> None:
    """Re-claim an expired card and expire it again (one more reclaim cycle)."""
    assert kb.claim_task(conn, tid) is not None
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET claim_expires = ? WHERE id = ?", (int(time.time()) - 3600, tid))


def _state(conn, tid: str) -> tuple[int, str, Optional[str]]:
    """``(consecutive_failures, status, last_failure_error)`` of a card."""
    row = conn.execute(
        "SELECT consecutive_failures, status, last_failure_error FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    return int(row["consecutive_failures"]), row["status"], row["last_failure_error"]


def test_reboot_stale_reclaim_does_not_charge_the_card(board, monkeypatch, boot_epoch):
    """A reclaim of a previous boot's claim is infrastructure: the claim is still
    reclaimed and the outcome still booked, but ``consecutive_failures`` does not
    move — so DEFAULT_FAILURE_LIMIT reclaims can no longer block a card that the
    host, not the card, killed."""
    conn = board
    signals: list[tuple[int, int]] = []
    sig = lambda pid, s: signals.append((pid, s))  # noqa: E731
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    tid = _dead_worker_row(conn, fingerprint=_fingerprint("previousboot:9"))

    # Two reclaim cycles: with the charge intact the second one trips the breaker
    # (DEFAULT_FAILURE_LIMIT == 2), which is exactly the reported symptom.
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert _state(conn, tid)[:2] == (0, "ready")

    _expire_again(conn, tid)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    failures, status, last_error = _state(conn, tid)
    assert failures == 0
    assert status == "ready"

    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert kinds.count("reclaimed") == 2
    # Reclaim still clears the stale lock so a fresh worker can be spawned.
    row = conn.execute(
        "SELECT claim_lock, claim_expires, worker_pid FROM tasks WHERE id = ?", (tid,),
    ).fetchone()
    assert row["claim_lock"] is None and row["claim_expires"] is None and row["worker_pid"] is None
    # The non-success outcome stays visible to operators even when uncharged.
    assert "stale_lock=" in (last_error or "")
    # Uncounted: no breaker event at all, so nothing holds the card for an operator.
    assert "gave_up" not in kinds


def test_same_boot_stale_reclaim_still_charges_and_blocks(board, monkeypatch, boot_epoch):
    """Regression guard: with the epoch MATCHING the current boot the worker really
    crashed. Accounting must be unchanged — the counter advances and the breaker
    still trips at the limit, with ``gave_up`` recorded."""
    conn = board
    sig = lambda pid, s: None  # noqa: E731
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    live_epoch = boot_epoch["value"]
    tid = _dead_worker_row(conn, fingerprint=_fingerprint(live_epoch))

    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert _state(conn, tid)[:2] == (1, "ready")

    _expire_again(conn, tid)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert _state(conn, tid)[:2] == (kbd.DEFAULT_FAILURE_LIMIT, "blocked")
    kinds = [e.kind for e in kb.list_events(conn, tid)]
    assert kinds.count("reclaimed") == 2
    assert kinds[-1] == "gave_up"


def test_legacy_integer_fingerprint_still_charges(board, monkeypatch, boot_epoch):
    """Pre-fingerprint rows carry an integer ``worker_started_at``: there is no epoch
    to compare, so the reclaim keeps counting (unchanged behaviour)."""
    conn = board
    sig = lambda pid, s: None  # noqa: E731
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    tid = _dead_worker_row(conn, fingerprint=534886)

    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert _state(conn, tid)[:2] == (1, "ready")

    _expire_again(conn, tid)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert _state(conn, tid)[:2] == (kbd.DEFAULT_FAILURE_LIMIT, "blocked")


def test_no_current_epoch_still_charges(board, monkeypatch, boot_epoch):
    """``current_instantiation_epoch()`` returns ``""`` off Linux / without ``/proc``:
    the epoch check is disabled, not failed-closed. The reclaim stays counted."""
    conn = board
    sig = lambda pid, s: None  # noqa: E731
    monkeypatch.setattr(kb, "_pid_alive", lambda pid: False)

    boot_epoch["value"] = ""
    tid = _dead_worker_row(conn, fingerprint=_fingerprint("previousboot:9"))

    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert _state(conn, tid)[:2] == (1, "ready")

    _expire_again(conn, tid)
    assert kb.release_stale_claims(conn, signal_fn=sig) == 1
    assert _state(conn, tid)[:2] == (kbd.DEFAULT_FAILURE_LIMIT, "blocked")
