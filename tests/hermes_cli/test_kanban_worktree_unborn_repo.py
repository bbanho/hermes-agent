"""RED test: provisioning a worktree in a repo with NO commits must not fail.

The kanban dispatcher failed with
    git worktree add failed for <repo>/.worktrees/<id> on branch probe-branch:
    fatal: invalid reference: HEAD

because ``_ensure_git_worktree`` unconditionally appended the start point
``HEAD`` when the branch did not exist yet. On an unborn repo (a fresh
``git init`` with zero commits) ``HEAD`` does not resolve, and git refuses
the whole operation. Passing ``-b <branch>`` alone lets git infer the
unborn/orphan start point and succeed.
"""

import subprocess
from pathlib import Path

import pytest

import hermes_cli.kanban_db as kb
import hermes_cli.kanban_db_connect as kbc
import hermes_cli.kanban_db_workspace as kbw


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )


@pytest.fixture()
def kanban_home(tmp_path: Path, monkeypatch) -> Path:
    """Isolated HERMES_HOME with an empty kanban DB (mirrors sibling suites)."""
    home = tmp_path / ".hermes"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    kb.init_db()
    return home


@pytest.fixture()
def unborn_repo(tmp_path: Path) -> Path:
    """A git repo with ZERO commits (unborn HEAD) — the failing shape."""
    repo = tmp_path / "probe-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    # A stray nested dir, mirroring the dispatcher's <repo>/.worktrees/<id> layout.
    (repo / ".worktrees").mkdir()
    return repo


def test_ensure_git_worktree_succeeds_on_unborn_repo(unborn_repo: Path):
    """The regression itself: no RuntimeError, worktree registered."""
    target = unborn_repo / ".worktrees" / "t_probe"

    kbw._ensure_git_worktree(unborn_repo, target, "probe-branch")

    assert target.exists(), "worktree was not materialized"
    listing = _git(unborn_repo, "worktree", "list").stdout
    assert str(target) in listing, f"worktree not registered:\n{listing}"


def test_ensure_git_worktree_on_unborn_repo_leaves_branch_unborn_not_dangling(
    unborn_repo: Path,
):
    """The worktree must be usable: checked out on the branch, not a broken ref.

    In a commitless repo ``git branch --list`` prints nothing (a branch with no
    commit cannot be listed yet), so the invariant is the worktree's own HEAD:
    it must already point at the branch we asked for, and a commit made inside
    the worktree must land on that branch in the shared repo.
    """
    target = unborn_repo / ".worktrees" / "t_probe"
    kbw._ensure_git_worktree(unborn_repo, target, "probe-branch")

    gitfile = (target / ".git").read_text()
    assert gitfile.startswith("gitdir:"), f"expected a linked worktree gitfile, got:\n{gitfile}"

    # The linked worktree's HEAD points at the branch, not at a detached sha.
    worktree_head = (unborn_repo / ".git" / "worktrees" / "t_probe" / "HEAD").read_text()
    assert worktree_head.strip() == "ref: refs/heads/probe-branch", (
        f"worktree HEAD should be on the new branch, got: {worktree_head!r}"
    )

    # And the worktree is genuinely writable: a commit there becomes the branch.
    subprocess.run(
        ["git", "-C", str(target), "-c", "user.email=p@p", "-c", "user.name=p",
         "commit", "-q", "--allow-empty", "-m", "first"],
        check=True, capture_output=True,
    )
    branches = _git(unborn_repo, "branch", "--list").stdout
    assert "probe-branch" in branches, f"branch missing after commit:\n{branches}"


def test_ensure_git_worktree_still_uses_head_on_repo_with_commits(tmp_path: Path):
    """Regression guard: the normal (committed) path must be unchanged."""
    repo = tmp_path / "real-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=p@p", "-c", "user.name=p",
         "commit", "-q", "--allow-empty", "-m", "init"],
        check=True, capture_output=True,
    )
    target = repo / ".worktrees" / "t_probe"

    kbw._ensure_git_worktree(repo, target, "probe-branch")

    assert target.exists()
    listing = _git(repo, "worktree", "list").stdout
    assert str(target) in listing
    # And it actually branched off the commit, i.e. HEAD was a real start point.
    got = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    assert got.returncode == 0, f"worktree has no resolvable HEAD:\n{got.stderr}"
    assert got.stdout.strip()


def test_ensure_git_worktree_uses_existing_branch_without_head(unborn_repo: Path):
    """When the branch already exists, the positional form must still be used."""
    # Give the repo a commit so the branch can be created, then delete nothing.
    subprocess.run(
        ["git", "-C", str(unborn_repo), "-c", "user.email=p@p", "-c", "user.name=p",
         "commit", "-q", "--allow-empty", "-m", "init"],
        check=True, capture_output=True,
    )
    _git(unborn_repo, "branch", "existing-branch")
    target = unborn_repo / ".worktrees" / "t_existing"

    kbw._ensure_git_worktree(unborn_repo, target, "existing-branch")

    assert target.exists()
    assert str(target) in _git(unborn_repo, "worktree", "list").stdout


class _Task:
    """Minimal stand-in for the dispatcher Task (only fields read are set)."""

    def __init__(self, task_id: str, workspace_path: str | None):
        self.id = task_id
        self.workspace_kind = "worktree"
        self.workspace_path = workspace_path
        self.branch_name = f"wt/{task_id}"


def test_resolve_worktree_workspace_on_unborn_repo_does_not_fail_dispatch(
    kanban_home, unborn_repo: Path
):
    """End-to-end through the function dispatch calls: the original crash path.

    The probe card was set to ``workspace_path`` = a repo with no commits, which
    sent ``_ensure_git_worktree`` down the ``-b <branch> <path> HEAD`` branch and
    raised ``fatal: invalid reference: HEAD``, killing the spawn.
    """
    target = unborn_repo / ".worktrees" / "t_dispatch"
    target.parent.mkdir(parents=True, exist_ok=True)

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="probe in commitless repo",
            workspace_kind="worktree",
            workspace_path=str(target),
        )
        task = kb.get_task(conn, tid)
        assert task is not None

    resolved, branch = kbw._resolve_worktree_workspace(task)

    assert Path(resolved) == target.resolve()
    assert branch == f"wt/{tid}"
    assert target.exists()
    assert str(target) in _git(unborn_repo, "worktree", "list").stdout


def test_resolve_worktree_workspace_is_idempotent_on_second_dispatch(
    kanban_home, unborn_repo: Path
):
    """A retry (spawn_failed -> ready -> spawn) must not fail on the re-provision."""
    target = unborn_repo / ".worktrees" / "t_retry"
    target.parent.mkdir(parents=True, exist_ok=True)

    with kbc.connect() as conn:
        tid = kb.create_task(
            conn,
            title="probe retry",
            workspace_kind="worktree",
            workspace_path=str(target),
        )
        task = kb.get_task(conn, tid)
        assert task is not None

    first, _ = kbw._resolve_worktree_workspace(task)
    second, _ = kbw._resolve_worktree_workspace(task)

    assert Path(first) == Path(second) == target.resolve()
    assert str(target) in _git(unborn_repo, "worktree", "list").stdout