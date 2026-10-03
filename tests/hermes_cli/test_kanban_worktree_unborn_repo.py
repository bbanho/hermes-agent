"""RED test: provisioning a worktree in a repo with NO commits must not fail.

The kanban dispatcher failed with::

    git worktree add failed for <repo>/.worktrees/<id> on branch probe-branch:
    fatal: invalid reference: HEAD

because ``_ensure_git_worktree`` unconditionally appended the start point
``HEAD`` when the branch did not exist yet. On an *unborn* repo (a fresh
``git init`` with zero commits) ``HEAD`` does not resolve, so git rejects the
whole operation and dispatch dies with ``spawn_failed``.

Dropping the start point lets git infer the orphan branch itself, which works
on a committed repo too (it simply branches from HEAD) and needs no new git
flag — see ``_git_has_commit``.
"""

import subprocess
from pathlib import Path

import pytest

import hermes_cli.kanban_db_workspace as kbw


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )


def _commit(repo: Path, message: str = "init") -> None:
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=p@p", "-c", "user.name=p",
         "commit", "-q", "--allow-empty", "-m", message],
        check=True, capture_output=True,
    )


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
    """The worktree must be usable: on the branch, not a broken detached ref.

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
    _commit(target, "first")
    branches = _git(unborn_repo, "branch", "--list").stdout
    assert "probe-branch" in branches, f"branch missing after commit:\n{branches}"


def test_ensure_git_worktree_does_not_leak_parent_files_into_unborn_worktree(
    unborn_repo: Path,
):
    """An orphan worktree shares the object store, not the parent checkout.

    Guards the one real hazard of dropping the start point: if the worktree
    were created against the parent index instead of an empty one, the parent's
    uncommitted files would follow the agent into the worktree.
    """
    (unborn_repo / "tracked-in-parent.txt").write_text("parent\n")
    _commit(unborn_repo, "add file")
    target = unborn_repo / ".worktrees" / "t_probe"

    kbw._ensure_git_worktree(unborn_repo, target, "probe-branch")

    assert not (target / "tracked-in-parent.txt").exists(), (
        "parent checkout leaked into the new worktree"
    )


def test_ensure_git_worktree_still_branches_from_head_on_repo_with_commits(tmp_path: Path):
    """Regression guard: the normal (committed) path must be unchanged.

    A committed repo must still branch off HEAD — if the start point were
    dropped unconditionally, the worktree would be an orphan and silently lose
    the entire existing history.
    """
    repo = tmp_path / "real-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    _commit(repo, "init")
    parent_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    target = repo / ".worktrees" / "t_probe"

    kbw._ensure_git_worktree(repo, target, "probe-branch")

    assert target.exists()
    listing = _git(repo, "worktree", "list").stdout
    assert str(target) in listing
    got = subprocess.run(
        ["git", "-C", str(target), "rev-parse", "HEAD"],
        capture_output=True, text=True,
    )
    assert got.returncode == 0, f"worktree has no resolvable HEAD:\n{got.stderr}"
    assert got.stdout.strip() == parent_sha, (
        "worktree should have branched from the repo's commit, "
        f"expected {parent_sha}, got {got.stdout.strip()}"
    )


def test_ensure_git_worktree_uses_existing_branch_without_start_point(tmp_path: Path):
    """When the branch already exists, the positional form must still be used."""
    repo = tmp_path / "real-repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True, capture_output=True)
    _commit(repo, "init")
    _git(repo, "branch", "existing-branch")
    target = repo / ".worktrees" / "t_existing"

    kbw._ensure_git_worktree(repo, target, "existing-branch")

    assert target.exists()
    assert str(target) in _git(repo, "worktree", "list").stdout


def test_ensure_git_worktree_is_idempotent_on_repeated_provisioning(unborn_repo: Path):
    """Re-provisioning an existing task must not blow up.

    Dispatch retries the same card, so this runs twice against one repo.
    """
    target = unborn_repo / ".worktrees" / "t_probe"

    kbw._ensure_git_worktree(unborn_repo, target, "probe-branch")
    kbw._ensure_git_worktree(unborn_repo, target, "probe-branch")

    assert str(target) in _git(unborn_repo, "worktree", "list").stdout