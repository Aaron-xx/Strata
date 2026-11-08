# Upstream sync, robot testing and the bug-fix loop

This fork (Aaron-xx/Strata) tracks upstream Niko1221/Strata. Two workflows
form an automated loop. Both files only take effect once they are merged into
the DEFAULT branch (main): GitHub runs scheduled workflows from the default
branch only. On a topic branch they are inert, so pushing them starts no timer.

## 1. Upstream sync (upstream-sync.yml)
Nightly and on manual dispatch: fetch Niko1221/Strata main and git merge it
into this fork main, then push. Using a merge instead of GitHub's "Sync a fork
branch" REST endpoint is deliberate: that endpoint only fast-forwards, so once
main carries our fork-only CI commits it would return 409 Conflict forever.
The merge keeps our four fork-only files (their paths do not overlap upstream)
and still brings in every upstream commit.

A push made with GITHUB_TOKEN does NOT trigger other workflows (GitHub loop
prevention). The sync job therefore dispatches upstream-robot-test.yml itself
after a changed merge. The nightly 03:41 robot run is a backstop either way.

## 2. Robot smoke test (upstream-robot-test.yml)
On pushes/PRs to main, nightly and on manual dispatch, runs the SAME commands
upstream runs in its own CI, so a red run means "upstream broke something",
not "our harness invented a failing test":
  - python -m pip install -r requirements.txt   (terminal-minimal-server.yml)
  - python -m compileall -q strata.py strata_cli strata_control serve tools
  - python -m unittest discover -s serve -p "test_*.py"  (terminal-minimal-server.yml)
  - node --test .github/scripts/item-check.test.js       (scripts-test.yml)
If any step fails the job opens one issue per SHA titled "[robot] ..." with the
run URL. That issue is the bug report. The workflow asks for issues: write
explicitly because this repo's token default is read.

We deliberately do NOT sweep tools/test_setup_*.py with a hand-written loop.
Measured on 2026-11-08 (63 files, conda python 3.x, repo root on PYTHONPATH):
60 pass; 3 fail for reasons unrelated to upstream health:
  - test_setup_download_length.py: "from tools.test_setup_download import ..."
    needs PYTHONPATH=. (or the -m module form); a bare script invocation is
    not how upstream calls it.
  - test_setup_download.py: test_a_big_file_comes_over_several_connections
    asserts each of 41 pieces is requested once, but the local mock recorded
    piece 0 twice (a timing-dependent retry). Flaky; not a stable gate.
  - test_setup_dsv41.py: still running after 90 s, killed by our probe timeout.
Upstream CI does not run these either; scripts-test.yml only runs the node
test and terminal-minimal-server.yml only the serve unittest discovery.

## 3. The bug-fix loop
robot test fails -> issue opened (report) -> a fix is committed on a topic
branch -> PR -> robot test re-runs -> PASS -> merge. The loop is deliberately
small: it only watches checks a GitHub-hosted runner can actually run.

## Known limits (what this workflow does NOT guarantee)
- The 6 GB / DSH serve regression gate (tools/dsh_serve_regression.py --runs 3)
  needs the running 0.0.0.0:8000 service on the GTX 1660 Ti host. A
  GitHub-hosted runner has neither that GPU nor that service, so the Live gate
  is not run here. Run it on the host itself, or add a self-hosted runner there
  (a separate change, not part of this skeleton).
- The node test could not be verified on the remote host (no node binary
  installed there); it is upstream's own command and runs under setup-node on
  GitHub-hosted runners.
- The commands here are a COPY of upstream's workflow steps. If upstream
  renames a test path or adds a required dependency, this job may need the
  same edit. The nightly sync will otherwise keep running an older command.
- If upstream ever changes one of our four fork-only paths, the sync merge can
  conflict. The job then fails with a red run; fix the conflict by hand.

## Activation checklist (measured on this fork, 2026-11-08)
Before merging, the read-only state was:
  actions/permissions            enabled=true, allowed_actions=all
  actions/permissions/workflow   default_workflow_permissions=read
  actions/workflows              0
  repo has_issues                false
  repo open issues               0
Actions is therefore ALREADY ENABLED on this fork; an empty workflows list
means no workflow has been registered yet, not that Actions is off. The first
push to main registers and runs them. The repo-level token default is read,
which is why BOTH workflows declare their own permissions block
(contents/actions: write / contents: read + issues: write).

## Verification (measured on this fork, 2026-11-08; both hand steps DONE)
  - git push ci/upstream-robot -> fork main: 146df3c4..3c230977, a clean
    fast-forward. The four fork-only files are on the default branch, so the
    nightly timers are live.
  - gh api -X PATCH repos/Aaron-xx/Strata -F has_issues=true -> true; the
    issue tracker is on and gh issue create no longer returns 410.
  - gh api actions/workflows: 10 active, including "Sync fork with upstream"
    and "Upstream robot smoke test". Cron 03:17 UTC (sync) / 03:41 UTC (robot).
  - manual workflow_dispatch, "Sync fork with upstream" run 42471114640:
    success. "Merge upstream main into this fork main" success; "Kick the
    robot test on the new main" skipped because fork main already contains
    upstream 146df3c4 as an ancestor (changed=false, so no push and no
    dispatch). This is the intended no-op when nothing new arrived.
  - manual workflow_dispatch, "Upstream robot smoke test" run 42471122200:
    success in 10 min 54 s (10:21:59Z -> 10:32:53Z). All four upstream
    commands passed, including the full serve unittest discovery (81 files)
    and the node item-check test. "Report a bug when the smoke test fails"
    correctly skipped; no issue was opened.
  - The push-triggered robot run 42471064684 was cancelled by the workflow's
    own concurrency group when the manual dispatch took over. Expected.
  - strata-serve on the 6 GB host stayed active / 5672 MiB throughout.
Remaining unverified: the failure -> issue branch has not been triggered on
purpose, because doing so would publish a real issue on this fork. To close
that loop, push a deliberately failing test on a topic branch (not main) or
temporarily break a step, then confirm exactly one deduplicated issue and
close it afterwards.
