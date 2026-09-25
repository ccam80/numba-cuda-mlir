# cubie-numba-cuda-mlir maintenance runbook

This branch (`cubie-wheel`) produces the `cubie-numba-cuda-mlir`
PyPI distribution: NVIDIA's numba-cuda-mlir plus the native-code
(C++) fixes CuBIE needs that are still open as upstream pull
requests. CuBIE's `mlir*` extras install it in place of the stock
wheel. The import package is unchanged (`numba_cuda_mlir`), so the
two distributions must never share an environment.

CuBIE-side context lives in cubie's `src/cubie/backend/AGENTS.md`;
this file is the fork-side runbook.

## Branch anatomy

`cubie-wheel` = an upstream `main` base, plus:

1. A union merge of the **native-code** PR branches only:

   | PR | Branch | Target |
   |---|---|---|
   | NVIDIA#255 | `selective-fastmath` | MLIRToLLVM70 |
   | NVIDIA#225 | `fix-lineinfo-multi-file-pr` | MLIRToLLVM70 |
   | NVIDIA#298 | `feat/loop-unroll-hints` | MLIRToLLVM70 + Python typing/lowering |
   | NVIDIA#333 | `kernel-dispatcher-gc-pr` | `_cext` launcher GC support |
   | ccam80#6 | `codex/lean-typed-scheduler` | Python typed-planner hook |

   Python-side upstream PRs stay **out** of this branch: cubie
   carries them as shims generated against this exact wheel (routine
   below). Exception: `register_typed_planner` must exist in the
   installed wheel, so the typed-planner hook rides here.

2. The packaging commits: distribution rename in `pyproject.toml`,
   version in `src/numba_cuda_mlir/VERSION`, provenance paragraph in
   `NOTICE`, the `.github/workflows/cubie-wheels.yml` workflow, and
   this file.

## Versioning

`<upstream release>.<patch iteration>` — `0.4.1.1` is the first
patched build of the 0.4.1 era. Bump the fourth component for a new
build against the same upstream release; when upstream releases
`X.Y.Z`, the next build is `X.Y.Z.1`. Edit
`src/numba_cuda_mlir/VERSION` (PyPI rejects re-uploads of an
existing version, and local versions like `+cubie1` are not allowed
on PyPI).

## Routine: pull in new upstream main

```bash
git fetch upstream
git checkout cubie-wheel
git rebase --onto upstream/main <old-base> cubie-wheel   # or re-create:
```

Re-creating from scratch is usually cleaner than rebasing the merge
knot:

```bash
git checkout -b cubie-wheel-next upstream/main
for b in selective-fastmath fix-lineinfo-multi-file-pr \
         feat/loop-unroll-hints kernel-dispatcher-gc-pr; do
  git merge --no-edit origin/$b
done
# typed-planner hook: apply only its own commits
git diff f51537c origin/codex/lean-typed-scheduler | git apply --index
git commit -m "feat: typed whole-function planner hook with cache-safety contract"
# packaging files from the previous wheel
git diff <old-base> <old cubie-wheel> -- .github/workflows/cubie-wheels.yml \
  CUBIE_WHEEL.md NOTICE pyproject.toml | git apply --index -3
```

- **Drop any branch whose upstream PR has merged**; the union then
  contains upstream's version.
- If a PR branch no longer merges cleanly, rebase that branch onto
  upstream main first (it needs it for the upstream PR anyway).
- Bump `VERSION`, tag the outgoing `cubie-wheel` head as
  `cubie-wheel-<old version>` and push the tag, then force-push the
  recreated branch to `cubie-wheel`. To patch a published build,
  branch from its tag.

A rebuild request means the full cycle: recreate the branch, build in
CI, sync cubie's shims, validate a built wheel in a fresh cubie env
(suites below), publish, and open the cubie PR. Stop before
publishing only when validation fails.

## Routine: sync cubie's shims

cubie's `src/cubie/backend/_mlir_compat.py` has one section per open
Python-side PR cubie uses, matching that PR as merged onto this wheel.
In a cubie branch off `main`, by hand:

1. Delete the sections of PRs that merged.
2. Update each remaining section to the PR's current code.
3. Add a section for each new Python-side PR cubie uses.
4. Pin cubie's `mlir*` extras to `==<version>`.

## Routine: add a new native-code patch

1. Develop the fix on its own branch off upstream main; open the
   upstream PR (with user approval, per project policy).
2. Merge the branch into `cubie-wheel`, add a row to the table
   above and to the `NOTICE` PR list, bump `VERSION`.
3. Build, validate, publish (below). Python-side fixes become cubie
   shims instead (routine above).

## Build

Every push to `cubie-wheel` builds the full matrix (12 wheels:
cp311–cp314 × manylinux_2_28 x86_64 / manylinux_2_28 aarch64 /
win_amd64; cp314t omitted — cubie has no free-threaded support).
Append `[skip ci]` to a commit subject for docs-only pushes.

The build does **not** compile LLVM. The `find-llvm` job locates the
newest successful NVIDIA CI run on their `main` whose
`ci/llvm-version.env` matches this branch's, and the build jobs
download that run's `llvm-modern-install-*`/`llvm7-install-*`
artifacts (refreshed on every upstream push, ~90-day retention).
Consequences:

- **If upstream bumps the LLVM pin**, this branch must be rebased so
  the pins match a run that still has live artifacts — `find-llvm`
  fails loudly ("No NVIDIA CI run … matching") when they don't.
- A specific run can be forced with the `upstream-run-id` dispatch
  input.
- If the default workflow token stops being able to download the
  cross-repo artifacts, add a fine-grained PAT with public-repository
  read access as the `NVIDIA_ACTIONS_READ_TOKEN` repository secret.

## Validate before publishing

Download a wheel artifact from the run and, in a fresh venv on a
CUDA machine:

```bash
pip install "<wheel>[cu12]" cupy-cuda12x
pip install -e "<cubie checkout>[test]"
# cubie's real-GPU suite (from the cubie checkout):
pytest -m "not specific_algos and not sim_only" --no-cov
# fork targeted tests (from this repo; needs filecheck + pytest-benchmark):
pip install filecheck pytest-benchmark
pytest tests/test_kernel_exceptions.py \
       tests/test_descriptor_launch_config.py tests/test_lineinfo.py \
       tests/test_math.py tests/numba_cuda_tests/cudapy/test_fastmath.py \
       --override-ini="addopts="
```

Reference result (0.5.3.1, RTX 4070 SUPER, CUDA 13): cubie 3800/0, fork 247/2xf (plus `tests/test_dispatcher_lifetime.py` and `tests/test_loop_unroll.py`).

## Publish

*Actions → Build cubie wheels → Run workflow* on `cubie-wheel` with
`publish` ticked. Publishing uses PyPI trusted publishing: the
`cubie-numba-cuda-mlir` project on PyPI trusts this repository,
workflow `cubie-wheels.yml`, environment `pypi`. No tokens are
stored anywhere.

Publish the wheels **before** merging the cubie shim PR that pins
them: cubie's `mlir-extras-resolve` CI job resolves the extras against
the live index and stays red until they exist.

## Retirement

When every native-code PR has merged upstream and NVIDIA ships a
release containing them, point cubie's `mlir*` extras back at
`numba-cuda-mlir` pinned to that release, sync the shims against it,
and stop publishing.
