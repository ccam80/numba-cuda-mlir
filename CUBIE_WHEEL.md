# cubie-numba-cuda-mlir maintenance runbook

This branch (`cubie-wheel`) produces the `cubie-numba-cuda-mlir`
PyPI distribution: NVIDIA's numba-cuda-mlir plus the native-code
(C++) fixes CuBIE needs that are still open as upstream pull
requests. CuBIE's `mlir*` extras install it in place of the stock
wheel. The import package is unchanged (`numba_cuda_mlir`), so the
two distributions must never share an environment.

CuBIE-side context lives in cubie's
`docs/source/developer_guide/mlir_patched_wheel.rst`; this file is
the fork-side runbook.

## Branch anatomy

`cubie-wheel` = an upstream `main` base, plus:

1. A union merge of the **native-code** PR branches only:

   | Upstream PR | Branch | Binary target |
   |---|---|---|
   | NVIDIA#217 | `fix-llvm70-lit-checks` | MLIRToLLVM70 |
   | NVIDIA#219 | `fix-llvm70-nvvm-ir-version` | MLIRToLLVM70 |
   | NVIDIA#221 | `selective-fastmath-pr` | MLIRToLLVM70 |
   | NVIDIA#225 | `fix-lineinfo-multi-file-pr` | MLIRToLLVM70 |
   | NVIDIA#233 | `fix-async-launch-raise-free-kernels-pr` | `_cext` launcher |

   Python-side upstream PRs stay **out** of this branch: cubie
   applies those at runtime via `cubie._mlir_compat`, which
   feature-detects the installed build and no-ops once a fix is
   present natively. Adding them here buys nothing and multiplies
   merge conflicts.

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
for b in fix-llvm70-lit-checks fix-llvm70-nvvm-ir-version \
         selective-fastmath-pr fix-lineinfo-multi-file-pr \
         fix-async-launch-raise-free-kernels-pr; do
  git merge --no-edit origin/$b
done
git cherry-pick <packaging commits from old cubie-wheel>
```

- **Drop any branch whose upstream PR has merged** — the union then
  simply contains upstream's version, and `_mlir_compat`/the wheel
  tolerate either state.
- If a PR branch no longer merges cleanly, rebase that branch onto
  upstream main first (it needs it for the upstream PR anyway).
- Bump `VERSION`, force-push the recreated branch to `cubie-wheel`.

## Routine: add a new native-code patch

1. Develop the fix on its own branch off upstream main; open the
   upstream PR (with user approval, per project policy).
2. Merge the branch into `cubie-wheel`, add a row to the table
   above and to the `NOTICE` PR list, bump `VERSION`.
3. Build, validate, publish (below). Python-side fixes go into
   `cubie._mlir_compat` instead, with feature detection.

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
pytest tests/test_async_launch.py tests/test_kernel_exceptions.py \
       tests/test_descriptor_launch_config.py tests/test_lineinfo.py \
       tests/test_math.py tests/numba_cuda_tests/cudapy/test_fastmath.py \
       --override-ini="addopts="
```

Reference result (0.4.1.1, RTX 4070 SUPER): cubie suite 2855
passed / 0 failed; fork tests 152 passed / 2 xfailed.

## Publish

*Actions → Build cubie wheels → Run workflow* on `cubie-wheel` with
`publish` ticked. Publishing uses PyPI trusted publishing: the
`cubie-numba-cuda-mlir` project on PyPI trusts this repository,
workflow `cubie-wheels.yml`, environment `pypi`. No tokens are
stored anywhere.

If cubie's dependency floor moves (`cubie-numba-cuda-mlir>=X.Y.Z.N`
in its `mlir*` extras), publish the wheels **before** merging the
cubie-side PR — cubie's `mlir-extras-resolve` CI job resolves the
extras against the live index and stays red until they exist.

## Retirement

When every native-code PR has merged upstream and NVIDIA ships a
release containing them, point cubie's `mlir*` extras back at
`numba-cuda-mlir` with the appropriate minimum version and stop
publishing. The runtime shims in `cubie._mlir_compat` already
no-op on such a release.
