"""Prepare a fresh git worktree of numba-cuda-mlir for agent work.

Orca runs this from the new worktree after ``worktree create`` (see
``orca.yaml``); it also runs by hand from any worktree::

    python toolshed/worktree_setup.py

The package needs untracked build products next to its sources: the
``_mlir`` bindings tree and the compiled extension modules. Both are
copied from the main checkout's ``src/numba_cuda_mlir`` (no junctions,
so a bridge rebuilt for one branch never leaks into another).

The main checkout's ``.venv`` holds the dependencies and an editable
install that resolves to the main checkout. Rather than rebuild that
environment, each worktree gets a thin venv on the same base
interpreter whose ``.pth`` puts ``<worktree>/src`` ahead of the main
venv's site-packages, so ``import numba_cuda_mlir`` resolves inside the
worktree with no ``PYTHONPATH`` discipline. A ``sitecustomize`` in that
venv exports ``LIBLLVM7`` from the main checkout's ``llvm7-install``.

Environment:

``ORCA_WORKTREE_PATH``
    The worktree to prepare (default: this file's repo root).
``ORCA_ROOT_PATH``
    The main checkout (default: the owner of the shared ``.git``).
"""

import configparser
import os
import shutil
import subprocess
import sys
from pathlib import Path

PACKAGE = Path("src") / "numba_cuda_mlir"
EXTENSION_SUFFIXES = (".pyd", ".so", ".dylib")
LOCAL_FILES = (
    Path(".claude") / "settings.local.json",
    Path("CLAUDE.local.md"),
)


def run(cmd, **kwargs):
    print("+", " ".join(str(part) for part in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kwargs)


def worktree_path():
    override = os.environ.get("ORCA_WORKTREE_PATH")
    if override:
        return Path(override).resolve()
    return Path(__file__).resolve().parents[1]


def root_path(worktree):
    override = os.environ.get("ORCA_ROOT_PATH")
    if override:
        return Path(override).resolve()
    common = subprocess.run(
        ["git", "rev-parse", "--path-format=absolute", "--git-common-dir"],
        cwd=worktree, check=True, capture_output=True, text=True,
    ).stdout.strip()
    return Path(common).resolve().parent


def venv_python(venv):
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def base_interpreter(root):
    """Interpreter the main checkout's .venv was built from."""
    config = root / ".venv" / "pyvenv.cfg"
    if config.is_file():
        parser = configparser.ConfigParser()
        parser.read_string("[venv]\n" + config.read_text(encoding="utf-8"))
        section = parser["venv"]
        executable = section.get("executable")
        if executable and Path(executable).is_file():
            return Path(executable)
        home = section.get("home")
        if home:
            for name in ("python.exe", "python3", "python"):
                candidate = Path(home) / name
                if candidate.is_file():
                    return candidate
    print(f"no usable {config}; falling back to {sys.executable}")
    return Path(sys.executable)


def site_packages(python):
    return Path(subprocess.run(
        [str(python), "-c",
         "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
        check=True, capture_output=True, text=True,
    ).stdout.strip())


def copy_build_products(root, worktree):
    source = root / PACKAGE
    target = worktree / PACKAGE
    bindings = source / "_mlir"
    if not bindings.is_dir():
        raise SystemExit(f"{bindings} is missing; restore it per "
                         "CLAUDE.local.md before creating worktrees")
    shutil.copytree(bindings, target / "_mlir", dirs_exist_ok=True)
    print(f"copied {bindings} -> {target / '_mlir'}")
    extensions = [path for path in source.iterdir()
                  if path.suffix.lower() in EXTENSION_SUFFIXES]
    for path in extensions:
        shutil.copy2(path, target / path.name)
    print(f"copied {len(extensions)} extension modules")


def find_libllvm7(root):
    install = root / "llvm7-install"
    patterns = ("bin/LLVM-C.dll", "lib/libLLVM-C.so*", "lib/libLLVM*.so*",
                "lib/libLLVM*.dylib")
    for pattern in patterns:
        matches = sorted(install.glob(pattern))
        if matches:
            return matches[0]
    return None


def build_layered_venv(root, worktree, interpreter):
    venv = worktree / ".venv"
    python = venv_python(venv)
    if not python.is_file():
        run([str(interpreter), "-m", "venv", "--without-pip", str(venv)])
    root_site = site_packages(venv_python(root / ".venv"))
    layer = site_packages(python)
    layer.mkdir(parents=True, exist_ok=True)
    pth = layer / "zz_worktree_layer.pth"
    pth.write_text(f"{worktree / 'src'}\n{root_site}\n", encoding="utf-8")
    print(f"wrote {pth}")
    libllvm7 = find_libllvm7(root)
    if libllvm7 is None:
        print(f"no LLVM-C library under {root / 'llvm7-install'}; "
              "LIBLLVM7 left unset")
    else:
        custom = layer / "sitecustomize.py"
        custom.write_text(
            "import os\n"
            f"os.environ.setdefault('LIBLLVM7', {str(libllvm7)!r})\n",
            encoding="utf-8")
        print(f"wrote {custom} (LIBLLVM7={libllvm7})")
    return python


def copy_local_files(root, worktree):
    """Gitignored machine-specific files travel with the worktree."""
    for relative in LOCAL_FILES:
        source = root / relative
        target = worktree / relative
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"copied {source} -> {target}")


def verify(python, worktree):
    probe = ("import numba_cuda_mlir, os, sys; "
             "print(numba_cuda_mlir.__file__); print(sys.executable); "
             "print(os.environ.get('LIBLLVM7', ''))")
    result = subprocess.run([str(python), "-c", probe], check=True,
                            capture_output=True, text=True, cwd=worktree)
    module_file, executable, libllvm7 = result.stdout.splitlines()
    resolved = Path(module_file).resolve()
    if worktree not in resolved.parents:
        raise SystemExit(
            f"numba_cuda_mlir resolves to {resolved}, not inside {worktree}")
    print(f"package    {resolved}")
    print(f"python     {executable}")
    print(f"LIBLLVM7   {libllvm7}")


def main():
    worktree = worktree_path()
    root = root_path(worktree)
    print(f"worktree   {worktree}")
    print(f"root       {root}")
    interpreter = base_interpreter(root)
    print(f"base       {interpreter}")
    copy_build_products(root, worktree)
    python = build_layered_venv(root, worktree, interpreter)
    copy_local_files(root, worktree)
    verify(python, worktree)


if __name__ == "__main__":
    main()
