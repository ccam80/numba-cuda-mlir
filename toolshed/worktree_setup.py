"""Stage a worktree's build products and a venv that imports it.

Copies ``_mlir`` and the extension modules from the main checkout; the
``.venv`` ``.pth`` lists ``<worktree>/src`` before the main venv's
site-packages and ``sitecustomize`` sets ``LIBLLVM7``. A ``.venv`` on
another interpreter is rebuilt. Env: ``ORCA_WORKTREE_PATH``,
``ORCA_ROOT_PATH``.
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


def venv_config(venv):
    """The ``pyvenv.cfg`` keys of ``venv``, or ``None`` without one."""
    config = venv / "pyvenv.cfg"
    if not config.is_file():
        return None
    parser = configparser.ConfigParser()
    try:
        parser.read_string("[venv]\n" + config.read_text(encoding="utf-8"))
    except configparser.Error as error:
        print(f"unreadable {config}: {error}")
        return None
    return parser["venv"]


def base_interpreter(root):
    """Interpreter the main checkout's .venv was built from."""
    section = venv_config(root / ".venv")
    if section is not None:
        executable = section.get("executable")
        if executable and Path(executable).is_file():
            return Path(executable)
        home = section.get("home")
        if home:
            for name in ("python.exe", "python3", "python"):
                candidate = Path(home) / name
                if candidate.is_file():
                    return candidate
    print(f"no usable {root / '.venv' / 'pyvenv.cfg'}; "
          f"falling back to {sys.executable}")
    return Path(sys.executable)


def venv_matches(venv, interpreter):
    """Whether ``venv`` records ``interpreter``'s directory as its home."""
    section = venv_config(venv)
    if section is None:
        return False
    home = section.get("home")
    if not home:
        return False
    return Path(home).resolve() == interpreter.parent.resolve()


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
    if venv.exists() and not venv_matches(venv, interpreter):
        print(f"rebuilding {venv}: not built on {interpreter}")
        shutil.rmtree(venv)
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
    """Copy the gitignored machine-specific files into the worktree."""
    for relative in LOCAL_FILES:
        source = root / relative
        target = worktree / relative
        if source.is_file() and not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"copied {source} -> {target}")


def python_version(python):
    result = subprocess.run(
        [str(python), "-c", "import sys; print(sys.version.split()[0])"],
        check=True, capture_output=True, text=True,
    )
    return result.stdout.strip()


def verify(python, interpreter, worktree):
    probe = ("import numba_cuda_mlir, os, sys; "
             "print(numba_cuda_mlir.__file__); print(sys.executable); "
             "print(os.environ.get('LIBLLVM7', ''))")
    result = subprocess.run([str(python), "-c", probe], check=True,
                            capture_output=True, text=True, cwd=worktree)
    module_file, executable, libllvm7 = result.stdout.strip().splitlines()
    resolved = Path(module_file).resolve()
    if worktree not in resolved.parents:
        raise SystemExit(
            f"numba_cuda_mlir resolves to {resolved}, not inside {worktree}")
    version = python_version(python)
    base_version = python_version(interpreter)
    if version != base_version:
        raise SystemExit(
            f"venv python is {version}; base {interpreter} is "
            f"{base_version}")
    print(f"package    {resolved}")
    print(f"python     {executable} ({version})")
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
    verify(python, interpreter, worktree)


if __name__ == "__main__":
    main()
