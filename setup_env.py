#!/usr/bin/env python3
"""
Bootstrap a local Python environment for this course repository.

- Creates a local venv (default: .venv/)
- Installs base requirements from requirements.txt
- Optionally installs grouped extras from requirements-optional.txt
- Verifies a small set of base imports with --check
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent

OPTIONAL_GROUPS = {
    "llm": {
        "openai",
        "anthropic",
        "sentence-transformers",
        "transformers",
        "bertopic",
        "gensim",
        "langchain",
        "langchain-openai",
        "plotly",
        "pyLDAvis",
    },
    "torch": {
        "torch",
        "pytorch-tabnet",
        "umap-learn",
        "hdbscan",
    },
    "boosting": {
        "xgboost",
        "lightgbm",
        "optuna",
        "shap",
    },
    "timeseries": {
        "statsmodels",
        "pmdarima",
        "prophet",
        "nixtla",
    },
    "survival": {
        "lifelines",
        "scikit-survival",
        "pycox",
    },
    "graph": {
        "networkx",
        "python-louvain",
        "node2vec",
        "scikit-surprise",
        "torch-geometric",
        "torch-scatter",
        "torch-sparse",
        "pykeen",
        "neo4j",
    },
}


def run(cmd: list[str], *, cwd: Path | None = None) -> None:
    subprocess.check_call(cmd, cwd=str(cwd) if cwd else None)


def venv_python(venv_dir: Path) -> Path:
    if os.name == "nt":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def read_requirements(path: Path) -> list[str]:
    if not path.exists():
        return []
    reqs: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(line)
    return reqs


def base_package_name(requirement: str) -> str:
    for sep in ("<=", ">=", "==", "~=", "!=", "<", ">"):
        if sep in requirement:
            return requirement.split(sep, 1)[0].strip()
    return requirement.strip()


def selected_optional_requirements(selected_groups: list[str], optional_reqs: list[str]) -> list[str]:
    if "all" in selected_groups:
        return optional_reqs

    selected_names: set[str] = set()
    for group in selected_groups:
        selected_names.update(OPTIONAL_GROUPS.get(group, set()))

    chosen: list[str] = []
    for req in optional_reqs:
        if base_package_name(req) in selected_names:
            chosen.append(req)
    return chosen


def create_venv(venv_dir: Path) -> None:
    if venv_dir.exists():
        return
    print(f"[1/4] Creating venv at: {venv_dir}")
    builder = venv.EnvBuilder(with_pip=True)
    builder.create(str(venv_dir))


def install_packages(python_exe: Path, packages: list[str], *, upgrade_pip: bool) -> None:
    if upgrade_pip:
        print("[2/4] Upgrading pip...")
        run([str(python_exe), "-m", "pip", "install", "--upgrade", "pip"])
    if not packages:
        return
    print("[3/4] Installing packages...")
    run([str(python_exe), "-m", "pip", "install", *packages])


def check_imports(python_exe: Path) -> None:
    print("[4/4] Verifying key imports...")
    code = r"""
import importlib
import sys

mods = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "sklearn",
    "scipy",
    "tqdm",
    "dotenv",
]

failed = []
for mod in mods:
    try:
        importlib.import_module(mod)
        print(f"OK  {mod}")
    except Exception as exc:
        failed.append((mod, repr(exc)))
        print(f"FAIL {mod}: {exc}")

if failed:
    print("\\nSome packages failed to import:")
    for mod, err in failed:
        print(f"- {mod}: {err}")
    raise SystemExit(1)

print("\\nAll base imports look good.")
print("python:", sys.version.split()[0])
"""
    run([str(python_exe), "-c", code])


def main() -> int:
    parser = argparse.ArgumentParser(description="Set up Python environment for this course repo.")
    parser.add_argument("--venv-dir", default=".venv", help="Virtualenv directory (default: .venv)")
    parser.add_argument("--skip-venv", action="store_true", help="Install into current Python instead of a venv.")
    parser.add_argument(
        "--extras",
        action="append",
        default=[],
        help="Optional groups: llm, torch, boosting, timeseries, survival, graph, all",
    )
    parser.add_argument("--no-upgrade-pip", action="store_true", help="Skip pip upgrade step.")
    parser.add_argument("--check", action="store_true", help="Only verify base imports.")
    args = parser.parse_args()

    print("=" * 80)
    print("setup_env.py: course environment bootstrap")
    print("=" * 80)
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    if sys.version_info < (3, 10):
        print("ERROR: Python 3.10+ is required.")
        return 2

    venv_dir = (ROOT / args.venv_dir).resolve()
    use_venv = not args.skip_venv

    if use_venv:
        create_venv(venv_dir)
        python_exe = venv_python(venv_dir)
    else:
        python_exe = Path(sys.executable)
        print("WARNING: --skip-venv selected; using current Python environment.")

    if not python_exe.exists():
        print(f"ERROR: Python executable not found: {python_exe}")
        return 2

    base_reqs = read_requirements(ROOT / "requirements.txt")
    optional_reqs = read_requirements(ROOT / "requirements-optional.txt")
    extras = [value.strip().lower() for value in args.extras if value.strip()]

    unknown = [group for group in extras if group not in OPTIONAL_GROUPS and group != "all"]
    if unknown:
        print(f"ERROR: Unknown extras group(s): {', '.join(sorted(unknown))}")
        return 2

    if args.check:
        check_imports(python_exe)
        return 0

    selected_optional = selected_optional_requirements(extras, optional_reqs)
    packages = base_reqs + selected_optional

    print(f"\nTarget python: {python_exe}")
    print(
        f"Will install {len(packages)} packages "
        f"({len(base_reqs)} base, {len(selected_optional)} optional)."
    )

    install_packages(python_exe, packages, upgrade_pip=not args.no_upgrade_pip)
    check_imports(python_exe)

    print("\nNext steps:")
    if use_venv:
        if os.name == "nt":
            print(r"- Activate: .\.venv\Scripts\Activate.ps1")
        else:
            print("- Activate: source .venv/bin/activate")
    print("- Review lecture/STYLE_GUIDE.md before writing lecture notes.")
    print("- Run a practice script, e.g.: python practice/chapter01/code/1-4-llm-eda.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
