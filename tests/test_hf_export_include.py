"""The Hugging Face export vendors a hand-listed subset of `spectre` into the published repo.

If a module reachable from `spectre.model` is missing from `INCLUDE`, the export still succeeds
and the failure only appears for whoever runs
`AutoModel.from_pretrained('cclaess/SPECTRE-Large', trust_remote_code=True)`. These tests catch
that here instead, without doing an export or touching the network.
"""
import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC_PACKAGE = ROOT / "src" / "spectre"
sys.path.insert(0, str(ROOT / "scripts"))

export_hf = pytest.importorskip("export_hf", reason="scripts/export_hf.py must be importable")


def _spectre_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "spectre":
                found.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "spectre":
                    found.add(alias.name)
    return found


def _reachable_from(entry: Path) -> set[Path]:
    """Files reachable from `entry` by absolute `spectre.*` imports."""
    seen, queue = set(), [entry]
    while queue:
        path = queue.pop()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for module in _spectre_imports(path):
            rel = Path(*module.split(".")[1:])
            for candidate in (SRC_PACKAGE / rel.with_suffix(".py"), SRC_PACKAGE / rel / "__init__.py"):
                if candidate.exists():
                    queue.append(candidate)
                    break
    return seen


def test_model_py_imports_are_all_exported():
    exported = export_hf._exported_modules()
    for module in _spectre_imports(SRC_PACKAGE / "model.py"):
        assert module in exported, (
            f"spectre/model.py imports {module!r}, which scripts/export_hf.py does not copy. "
            f"Add it to INCLUDE or the published Hub model will fail to import."
        )


def test_everything_reachable_from_model_is_exported():
    exported = export_hf._exported_modules()
    for path in _reachable_from(SRC_PACKAGE / "model.py"):
        rel = path.relative_to(SRC_PACKAGE).with_suffix("")
        parts = [p for p in rel.parts if p != "__init__"]
        module = ".".join(["spectre", *parts]) if parts else "spectre"
        assert module in exported, (
            f"{module!r} is reachable from spectre/model.py but is not in export_hf.INCLUDE."
        )


def test_windowing_and_presets_are_exported():
    """Both are imported by model.py, so both must ship with the Hub model."""
    exported = export_hf._exported_modules()
    assert "spectre.windowing" in exported
    assert "spectre.presets" in exported


def test_io_and_cli_are_not_exported():
    """These need the optional [inference] deps, which the Hub model must not require."""
    exported = export_hf._exported_modules()
    assert "spectre.io" not in exported
    assert "spectre.cli" not in exported


def test_windowing_imports_only_torch_and_stdlib():
    """windowing.py is vendored, so a stray dependency would break the Hub model."""
    tree = ast.parse((SRC_PACKAGE / "windowing.py").read_text(encoding="utf-8"))
    allowed = {"torch", "warnings", "typing", "__future__", "math"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, f"windowing.py imports {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] in allowed, f"windowing.py imports from {node.module}"


def test_presets_imports_only_stdlib():
    tree = ast.parse((SRC_PACKAGE / "presets.py").read_text(encoding="utf-8"))
    allowed = {"copy", "difflib", "dataclasses", "typing", "__future__"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] in allowed, f"presets.py imports {alias.name}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] in allowed, f"presets.py imports from {node.module}"


def test_model_py_does_not_import_optional_deps():
    """model.py must import cleanly on a base install (no monai/nibabel)."""
    forbidden = {"monai", "nibabel", "SimpleITK", "pandas", "spectre.io", "spectre.cli"}
    for module in _spectre_imports(SRC_PACKAGE / "model.py"):
        assert module not in forbidden, f"spectre/model.py must not import {module}"
    tree = ast.parse((SRC_PACKAGE / "model.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name.split(".")[0] not in forbidden
        elif isinstance(node, ast.ImportFrom) and node.module:
            assert node.module not in forbidden


def test_model_py_has_no_module_level_warning():
    """A module-level warn would fire inside every Hub trust_remote_code import."""
    tree = ast.parse((SRC_PACKAGE / "model.py").read_text(encoding="utf-8"))
    for node in tree.body:
        assert not (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "warn"
        ), "model.py warns at module level; move it inside the deprecated accessor"


def test_patch_init_drops_excluded_imports_and_all_entries(tmp_path):
    """The vendored __init__.py must not import or advertise what the export does not ship."""
    init = tmp_path / "__init__.py"
    init.write_text((SRC_PACKAGE / "__init__.py").read_text(encoding="utf-8"), encoding="utf-8")

    keep = {Path(name).stem for name in export_hf.INCLUDE if name != "__init__.py"}
    export_hf._patch_init(init, keep)
    tree = ast.parse(init.read_text(encoding="utf-8"))

    imported = {n.module.lstrip(".") for n in ast.walk(tree)
                if isinstance(n, ast.ImportFrom) and n.module}
    imported |= {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and not n.module
                 for a in n.names}
    assert "io" not in imported, "the vendored package must not import spectre.io"
    for shipped in ("model", "presets", "windowing"):
        assert shipped in imported

    exported = {e.value for n in ast.walk(tree) if isinstance(n, ast.Assign)
                for t in n.targets if isinstance(t, ast.Name) and t.id == "__all__"
                for e in n.value.elts if isinstance(e, ast.Constant)}
    for gone in ("load_ct", "load_and_window", "find_ct_files", "data", "transforms", "ssl"):
        assert gone not in exported, f"__all__ still advertises {gone!r}, which is not shipped"
    for kept in ("SpectreImageFeatureExtractor", "window_scan", "list_presets"):
        assert kept in exported


def test_patch_init_handles_forms_the_old_regex_could_not(tmp_path):
    """Parenthesised, aliased and indented imports must all be understood."""
    init = tmp_path / "__init__.py"
    init.write_text(
        '"""Docstring."""\n'
        "from .model import SpectreImageFeatureExtractor\n"
        "from .io import (\n"
        "    load_ct,\n"
        "    find_ct_files as _find,\n"
        ")\n"
        "from . import transforms\n"
        "__all__ = ['SpectreImageFeatureExtractor', 'load_ct', '_find', 'transforms']\n",
        encoding="utf-8",
    )
    export_hf._patch_init(init, {"model"})
    result = init.read_text(encoding="utf-8")

    assert "load_ct" not in result, "multi-line parenthesised import was not stripped"
    assert "_find" not in result, "aliased name was not stripped from __all__"
    assert "transforms" not in result
    assert "SpectreImageFeatureExtractor" in result
    assert "Docstring." in result, "the module docstring must survive"
    ast.parse(result)
