import os
import ast
import sys
import shutil
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SRC_PACKAGE = ROOT / "src" / "spectre"
EXPORT_DIR = ROOT / "hf_export"
DEST_PACKAGE = EXPORT_DIR / "spectre"

# Subset of the source package to copy — training code is intentionally excluded.
# Map name -> True  (copy the whole file/directory)
#          -> list  (copy only these filenames from that subdirectory)
# Any __init__.py is automatically patched to drop imports of excluded items.
# Anything model.py imports (directly or transitively) MUST be listed here, or the published
# model fails at trust_remote_code import time. tests/test_hf_export_include.py enforces this,
# and _verify_export() re-checks the copied tree before we upload anything.
# io.py and cli.py are deliberately excluded: they need the optional [inference] dependencies.
INCLUDE = {
    "__init__.py": True,
    "model.py": True,
    "presets.py": True,
    "windowing.py": True,
    "models": [
        "layers",
        "__init__.py",
        "vision_transformer.py",
        "vision_transformer_features.py",
    ],
    "utils": [
        "__init__.py",
        "_utils.py",
        "modeling.py",
    ],
}


def _exported_modules(include=INCLUDE, prefix="spectre"):
    """Every module path the export will contain, e.g. {'spectre.model', 'spectre.utils', ...}."""
    modules = {prefix}
    for name, what in include.items():
        stem = Path(name).stem
        if stem == "__init__":
            continue
        if what is True:
            modules.add(f"{prefix}.{stem}")
            if (SRC_PACKAGE / name).is_dir():
                for child in (SRC_PACKAGE / name).rglob("*.py"):
                    rel = child.relative_to(SRC_PACKAGE / name).with_suffix("")
                    parts = [p for p in rel.parts if p != "__init__"]
                    modules.add(".".join([prefix, stem, *parts]) if parts else f"{prefix}.{stem}")
        else:
            modules.add(f"{prefix}.{stem}")
            modules.update(_exported_modules({f: True for f in what}, f"{prefix}.{stem}"))
    return modules


def _spectre_imports(path):
    """Every `spectre.*` module a file imports, with the line it happens on."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            if node.module.split(".")[0] == "spectre":
                yield node.module, node.lineno
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == "spectre":
                    yield alias.name, node.lineno


def _verify_export(dest=None):
    """Fail loudly if anything we publish imports a `spectre.*` module we did not copy.

    Covers both the vendored package and the remote-code files beside it, which import spectre
    too (configuration_spectre pulls in spectre.presets and spectre.windowing). Without this the
    omission only surfaces as a ModuleNotFoundError for whoever runs
    AutoModel.from_pretrained(..., trust_remote_code=True) against the published repo.
    """
    dest = Path(dest) if dest is not None else DEST_PACKAGE
    available = {
        ".".join(["spectre", *[p for p in path.relative_to(dest).with_suffix("").parts if p != "__init__"]])
        for path in dest.rglob("*.py")
    }
    available.add("spectre")

    to_check = list(dest.rglob("*.py"))
    to_check += [p for p in dest.parent.glob("*.py")]  # configuration_spectre.py, modeling_spectre.py

    problems = []
    for path in sorted(to_check):
        for module, lineno in _spectre_imports(path):
            if module not in available:
                problems.append(f"{path.relative_to(dest.parent)}:{lineno} imports {module}")

    if problems:
        raise RuntimeError(
            "The export publishes files that import spectre modules it does not copy. Add them "
            "to INCLUDE in scripts/export_hf.py:\n  " + "\n  ".join(problems)
        )
    return available


def _imported_submodule(node):
    """The submodule a relative import pulls in, or None if it isn't one.

    Handles both spellings: `from .models import X` -> "models", and
    `from . import models` -> "models".
    """
    if not isinstance(node, ast.ImportFrom) or node.level < 1:
        return None
    if node.module:                       # from .models import X
        return node.module.split(".")[0]
    if len(node.names) == 1:              # from . import models
        return node.names[0].name
    return None


def _patch_init(init_path, keep_modules):
    """Drop imports of excluded submodules from an __init__.py, and their names from __all__.

    Parsed rather than pattern-matched, so multi-line and parenthesised imports, indentation and
    aliases all work without special cases. Comments are not preserved (ast.unparse drops them),
    which is fine for a generated file - docstrings survive.
    """
    tree = ast.parse(init_path.read_text(encoding="utf-8"))

    dropped_names = set()
    kept_body = []
    for node in tree.body:
        module = _imported_submodule(node)
        if module is not None and module not in keep_modules:
            # `from . import models` binds "models"; `from .x import a as b` binds "b".
            dropped_names.update(alias.asname or alias.name for alias in node.names)
            continue
        kept_body.append(node)
    tree.body = kept_body

    if dropped_names:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                continue
            if isinstance(node.value, (ast.List, ast.Tuple)):
                node.value.elts = [
                    e for e in node.value.elts
                    if not (isinstance(e, ast.Constant) and e.value in dropped_names)
                ]

    init_path.write_text(ast.unparse(tree) + "\n", encoding="utf-8")


def _copy_selective(src, dst, include):
    """Copy src to dst including only the items in include.

    include maps name -> True (copy everything) or list[str] (selective filenames).
    The __init__.py in dst is patched to drop imports of any excluded items.
    """
    dst.mkdir(parents=True, exist_ok=True)
    keep_modules = {Path(name).stem for name in include if name != "__init__.py"}
    for name, what in include.items():
        src_path = src / name
        dst_path = dst / name
        if what is True:
            if src_path.is_dir():
                shutil.copytree(src_path, dst_path, ignore=shutil.ignore_patterns("__pycache__"))
            else:
                shutil.copy2(src_path, dst_path)
        else:
            _copy_selective(src_path, dst_path, {f: True for f in what})
    init_path = dst / "__init__.py"
    if init_path.exists():
        _patch_init(init_path, keep_modules)


def get_args():
    parser = argparse.ArgumentParser(
        description="Export SPECTRE model to HuggingFace Hub format"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="Upload model to HuggingFace Hub (default: False for safety)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="cclaess/SPECTRE-Large",
        help="HuggingFace repo ID (default: cclaess/SPECTRE-Large)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="spectre-large",
        help="SPECTRE preset to export; must have published weights (default: spectre-large)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="HuggingFace token (if not provided, uses HF_TOKEN env variable)",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Skip local testing (default: False)",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Initial commit",
        help="Commit message for HuggingFace upload (default: 'Initial commit')",
    )
    
    return parser.parse_args()


def main(args): 
    # Determine if we should test locally
    test_locally = not args.skip_test
    
    # Get HF token from argument or environment
    hf_token = args.hf_token or os.getenv("HF_TOKEN")
    
    if args.release and not hf_token:
        print("ERROR: --release flag requires HF_TOKEN env variable or --hf-token argument")
        sys.exit(1)
    
    print("=" * 60)
    print("SPECTRE Model Export to HuggingFace")
    print("=" * 60)
    print(f"Export directory: {EXPORT_DIR}")
    print(f"Repo ID: {args.repo_id}")
    print(f"Test locally: {test_locally}")
    print(f"Release (upload): {args.release}")
    print("=" * 60)
    
    # Clean export
    if DEST_PACKAGE.exists():
        print("Cleaning existing export directory...")
        shutil.rmtree(DEST_PACKAGE)

    # Copy selected parts of the spectre package
    print("Copying spectre package...")
    _copy_selective(SRC_PACKAGE, DEST_PACKAGE, INCLUDE)
    print(f"✓ Exported spectre package to {DEST_PACKAGE}")

    print("Verifying exported imports resolve...")
    _verify_export()
    print("✓ All spectre imports in the exported package resolve")

    # Prepend export dir to sys.path
    sys.path.insert(0, str(EXPORT_DIR))

    from spectre import SpectreImageFeatureExtractor
    from configuration_spectre import SpectreConfig
    from modeling_spectre import SpectreModel

    SpectreConfig.register_for_auto_class()
    SpectreModel.register_for_auto_class("AutoModel")

    print("Loading base model weights...")
    base = SpectreImageFeatureExtractor.from_pretrained(args.preset, verbose=True)

    print("Building model configuration...")
    # Nothing about the architecture is spelled out here: SpectreConfig resolves it from
    # spectre.presets, which is the only place those values live.
    config = SpectreConfig(preset=args.preset, crop_size=base.crop_size)

    print("Creating HuggingFace model...")
    hf_model = SpectreModel(config)

    # strict=True is the desync guard: if the config resolved a different architecture than the
    # preset built, the key sets differ and this raises instead of shipping a broken model.
    hf_model.model.load_state_dict(base.state_dict(), strict=True)

    print("Saving model and config...")
    # save_pretrained copies the modeling file into the save directory; saving
    # directly to EXPORT_DIR (where the file already lives) raises SameFileError,
    # so we save to a temp dir and promote the outputs afterward.
    import tempfile
    with tempfile.TemporaryDirectory() as _tmp:
        _tmp_path = Path(_tmp)
        hf_model.save_pretrained(_tmp_path, safe_serialization=True)
        config.save_pretrained(_tmp_path)
        for f in _tmp_path.iterdir():
            shutil.copy2(f, EXPORT_DIR / f.name)
    print(f"✓ Saved model and config to {EXPORT_DIR}")

    # Test loading the model locally
    if test_locally:
        print("\nTesting local model loading...")
        from transformers import AutoModel

        try:
            model = AutoModel.from_pretrained(EXPORT_DIR, trust_remote_code=True)
            print("✓ Model loaded successfully")
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            sys.exit(1)

    # Upload to HuggingFace Hub
    if args.release:
        # Sync README from the repo root, prepending HF model card metadata as YAML frontmatter
        metadata = (EXPORT_DIR / "metadata.yaml").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        (EXPORT_DIR / "README.md").write_text(f"---\n{metadata}\n---\n\n{readme}", encoding="utf-8")
        shutil.copytree(ROOT / "imgs", EXPORT_DIR / "imgs", dirs_exist_ok=True)

        shutil.copy2(ROOT / "LICENSE", EXPORT_DIR / "LICENSE")
        shutil.copy2(ROOT / "LICENSE_MODELS", EXPORT_DIR / "LICENSE_MODELS")
        print("✓ Copied README and LICENSE files")


        # Strip __pycache__ directories so they are not uploaded
        for pycache in EXPORT_DIR.rglob("__pycache__"):
            shutil.rmtree(pycache)
        print("✓ Removed __pycache__ directories")

        print("\nUploading to HuggingFace Hub...")
        from huggingface_hub import HfApi

        try:
            api = HfApi(token=hf_token)
            api.create_repo(
                repo_id=args.repo_id,
                repo_type="model",
                exist_ok=True,
            )
            api.upload_folder(
                folder_path=str(EXPORT_DIR),
                path_in_repo=".",
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=args.commit_message,
                delete_patterns=["*"],
            )
            print(f"✓ Successfully uploaded to {args.repo_id}")
        except Exception as e:
            print(f"✗ Error uploading to HuggingFace: {e}")
            sys.exit(1)
    else:
        print("\nSkipping upload (--release not specified)")
        print(f"To upload, run: python export_hf.py --release --repo-id {args.repo_id}")


if __name__ == "__main__":
    args = get_args()
    main(args)
