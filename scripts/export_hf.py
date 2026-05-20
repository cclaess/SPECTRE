import os
import re
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
INCLUDE = {
    "__init__.py": True,
    "model.py": True,
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


def _patch_init(init_path, keep_modules):
    """Strip import lines for modules not in keep_modules and remove their
    exported names from __all__, handling multi-line imports throughout."""
    lines = init_path.read_text(encoding="utf-8").splitlines(keepends=True)

    # --- Pass 1: collect names exported by excluded imports ---
    _KEYWORDS = {"as", "True", "False", "None"}
    excluded_names = set()
    i = 0
    while i < len(lines):
        line = lines[i]
        m_direct = re.match(r"^from \. import ([\w]+)\b", line)
        m_from   = re.match(r"^from \.([\w]+)\b", line)
        module = (m_direct or m_from)
        if module and module.group(1) not in keep_modules:
            if m_direct:
                excluded_names.add(m_direct.group(1))
            else:
                after = line.partition("import")[2]
                excluded_names.update(n for n in re.findall(r"\b([A-Za-z_]\w*)\b", after) if n not in _KEYWORDS)
                depth = line.count("(") - line.count(")")
                while depth > 0 and i + 1 < len(lines):
                    i += 1
                    excluded_names.update(n for n in re.findall(r"\b([A-Za-z_]\w*)\b", lines[i]) if n not in _KEYWORDS)
                    depth += lines[i].count("(") - lines[i].count(")")
        i += 1

    # --- Pass 2: remove excluded import lines ---
    result = []
    skip = False
    paren_depth = 0
    for line in lines:
        if not skip:
            m = re.match(r"^from \.([\w]+)\b", line) or re.match(r"^from \. import ([\w]+)\b", line)
            if m and m.group(1) not in keep_modules:
                skip = True
                paren_depth = line.count("(") - line.count(")")
                if paren_depth <= 0:
                    skip = False
                continue
            result.append(line)
        else:
            paren_depth += line.count("(") - line.count(")")
            if paren_depth <= 0:
                skip = False

    # --- Pass 3: remove excluded names from __all__ ---
    if excluded_names:
        patched = []
        in_all = False
        bracket_depth = 0
        for line in result:
            if not in_all:
                if re.match(r"^\s*__all__\s*=\s*\[", line):
                    in_all = True
                    bracket_depth = line.count("[") - line.count("]")
                    if bracket_depth <= 0:
                        in_all = False
                patched.append(line)
            else:
                m = re.search(r'["\']([A-Za-z_]\w*)["\']', line)
                if m and m.group(1) in excluded_names:
                    pass  # drop this __all__ entry
                else:
                    patched.append(line)
                bracket_depth += line.count("[") - line.count("]")
                if bracket_depth <= 0:
                    in_all = False
        result = patched

    init_path.write_text("".join(result), encoding="utf-8")


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

    # Prepend export dir to sys.path
    sys.path.insert(0, str(EXPORT_DIR))

    from spectre import SpectreImageFeatureExtractor, MODEL_CONFIGS
    from configuration_spectre import SpectreConfig
    from modeling_spectre import SpectreModel

    SpectreConfig.register_for_auto_class()
    SpectreModel.register_for_auto_class("AutoModel")

    print("Building model configuration...")
    config = SpectreConfig(
        backbone_name="vit_large_patch16_128",
        backbone_kwargs={
            "num_classes": 0,
            "global_pool": "",
            "pos_embed": "rope",
            "rope_kwargs": {"base": 1000.0},
            "init_values": 1.0,
        },
        feature_combiner_name="feat_vit_large",
        feature_combiner_kwargs={
            "num_classes": 0,
            "global_pool": "",
            "pos_embed": "rope",
            "rope_kwargs": {"base": 100.0},
            "init_values": 1.0,
        },
    )

    print("Loading base model weights...")
    base = SpectreImageFeatureExtractor.from_config(MODEL_CONFIGS["spectre-large-pretrained"])

    print("Creating HuggingFace model...")
    hf_model = SpectreModel(config)

    hf_model.model.load_state_dict(base.state_dict())

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
