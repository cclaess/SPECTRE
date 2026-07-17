"""Command-line interface for SPECTRE.

    spectre embed scan.nii.gz -o embeddings/
    spectre embed /data/scans/ -o embeddings/ --backbone-only
    spectre list-models

Needs the optional inference dependencies:

    pip install "spectre-fm[inference]"

Heavy imports (torch, the model) live inside the command functions so that `spectre --help` and
`spectre list-models` stay instant.
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

DEFAULT_CROP_SIZE = (128, 128, 64)
DEFAULT_HU_RANGE = (-1000.0, 1000.0)

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_PARTIAL_FAILURE = 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="spectre",
        description=(
            "Extract SPECTRE embeddings from CT scans. Point it at a .nii/.nii.gz file or a "
            "folder of them; preprocessing (HU windowing, cropping, tiling) is handled for you."
        ),
        epilog="Example:  spectre embed scan.nii.gz -o embeddings/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--version", action="store_true", help="print the spectre-fm version and exit")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    embed = sub.add_parser(
        "embed",
        help="embed a CT scan, or every scan in a folder",
        description=(
            "Embed a CT scan into a feature vector. Defaults match how SPECTRE was pretrained: "
            "HU window [-1000, 1000], RAS orientation, 128x128x64 crops, native voxel spacing."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    embed.add_argument("input", type=Path, help="a .nii/.nii.gz file, or a folder containing them")
    embed.add_argument(
        "-o", "--output", type=Path, default=Path("./spectre_embeddings"),
        help="folder to write embeddings to",
    )
    embed.add_argument("-m", "--model", default="spectre-large", help="which SPECTRE model to use")
    embed.add_argument(
        "--backbone-only", action="store_true",
        help="skip the feature combiner and save per-crop backbone features instead",
    )
    embed.add_argument(
        "--device", default="auto",
        help="'auto', 'cpu', 'cuda', or e.g. 'cuda:0'",
    )
    embed.add_argument(
        "--crop-size", type=int, nargs=3, default=list(DEFAULT_CROP_SIZE), metavar=("H", "W", "D"),
        help="size of one CT crop; must match what the model was trained on",
    )
    embed.add_argument(
        "--spacing", type=float, nargs=3, default=None, metavar=("SX", "SY", "SZ"),
        help="resample to this voxel spacing in mm, e.g. --spacing 0.5 0.5 1.0 "
             "(default: leave the scan at its native spacing)",
    )
    embed.add_argument(
        "--hu-range", type=float, nargs=2, default=list(DEFAULT_HU_RANGE), metavar=("MIN", "MAX"),
        help="Hounsfield Unit window mapped onto [0, 1]",
    )
    embed.add_argument(
        "--batch-size", type=int, default=1, help="scans to embed per forward pass",
    )
    embed.add_argument(
        "--max-crops-per-forward", type=int, default=16,
        help="cap on crops in one backbone pass; lower this if you run out of memory (0 = no cap)",
    )
    embed.add_argument("--format", choices=("npz", "pt"), default="npz", help="output file format")
    embed.add_argument(
        "--no-recursive", action="store_true", help="don't search subfolders of a folder input",
    )
    embed.add_argument(
        "--overwrite", action="store_true", help="re-embed scans that already have an output file",
    )
    embed.add_argument(
        "--no-pad-short-axes", action="store_true",
        help="fail on scans smaller than one crop instead of padding them with air",
    )
    embed.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    embed.set_defaults(func=_cmd_embed)

    listing = sub.add_parser(
        "list-models", help="show the available SPECTRE models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    listing.set_defaults(func=_cmd_list_models)

    return parser


def _version() -> str:
    from importlib.metadata import PackageNotFoundError, version
    try:
        return version("spectre-fm")
    except PackageNotFoundError:
        from spectre import __version__
        return __version__


def _cmd_list_models(args: argparse.Namespace) -> int:
    from spectre.presets import PRESETS

    width = max(len(name) for name in PRESETS)
    print(f"{'MODEL'.ljust(width)}  WEIGHTS      DESCRIPTION")
    for name, preset in PRESETS.items():
        weights = "available" if preset.has_pretrained_weights else "-"
        print(f"{name.ljust(width)}  {weights.ljust(11)}  {preset.description}")
    print("\nUse with:  spectre embed scan.nii.gz -m <model>")
    return EXIT_OK


def _resolve_device(requested: str):
    import torch

    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _save(path: Path, payload: dict, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "pt":
        import torch
        torch.save(payload, path)
    else:
        import numpy as np
        np.savez_compressed(path, **payload)


def _payload_for(features, grid, source: Path, model_name: str, backbone_only: bool) -> dict:
    """Split the model output into the pieces worth saving."""
    import numpy as np

    features = features.detach().cpu()
    common = {
        "grid_size": np.asarray(grid, dtype=np.int32),
        "model": model_name,
        "source": str(source),
        "spectre_version": _version(),
    }
    if backbone_only:
        # (N, T, F) per-crop tokens; keep the grid so callers can fold N back to (nH, nW, nD).
        return {"crop_tokens": features.numpy(), **common}

    # (T', F'): a CLS token followed by one token per crop.
    return {
        "cls": features[0].numpy(),
        "patch_tokens": features[1:].reshape(*grid, -1).numpy(),
        **common,
    }


def _cmd_embed(args: argparse.Namespace) -> int:
    from spectre.io import find_ct_files, load_and_window
    from spectre.model import SpectreImageFeatureExtractor

    def log(*parts):
        if not args.quiet:
            print(*parts, flush=True)

    try:
        scans = find_ct_files(args.input, recursive=not args.no_recursive)
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    if not scans:
        print(f"error: no .nii or .nii.gz files found in {args.input}", file=sys.stderr)
        return EXIT_USAGE

    args.output.mkdir(parents=True, exist_ok=True)
    suffix = ".pt" if args.format == "pt" else ".npz"

    todo = []
    for scan in scans:
        stem = scan.name[: -len(".nii.gz")] if scan.name.endswith(".nii.gz") else scan.stem
        destination = args.output / f"{stem}{suffix}"
        if destination.exists() and not args.overwrite:
            log(f"skipping {scan.name} (already embedded; pass --overwrite to redo)")
            continue
        todo.append((scan, destination))

    if not todo:
        log("nothing to do: every scan already has an embedding")
        return EXIT_OK

    device = _resolve_device(args.device)
    log(f"loading {args.model} on {device} ...")
    try:
        model = SpectreImageFeatureExtractor.from_pretrained(
            args.model,
            include_feature_combiner=not args.backbone_only,
            device=device,
            verbose=not args.quiet,
        )
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE

    crop_size = tuple(args.crop_size)
    if crop_size != model.crop_size:
        print(
            f"error: --crop-size {crop_size} does not match what {args.model} expects "
            f"({model.crop_size}). Embeddings would be meaningless.",
            file=sys.stderr,
        )
        return EXIT_USAGE

    max_crops = args.max_crops_per_forward or None
    failures: List[tuple] = []
    padded: List[str] = []
    manifest_rows = []
    started = time.time()

    for batch_start in range(0, len(todo), args.batch_size):
        batch = todo[batch_start:batch_start + args.batch_size]

        loaded, loaded_paths = [], []
        for offset, (scan, destination) in enumerate(batch):
            index = batch_start + offset + 1
            try:
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always", RuntimeWarning)
                    crops, grid, _ = load_and_window(
                        scan,
                        crop_size=crop_size,
                        spacing=args.spacing,
                        hu_range=tuple(args.hu_range),
                        pad_short_axes=not args.no_pad_short_axes,
                    )
                if any("smaller than one crop" in str(w.message) for w in caught):
                    padded.append(scan.name)
                loaded.append((crops, grid))
                loaded_paths.append((scan, destination, index))
            except Exception as e:  # one bad scan must not kill a long run
                print(f"error: {scan.name}: {e}", file=sys.stderr)
                failures.append((scan, e))

        if not loaded:
            continue

        try:
            outputs = model.extract(
                [crops for crops, _ in loaded],
                grid_size=[grid for _, grid in loaded],
                max_crops_per_forward=max_crops,
            )
        except Exception as e:
            for scan, _, _ in loaded_paths:
                print(f"error: {scan.name}: {e}", file=sys.stderr)
                failures.append((scan, e))
            continue

        for (scan, destination, index), (crops, grid), features in zip(loaded_paths, loaded, outputs):
            payload = _payload_for(features, grid, scan, args.model, args.backbone_only)
            _save(destination, payload, args.format)
            log(f"[{index}/{len(todo)}] {scan.name} -> {destination.name} ({crops.shape[0]} crops)")
            manifest_rows.append({
                "source": str(scan),
                "output": str(destination),
                "crops": crops.shape[0],
                "grid_size": "x".join(str(g) for g in grid),
            })

    if manifest_rows:
        manifest = args.output / "manifest.csv"
        write_header = not manifest.exists()
        with open(manifest, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
            if write_header:
                writer.writeheader()
            writer.writerows(manifest_rows)

    elapsed = time.time() - started
    log(f"\nembedded {len(manifest_rows)}/{len(todo)} scans in {elapsed:.1f}s -> {args.output}")

    if padded:
        log(
            f"warning: {len(padded)} scan(s) were smaller than one {crop_size} crop and were "
            f"padded with air, so their embeddings are out of distribution: "
            f"{', '.join(padded[:5])}{' ...' if len(padded) > 5 else ''}"
        )
    if failures:
        print(f"\n{len(failures)} scan(s) failed:", file=sys.stderr)
        for scan, error in failures:
            print(f"  {scan.name}: {error}", file=sys.stderr)
        return EXIT_PARTIAL_FAILURE

    return EXIT_OK


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "version", False):
        print(f"spectre-fm {_version()}")
        return EXIT_OK

    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_USAGE

    try:
        return args.func(args)
    except ImportError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    sys.exit(main())
