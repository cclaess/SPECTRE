"""Measure how much SPECTRE embeddings move when preprocessing changes.

The two preprocessing knobs SPECTRE exposes at inference time are the voxel spacing (whether, and
to what, a scan is resampled) and the HU window that intensities are clipped and scaled into.
Neither has an obviously-correct value - the pretraining stages themselves disagreed on spacing -
so this script quantifies how sensitive the embeddings are to each.

For every scan it computes a reference embedding at a baseline configuration, then re-embeds the
same scan while varying one knob at a time, and reports how far the embedding moved:

* cosine distance  1 - cos(e, e_ref)      - direction change, in [0, 2]
* relative L2       ||e - e_ref|| / ||e_ref||  - magnitude change, scale-free

"how heavy" is the size of those numbers; "how" is their pattern across the sweep and the
difference between the spacing panel and the HU panel.

Example:
    python eval/preprocessing_sensitivity.py --scans /data/scans --device cuda --plot

To override the sweeps, pass semicolon-separated lists. HU windows start with a minus sign, so
use the =form so argparse does not mistake them for flags:
    ... --spacings="native;1.0,1.0,1.0" --hu_windows="-1000,1000;-500,500"

Needs the inference extra (nibabel, monai):  pip install "spectre-fm[inference]"
"""
import argparse
import csv
import warnings
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from spectre.io import find_ct_files, load_and_window
from spectre.model import SpectreImageFeatureExtractor
from spectre.windowing import DEFAULT_CROP_SIZE

# One knob varied at a time; the other stays at the reference. Semicolon-separated lists (rather
# than argparse nargs) because HU windows begin with a minus sign, which argparse reads as a flag.
# The reference value is always included so its zero distance anchors the scale.
DEFAULT_SPACINGS = "native;0.5,0.5,1.0;0.75,0.75,1.5;1.0,1.0,1.0;1.5,1.5,1.5"
DEFAULT_HU_WINDOWS = "-1000,1000;-1350,150;-160,240;-1000,2000;-500,500"


def get_args_parser():
    parser = argparse.ArgumentParser(
        description="Measure SPECTRE embedding sensitivity to voxel spacing and HU windowing.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--scans", type=str, nargs="+", required=True,
        help="One or more .nii/.nii.gz files or folders containing them.",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./preprocessing_sensitivity",
        help="Where to write per-scan and summary CSVs (and the plot).",
    )
    parser.add_argument("--model", type=str, default="spectre-large", help="Which SPECTRE model.")
    parser.add_argument("--device", type=str, default="auto", help="'auto', 'cpu', 'cuda', ...")
    parser.add_argument(
        "--crop_size", type=int, nargs=3, default=list(DEFAULT_CROP_SIZE), metavar=("H", "W", "D"),
    )
    parser.add_argument(
        "--max_crops_per_forward", type=int, default=16,
        help="Cap crops per backbone pass to bound memory (0 = no cap).",
    )
    parser.add_argument(
        "--embedding", choices=("cls", "mean_patch"), default="cls",
        help="Scan embedding to compare: the combiner CLS token, or the mean of its patch tokens.",
    )
    parser.add_argument(
        "--reference_spacing", type=str, default="native",
        help="Baseline spacing held fixed while HU is swept ('native' or 'sx,sy,sz').",
    )
    parser.add_argument(
        "--reference_hu", type=str, default="-1000,1000",
        help="Baseline HU window held fixed while spacing is swept ('a_min,a_max').",
    )
    parser.add_argument(
        "--spacings", type=str, default=DEFAULT_SPACINGS,
        help="Semicolon-separated spacings to test (each 'native' or 'sx,sy,sz').",
    )
    parser.add_argument(
        "--hu_windows", type=str, default=DEFAULT_HU_WINDOWS,
        help="Semicolon-separated HU windows to test (each 'a_min,a_max'). Use --hu_windows=...",
    )
    parser.add_argument("--plot", action="store_true", help="Also write sensitivity.png (needs matplotlib).")
    return parser


def parse_spacing(text):
    """'native' -> None; 'sx,sy,sz' -> (sx, sy, sz)."""
    if text.strip().lower() == "native":
        return None
    parts = [float(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Spacing must be 'native' or three comma-separated numbers, got {text!r}.")
    return tuple(parts)


def parse_hu(text):
    """'a_min,a_max' -> (a_min, a_max)."""
    parts = [float(v) for v in text.replace(" ", "").split(",")]
    if len(parts) != 2:
        raise ValueError(f"HU window must be 'a_min,a_max', got {text!r}.")
    return tuple(parts)


def resolve_device(requested):
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def embed(model, path, *, crop_size, spacing, hu_range, which, max_crops):
    """Preprocess `path` with the given knobs and return one scan-level embedding vector."""
    with warnings.catch_warnings():
        # A scan smaller than one crop is padded (and warns); that is orthogonal to this analysis.
        warnings.simplefilter("ignore", RuntimeWarning)
        crops, grid, _ = load_and_window(
            path, crop_size=crop_size, spacing=spacing, hu_range=hu_range,
        )
    with torch.inference_mode():
        tokens = model(crops, grid_size=grid, max_crops_per_forward=max_crops)  # (T', F')
    return tokens[0] if which == "cls" else tokens[1:].mean(dim=0)


def compare(embedding, reference):
    """(cosine distance, relative L2) between an embedding and the reference."""
    embedding = embedding.flatten().float()
    reference = reference.flatten().float()
    # max(0, .) drops the ~1e-7 float overshoot when comparing an embedding with itself.
    cosine_distance = max(0.0, 1.0 - F.cosine_similarity(embedding, reference, dim=0).item())
    relative_l2 = (torch.norm(embedding - reference) / torch.norm(reference)).item()
    return cosine_distance, relative_l2


def sweep_configs(values, reference_value, parse):
    """Ordered, de-duplicated (label, parsed) configs, with the reference first.

    `values` is a semicolon-separated string; blank entries are ignored.
    """
    requested = [v.strip() for v in values.split(";") if v.strip()]
    labels = [reference_value] + [v for v in requested if v != reference_value]
    seen, configs = set(), []
    for label in labels:
        parsed = parse(label)
        key = str(parsed)
        if key not in seen:
            seen.add(key)
            configs.append((label, parsed))
    return configs


def main(args):
    device = resolve_device(args.device)
    crop_size = tuple(args.crop_size)
    max_crops = args.max_crops_per_forward or None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_spacing = parse_spacing(args.reference_spacing)
    reference_hu = parse_hu(args.reference_hu)
    spacing_configs = sweep_configs(args.spacings, args.reference_spacing, parse_spacing)
    hu_configs = sweep_configs(args.hu_windows, args.reference_hu, parse_hu)

    scans = []
    for entry in args.scans:
        scans.extend(find_ct_files(entry))
    scans = sorted(set(scans))
    if not scans:
        raise SystemExit(f"error: no .nii/.nii.gz files found in {args.scans}")

    print(f"Loading {args.model} on {device} ...")
    model = SpectreImageFeatureExtractor.from_pretrained(args.model, device=device)
    if crop_size != model.crop_size:
        raise SystemExit(
            f"error: --crop_size {crop_size} does not match what {args.model} expects "
            f"({model.crop_size})."
        )

    print(f"{len(scans)} scan(s); reference = spacing {args.reference_spacing}, HU {reference_hu}\n")

    # rows: (axis, config_label, scan_name, cosine_distance, relative_l2)
    rows = []
    failures = []
    for i, scan in enumerate(scans, 1):
        name = scan.name
        try:
            reference = embed(
                model, scan, crop_size=crop_size, spacing=reference_spacing,
                hu_range=reference_hu, which=args.embedding, max_crops=max_crops,
            )
        except Exception as e:  # one bad scan should not kill the run
            print(f"[{i}/{len(scans)}] {name}: FAILED ({e})")
            failures.append((name, str(e)))
            continue

        for axis, configs, fixed in (
            ("spacing", spacing_configs, dict(hu_range=reference_hu)),
            ("hu", hu_configs, dict(spacing=reference_spacing)),
        ):
            for label, parsed in configs:
                kw = dict(spacing=parsed) if axis == "spacing" else dict(hu_range=parsed)
                emb = embed(
                    model, scan, crop_size=crop_size, which=args.embedding,
                    max_crops=max_crops, **{**fixed, **kw},
                )
                cos_d, rel_l2 = compare(emb, reference)
                rows.append((axis, label, name, cos_d, rel_l2))
        print(f"[{i}/{len(scans)}] {name}: done")

    if not rows:
        raise SystemExit("error: no scans could be embedded.")

    _write_per_scan(output_dir / "per_scan.csv", rows)
    summary = _summarise(rows)
    _write_summary(output_dir / "summary.csv", summary)
    _print_summary(summary, args)

    if args.plot:
        _plot(summary, output_dir / "sensitivity.png", args)

    if failures:
        print(f"\n{len(failures)} scan(s) failed:")
        for name, err in failures:
            print(f"  {name}: {err}")

    print(f"\nWrote results to {output_dir}")


def _summarise(rows):
    """Aggregate per-scan rows into per-config statistics, preserving config order."""
    grouped = defaultdict(lambda: {"cos": [], "l2": []})
    order = []
    for axis, label, _, cos_d, rel_l2 in rows:
        key = (axis, label)
        if key not in grouped:
            order.append(key)
        grouped[key]["cos"].append(cos_d)
        grouped[key]["l2"].append(rel_l2)

    summary = []
    for axis, label in order:
        cos = torch.tensor(grouped[(axis, label)]["cos"])
        l2 = torch.tensor(grouped[(axis, label)]["l2"])
        summary.append({
            "axis": axis,
            "config": label,
            "n": cos.numel(),
            "cos_mean": cos.mean().item(),
            "cos_std": cos.std(unbiased=False).item(),
            "cos_max": cos.max().item(),
            "l2_mean": l2.mean().item(),
            "l2_std": l2.std(unbiased=False).item(),
            "l2_max": l2.max().item(),
        })
    return summary


def _write_per_scan(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["axis", "config", "scan", "cosine_distance", "relative_l2"])
        writer.writerows(rows)


def _write_summary(path, summary):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)


def _print_summary(summary, args):
    for axis, title, ref in (
        ("spacing", "SPACING sensitivity (HU fixed)", args.reference_spacing),
        ("hu", "HU-WINDOW sensitivity (spacing fixed)", args.reference_hu),
    ):
        entries = [s for s in summary if s["axis"] == axis]
        print(f"\n{title}   [reference: {ref}]")
        print(f"  {'config':<16}{'cos dist (mean±std)':<26}{'rel L2 (mean±std)':<24}{'cos max':>8}")
        for s in entries:
            cos = f"{s['cos_mean']:.4f} ± {s['cos_std']:.4f}"
            l2 = f"{s['l2_mean']:.4f} ± {s['l2_std']:.4f}"
            tag = "  (ref)" if s["cos_mean"] == 0.0 and s["cos_std"] == 0.0 else ""
            print(f"  {s['config']:<16}{cos:<26}{l2:<24}{s['cos_max']:>8.4f}{tag}")

    # Report the loudest non-reference config on each axis: this is the "how" - which knob moves
    # the embedding most, and by how much.
    def loudest(axis):
        cands = [s for s in summary if s["axis"] == axis and not (s["cos_mean"] == 0.0 and s["cos_std"] == 0.0)]
        return max(cands, key=lambda s: s["cos_mean"]) if cands else None

    sp, hu = loudest("spacing"), loudest("hu")
    print("\nLargest movement from the reference:")
    if sp:
        print(f"  spacing -> {sp['config']:<14} cosine dist {sp['cos_mean']:.4f}, rel L2 {sp['l2_mean']:.4f}")
    if hu:
        print(f"  HU      -> {hu['config']:<14} cosine dist {hu['cos_mean']:.4f}, rel L2 {hu['l2_mean']:.4f}")


def _plot(summary, path, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("warning: matplotlib not installed; skipping --plot")
        return

    # Magnitude-of-change across categorical configs -> bars with error bars. One measure per
    # panel (never a second y-axis). A single colorblind-safe hue; the zero-distance reference bar
    # is greyed so the baseline reads as "no change", not "smallest change".
    bar, ref_bar, err = "#4C78A8", "#BAB0AC", "#333333"
    panels = [
        ("spacing", "cos_mean", "cos_std", "cosine distance", f"Spacing (HU {args.reference_hu})"),
        ("hu", "cos_mean", "cos_std", "cosine distance", f"HU window (spacing {args.reference_spacing})"),
        ("spacing", "l2_mean", "l2_std", "relative L2", None),
        ("hu", "l2_mean", "l2_std", "relative L2", None),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (axis, mean_k, std_k, ylabel, title) in zip(axes.flat, panels):
        entries = [s for s in summary if s["axis"] == axis]
        labels = [s["config"] for s in entries]
        means = [s[mean_k] for s in entries]
        stds = [s[std_k] for s in entries]
        colors = [ref_bar if (s["cos_mean"] == 0.0 and s["cos_std"] == 0.0) else bar for s in entries]
        x = range(len(entries))
        ax.bar(x, means, yerr=stds, color=colors, ecolor=err, capsize=3, width=0.7, zorder=3)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title, fontsize=10)
        ax.grid(axis="y", color="0.9", zorder=0)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    n = summary[0]["n"] if summary else 0
    fig.suptitle(
        f"SPECTRE embedding sensitivity to preprocessing ({args.embedding}, n={n} scans)\n"
        f"grey bar = reference (0 by construction); taller = embedding moved more",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Wrote plot to {path}")


if __name__ == "__main__":
    main(get_args_parser().parse_args())
