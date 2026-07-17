"""CLI behaviour, exercised against a small randomly-initialised model.

The published spectre-large weights are ~1 GB, so these tests patch `from_pretrained` to hand
back a tiny model. They cover the plumbing (file discovery, saving, skipping, exit codes), not
embedding quality.
"""
import csv

import numpy as np
import pytest
import torch

pytest.importorskip("nibabel", reason="the CLI needs the [inference] extra")
import nibabel as nib  # noqa: E402

from spectre.cli import build_parser, main  # noqa: E402
from spectre.model import SpectreImageFeatureExtractor  # noqa: E402


@pytest.fixture
def small_model(monkeypatch):
    """Serve a tiny random model wherever the CLI asks for a pretrained one."""
    built = {}

    def fake_from_pretrained(name="spectre-large", *, include_feature_combiner=True, device=None, **kwargs):
        key = (name, include_feature_combiner)
        if key not in built:
            model = SpectreImageFeatureExtractor(
                backbone_name="vit_small_patch16_128",
                backbone_kwargs={"num_classes": 0, "global_pool": "", "pos_embed": "rope",
                                 "rope_kwargs": {"base": 1000.0}, "init_values": 1.0},
                feature_combiner_name="feat_vit_small" if include_feature_combiner else None,
                feature_combiner_kwargs={"num_classes": 0, "global_pool": "", "pos_embed": "rope",
                                         "rope_kwargs": {"base": 100.0}, "init_values": 1.0},
            )
            built[key] = model.eval()
        return built[key]

    monkeypatch.setattr(SpectreImageFeatureExtractor, "from_pretrained", fake_from_pretrained)
    return fake_from_pretrained


def write_scan(path, shape=(256, 128, 64), seed=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    data = (rng.random(shape) * 3000 - 1200).astype(np.float32)
    nib.save(nib.Nifti1Image(data, np.diag([0.7, 0.7, 1.5, 1.0])), str(path))
    return path


def test_list_models_runs_without_torch_model(capsys):
    assert main(["list-models"]) == 0
    out = capsys.readouterr().out
    assert "spectre-large" in out and "available" in out
    assert "spectre-small" in out


def test_no_command_prints_help():
    assert main([]) == 1


def test_version_flag(capsys):
    assert main(["--version"]) == 0
    assert "spectre-fm" in capsys.readouterr().out


def test_embed_single_file(tmp_path, small_model):
    scan = write_scan(tmp_path / "scan.nii.gz")
    out = tmp_path / "out"
    assert main(["embed", str(scan), "-o", str(out), "--device", "cpu", "-q"]) == 0

    saved = out / "scan.npz"
    assert saved.exists()
    with np.load(saved, allow_pickle=False) as f:
        assert f["cls"].ndim == 1
        assert f["patch_tokens"].shape[:3] == (2, 1, 1)  # grid for 256x128x64
        assert list(f["grid_size"]) == [2, 1, 1]
        assert np.isfinite(f["cls"]).all()
        assert np.abs(f["cls"]).sum() > 0  # not a degenerate all-zero embedding


def test_embed_directory_of_unequal_scans(tmp_path, small_model):
    scans = tmp_path / "scans"
    scans.mkdir()
    write_scan(scans / "a.nii.gz", (300, 260, 130), seed=1)
    write_scan(scans / "b.nii.gz", (256, 256, 128), seed=2)
    write_scan(scans / "c.nii.gz", (128, 128, 64), seed=3)
    out = tmp_path / "out"

    assert main(["embed", str(scans), "-o", str(out), "--device", "cpu", "--batch-size", "3", "-q"]) == 0
    assert {p.name for p in out.glob("*.npz")} == {"a.npz", "b.npz", "c.npz"}
    assert (out / "manifest.csv").exists()

    # Differently-sized scans must produce differently-shaped patch grids.
    with np.load(out / "a.npz") as fa, np.load(out / "c.npz") as fc:
        assert list(fa["grid_size"]) == [2, 2, 2]
        assert list(fc["grid_size"]) == [1, 1, 1]


def test_nested_scans_mirror_the_input_structure(tmp_path, small_model):
    """Same-named scans in different subfolders must not overwrite each other.

    Both are planned before either is written, so the "already exists" skip does not catch this;
    without mirroring, the second silently clobbered the first.
    """
    scans = tmp_path / "scans"
    (scans / "patient_a" / "study_1").mkdir(parents=True)
    (scans / "patient_b" / "study_1").mkdir(parents=True)
    write_scan(scans / "patient_a" / "study_1" / "scan.nii.gz", seed=1)
    write_scan(scans / "patient_b" / "study_1" / "scan.nii.gz", seed=2)
    write_scan(scans / "top_level.nii.gz", seed=3)
    out = tmp_path / "out"

    assert main(["embed", str(scans), "-o", str(out), "--device", "cpu", "-q"]) == 0

    assert (out / "patient_a" / "study_1" / "scan.npz").exists()
    assert (out / "patient_b" / "study_1" / "scan.npz").exists()
    assert (out / "top_level.npz").exists()

    # Three inputs must yield three distinct outputs, not two.
    assert len(list(out.rglob("*.npz"))) == 3

    # And the two same-named scans must hold their own embeddings.
    with np.load(out / "patient_a" / "study_1" / "scan.npz") as a, \
         np.load(out / "patient_b" / "study_1" / "scan.npz") as b:
        assert not np.allclose(a["cls"], b["cls"]), "one scan overwrote the other"


def test_nested_source_is_recorded_in_the_manifest(tmp_path, small_model):
    scans = tmp_path / "scans"
    (scans / "sub").mkdir(parents=True)
    write_scan(scans / "sub" / "scan.nii.gz")
    out = tmp_path / "out"

    main(["embed", str(scans), "-o", str(out), "--device", "cpu", "-q"])

    rows = list(csv.DictReader((out / "manifest.csv").open()))
    assert len(rows) == 1
    assert rows[0]["relative_output"] in (r"sub\scan.npz", "sub/scan.npz")
    assert "sub" in rows[0]["source"]


def test_single_file_input_is_not_nested(tmp_path, small_model):
    """Pointing at one file should put its output straight in the output folder."""
    scan = write_scan(tmp_path / "deep" / "nested" / "scan.nii.gz")
    (tmp_path / "out").mkdir()
    assert main(["embed", str(scan), "-o", str(tmp_path / "out"), "--device", "cpu", "-q"]) == 0
    assert (tmp_path / "out" / "scan.npz").exists()
    assert not (tmp_path / "out" / "deep").exists()


def test_nested_skip_and_overwrite_still_work(tmp_path, small_model, capsys):
    scans = tmp_path / "scans"
    (scans / "sub").mkdir(parents=True)
    write_scan(scans / "sub" / "scan.nii.gz")
    out = tmp_path / "out"

    main(["embed", str(scans), "-o", str(out), "--device", "cpu", "-q"])
    mtime = (out / "sub" / "scan.npz").stat().st_mtime_ns

    main(["embed", str(scans), "-o", str(out), "--device", "cpu"])
    captured = capsys.readouterr().out
    assert "skipping" in captured
    assert "sub" in captured, "the skip message should name the scan's subfolder"
    assert (out / "sub" / "scan.npz").stat().st_mtime_ns == mtime


def test_colliding_outputs_are_refused(tmp_path, small_model, capsys):
    """Mirroring cannot separate a.nii and a.nii.gz side by side; fail rather than clobber."""
    scans = tmp_path / "scans"
    scans.mkdir()
    write_scan(scans / "scan.nii.gz")
    write_scan(scans / "scan.nii")

    code = main(["embed", str(scans), "-o", str(tmp_path / "out"), "--device", "cpu", "-q"])
    assert code == 1
    err = capsys.readouterr().err
    assert "would both be written to" in err


def test_no_recursive_ignores_subfolders(tmp_path, small_model):
    scans = tmp_path / "scans"
    (scans / "sub").mkdir(parents=True)
    write_scan(scans / "top.nii.gz")
    write_scan(scans / "sub" / "deep.nii.gz")
    out = tmp_path / "out"

    assert main(["embed", str(scans), "-o", str(out), "--no-recursive", "--device", "cpu", "-q"]) == 0
    assert (out / "top.npz").exists()
    assert not (out / "sub" / "deep.npz").exists()


def test_embed_backbone_only(tmp_path, small_model):
    scan = write_scan(tmp_path / "scan.nii.gz")
    out = tmp_path / "out"
    assert main(["embed", str(scan), "-o", str(out), "--backbone-only", "--device", "cpu", "-q"]) == 0
    with np.load(out / "scan.npz") as f:
        assert "crop_tokens" in f and "cls" not in f
        assert f["crop_tokens"].shape[0] == 2  # one entry per crop


def test_embed_skips_existing_and_overwrite_redoes(tmp_path, small_model, capsys):
    scan = write_scan(tmp_path / "scan.nii.gz")
    out = tmp_path / "out"
    main(["embed", str(scan), "-o", str(out), "--device", "cpu", "-q"])
    mtime = (out / "scan.npz").stat().st_mtime_ns

    main(["embed", str(scan), "-o", str(out), "--device", "cpu"])
    assert "skipping" in capsys.readouterr().out
    assert (out / "scan.npz").stat().st_mtime_ns == mtime

    main(["embed", str(scan), "-o", str(out), "--device", "cpu", "--overwrite", "-q"])
    assert (out / "scan.npz").stat().st_mtime_ns != mtime


def test_embed_pt_format(tmp_path, small_model):
    scan = write_scan(tmp_path / "scan.nii.gz")
    out = tmp_path / "out"
    assert main(["embed", str(scan), "-o", str(out), "--format", "pt", "--device", "cpu", "-q"]) == 0
    payload = torch.load(out / "scan.pt", weights_only=False)
    assert "cls" in payload and payload["model"] == "spectre-large"


def test_missing_input_is_a_usage_error(tmp_path, small_model, capsys):
    assert main(["embed", str(tmp_path / "nope.nii.gz"), "-o", str(tmp_path / "o")]) == 1
    assert "error" in capsys.readouterr().err


def test_empty_directory_is_a_usage_error(tmp_path, small_model, capsys):
    empty = tmp_path / "empty"
    empty.mkdir()
    assert main(["embed", str(empty), "-o", str(tmp_path / "o")]) == 1
    assert "no .nii" in capsys.readouterr().err


def test_unknown_model_is_a_usage_error(tmp_path, capsys):
    scan = write_scan(tmp_path / "scan.nii.gz")
    assert main(["embed", str(scan), "-o", str(tmp_path / "o"), "-m", "spectre-huge"]) == 1
    assert "Unknown SPECTRE preset" in capsys.readouterr().err


def test_wrong_crop_size_is_rejected(tmp_path, small_model, capsys):
    scan = write_scan(tmp_path / "scan.nii.gz")
    code = main([
        "embed", str(scan), "-o", str(tmp_path / "o"), "--crop-size", "64", "64", "32", "--device", "cpu",
    ])
    assert code == 1
    assert "does not match" in capsys.readouterr().err


def test_one_bad_scan_does_not_kill_the_run(tmp_path, small_model, capsys):
    scans = tmp_path / "scans"
    scans.mkdir()
    write_scan(scans / "good.nii.gz")
    (scans / "broken.nii.gz").write_bytes(b"this is not a nifti file")

    code = main(["embed", str(scans), "-o", str(tmp_path / "out"), "--device", "cpu", "-q"])
    assert code == 2  # partial failure
    assert (tmp_path / "out" / "good.npz").exists()
    assert "broken.nii.gz" in capsys.readouterr().err


def test_short_scan_is_padded_and_reported(tmp_path, small_model, capsys):
    scan = write_scan(tmp_path / "thin.nii.gz", shape=(128, 128, 40))
    out = tmp_path / "out"
    assert main(["embed", str(scan), "-o", str(out), "--device", "cpu"]) == 0
    assert "padded with air" in capsys.readouterr().out
    assert (out / "thin.npz").exists()


def test_short_scan_can_be_refused(tmp_path, small_model, capsys):
    scan = write_scan(tmp_path / "thin.nii.gz", shape=(128, 128, 40))
    code = main([
        "embed", str(scan), "-o", str(tmp_path / "out"), "--no-pad-short-axes", "--device", "cpu", "-q",
    ])
    assert code == 2
    assert "smaller than one crop" in capsys.readouterr().err


def test_max_crops_per_forward_does_not_change_output(tmp_path, small_model):
    scan = write_scan(tmp_path / "scan.nii.gz", (256, 256, 128))
    a, b = tmp_path / "a", tmp_path / "b"
    main(["embed", str(scan), "-o", str(a), "--max-crops-per-forward", "0", "--device", "cpu", "-q"])
    main(["embed", str(scan), "-o", str(b), "--max-crops-per-forward", "2", "--device", "cpu", "-q"])
    with np.load(a / "scan.npz") as fa, np.load(b / "scan.npz") as fb:
        assert np.allclose(fa["cls"], fb["cls"], atol=1e-5)


def test_parser_help_mentions_the_simple_case():
    text = build_parser().format_help()
    assert "spectre embed scan.nii.gz" in text
