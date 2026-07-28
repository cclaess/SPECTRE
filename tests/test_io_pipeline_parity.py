"""End-to-end parity between `spectre.io.load_and_window` and the MONAI training/eval pipeline.

`test_windowing_parity.py` already proves the in-memory tail (scale -> crop -> patch) is bit-exact.
This file closes the loop over the part that only exists once you read a file off disk: loading,
orientation to RAS, and voxel resampling. The reference is the exact deterministic pipeline from
`eval/save_embeddings_ct_rate.py`, which shares its head
(LoadImage -> EnsureChannelFirst -> ScaleIntensityRange -> Orientation(RAS) -> Spacing) with the
SSL training transforms in `spectre.ssl.transforms`.

The two agree bit-for-bit because they do the same operations: HU scaling and orientation commute
(orientation is pure axis flips/permutes, which do not touch intensities), and both resample with
MONAI's own `Spacing`. The one deliberate divergence - short axes - is pinned separately.
"""
import numpy as np
import pytest
import torch

pytest.importorskip("nibabel", reason="spectre.io needs the [inference] extra")
pytest.importorskip("monai", reason="the reference pipeline needs MONAI")

import nibabel as nib  # noqa: E402
from monai.transforms import (  # noqa: E402
    Compose,
    EnsureChannelFirstd,
    GridPatchd,
    LoadImaged,
    Orientationd,
    ScaleIntensityRanged,
    Spacingd,
)

from spectre.io import load_and_window, load_ct  # noqa: E402
from spectre.transforms import LargestMultipleCenterCropd  # noqa: E402

CROP = (128, 128, 64)

# Affines that place the scan in different orientations on disk, so RAS reorientation actually
# does something. Diagonal-negative -> axis flips; off-diagonal -> axis permutation. Spacings are
# chosen so that, at the shape below, every physical axis is at least ~150 mm - comfortably above
# the 128 mm the coarsest target spacing needs, so the short-axis path never triggers here.
AFFINES = {
    "RAS": np.diag([1.0, 1.0, 1.1, 1.0]),
    "LPS": np.diag([-1.0, -1.0, 1.1, 1.0]),          # radiological; flips two axes
    "rotated": np.array([[0, -1.1, 0, 10],           # I,L,A -> RAS needs an axis permutation
                         [0, 0, 0.9, -5],
                         [-1.0, 0, 0, 3],
                         [0, 0, 0, 1]], dtype=float),
}

# Every axis large enough that even coarse (1.0 mm) resampling stays above the crop on every axis
# and orientation: min physical extent is 170 * 0.9 = 153 mm > 128.
PARITY_SHAPE = (190, 200, 170)

SPACINGS = {
    "native": None,
    "fine": (0.6, 0.6, 1.0),
    "coarse": (1.0, 1.0, 1.0),
}


def reference_pipeline(path, spacing):
    """The deterministic preprocessing from eval/save_embeddings_ct_rate.py."""
    transforms = [
        LoadImaged(keys=("image",)),
        EnsureChannelFirstd(keys=("image",), channel_dim="no_channel"),
        ScaleIntensityRanged(
            keys=("image",), a_min=-1000, a_max=1000, b_min=0.0, b_max=1.0, clip=True,
        ),
        Orientationd(keys=("image",), axcodes="RAS"),
    ]
    if spacing is not None:
        transforms.append(Spacingd(keys=("image",), pixdim=spacing, mode=("bilinear",)))
    transforms += [
        LargestMultipleCenterCropd(keys=("image",), patch_size=CROP),
        GridPatchd(keys=("image",), patch_size=CROP, overlap=0.0),
    ]
    out = Compose(transforms)({"image": str(path)})["image"]
    return torch.as_tensor(np.asarray(out))


def write_scan(path, shape, affine, seed=0):
    rng = np.random.default_rng(seed)
    # Span well past the [-1000, 1000] window so HU clipping is genuinely exercised.
    data = (rng.random(shape) * 3400 - 1300).astype(np.float32)
    nib.save(nib.Nifti1Image(data, affine), str(path))
    return path


@pytest.mark.parametrize("orientation", list(AFFINES))
@pytest.mark.parametrize("spacing_name", list(SPACINGS))
def test_load_and_window_matches_monai_bit_exact(tmp_path, orientation, spacing_name):
    """Full pipeline from disk == the MONAI training/eval pipeline, to the bit.

    All axes are >= the crop, so the short-axis divergence does not apply here.
    """
    scan = write_scan(tmp_path / "scan.nii.gz", PARITY_SHAPE, AFFINES[orientation])
    spacing = SPACINGS[spacing_name]

    reference = reference_pipeline(scan, spacing)
    ours, grid, _ = load_and_window(scan, crop_size=CROP, spacing=spacing)

    assert ours.shape == reference.shape, (
        f"shape differs from MONAI for {orientation}/{spacing_name}"
    )
    assert torch.equal(ours, reference), (
        f"content differs from MONAI for {orientation}/{spacing_name} "
        f"(max abs diff {float((ours - reference).abs().max()):.2e})"
    )
    assert grid[0] * grid[1] * grid[2] == ours.shape[0]


def test_hu_clipping_is_applied(tmp_path):
    """Values outside [-1000, 1000] HU must be clipped to [0, 1], like ScaleIntensityRange."""
    scan = write_scan(tmp_path / "scan.nii.gz", PARITY_SHAPE, AFFINES["RAS"])
    ours, _, _ = load_and_window(scan, crop_size=CROP)
    assert float(ours.min()) == pytest.approx(0.0, abs=1e-6)
    assert float(ours.max()) == pytest.approx(1.0, abs=1e-6)


def test_non_ras_scan_loads(tmp_path):
    """Regression: reorienting a non-RAS scan yields negative numpy strides, which
    torch.from_numpy rejects unless the array is made contiguous first."""
    scan = write_scan(tmp_path / "lps.nii.gz", (140, 140, 140), AFFINES["LPS"])
    volume, meta = load_ct(scan)
    assert volume.shape[0] == 1
    assert meta.orientation == "RAS"
    assert torch.isfinite(volume).all()


def test_orientation_actually_reorients(tmp_path):
    """A scan stored non-RAS must come back RAS - proof the reorientation ran, not a no-op."""
    scan = write_scan(tmp_path / "rot.nii.gz", PARITY_SHAPE, AFFINES["rotated"])
    on_disk = "".join(nib.aff2axcodes(nib.load(str(scan)).affine))
    assert on_disk != "RAS"  # the fixture really is non-RAS
    _, meta = load_ct(scan)
    assert meta.orientation == "RAS"


def test_short_axis_is_the_only_intended_divergence(tmp_path):
    """Where a RAS axis is shorter than its crop, we deliberately differ from MONAI.

    MONAI keeps the short axis (LargestMultipleCenterCrop) and GridPatch emits an undersized
    patch - which the backbone's strict_img_size PatchEmbed would reject. load_and_window instead
    pads the axis to a full crop with air and warns, so the output is model-ingestible.
    """
    # 40 voxels on the last axis, well under the depth crop of 64.
    scan = write_scan(tmp_path / "thin.nii.gz", (200, 180, 40), AFFINES["RAS"])

    reference = reference_pipeline(scan, None)
    with pytest.warns(RuntimeWarning, match="smaller than one crop"):
        ours, _, _ = load_and_window(scan, crop_size=CROP)

    # MONAI leaves the short axis at 40; we pad it to the full 64.
    assert reference.shape[-1] == 40
    assert ours.shape[-1] == 64
    assert ours.shape[1:] == (1, 128, 128, 64)  # a shape the model can actually consume
