"""Top-level package for spectre.

Expose a small, stable public API here so users can do:

	from spectre import SpectreImageFeatureExtractor
	model = SpectreImageFeatureExtractor.from_pretrained("spectre-large")

Keep implementations in subpackages; this file only re-exports the most
important symbols and subpackages for convenience.

Note for maintainers: `scripts/export_hf.py` rewrites this file when vendoring the package into
the Hugging Face Hub repo, dropping the imports it does not ship. Never import `.cli` here, or
anything that needs an optional dependency at module level.
"""
from .model import SpectreImageFeatureExtractor, MODEL_CONFIGS
from .presets import SpectrePreset, PRESETS, get_preset, list_presets, list_pretrained
from .windowing import window_scan, largest_multiple_center_crop, grid_patch, scale_intensity_range
from .io import load_ct, load_and_window, find_ct_files
from . import models
from . import data
from . import transforms
from . import ssl
from . import utils

__version__ = "0.2.1"
__author__ = "Cris Claessens"
__email__ = "c.h.b.claessens@tue.nl"

__all__ = [
	"SpectreImageFeatureExtractor",
	"MODEL_CONFIGS",
	"SpectrePreset",
	"PRESETS",
	"get_preset",
	"list_presets",
	"list_pretrained",
	"window_scan",
	"largest_multiple_center_crop",
	"grid_patch",
	"scale_intensity_range",
	"load_ct",
	"load_and_window",
	"find_ct_files",
	"models",
	"data",
	"transforms",
	"ssl",
	"utils",
	"__version__",
    "__author__",
    "__email__",
]
