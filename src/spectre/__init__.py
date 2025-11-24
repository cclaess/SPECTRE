"""Top-level package for spectre.

Expose a small, stable public API here so users can do:

	from spectre import SpectreImageFeatureExtractor, models

Keep implementations in subpackages; this file only re-exports the most
important symbols and subpackages for convenience.
"""
from .model import SpectreImageFeatureExtractor, MODEL_CONFIGS
from . import models
from . import data
from . import transforms
from . import ssl
from . import utils

__version__ = "0.1.0"
__author__ = "Cris Claessens"
__email__ = "c.h.b.claessens@tue.nl"

__all__ = [
	"SpectreImageFeatureExtractor",
	"MODEL_CONFIGS",
	"models",
	"data",
	"transforms",
	"ssl",
	"utils",
	"__version__",
    "__author__",
    "__email__",
]
