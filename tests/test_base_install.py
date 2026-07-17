"""`pip install spectre-fm` must be enough to load and run the model.

The optional [inference] dependencies (nibabel, monai) may only be required when actually reading
a file off disk. These tests run in a subprocess with those modules blocked, which is the only
honest way to check it from an environment that has them installed.
"""
import subprocess
import sys
import textwrap

import pytest

PREAMBLE = textwrap.dedent("""
    import sys

    class Blocker:
        BLOCKED = {'nibabel', 'monai'}
        def find_spec(self, name, path=None, target=None):
            if name.split('.')[0] in self.BLOCKED:
                raise ImportError(f'No module named {name!r} (simulated base install)')
            return None

    sys.meta_path.insert(0, Blocker())

    import warnings
    warnings.filterwarnings('ignore')
""")


def run_without_optional_deps(body: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", PREAMBLE + textwrap.dedent(body)],
        capture_output=True, text=True,
    )


def test_import_spectre_without_optional_deps():
    result = run_without_optional_deps("""
        import spectre
        print(spectre.__version__)
    """)
    assert result.returncode == 0, f"import spectre failed on a base install:\n{result.stderr}"


def test_model_runs_without_optional_deps():
    result = run_without_optional_deps("""
        import torch
        from spectre import SpectreImageFeatureExtractor
        model = SpectreImageFeatureExtractor.from_pretrained('spectre-small', pretrained=False)
        with torch.no_grad():
            out = model(torch.rand(1, 128, 128, 64) * 3000 - 1200)
        assert out.ndim == 2, out.shape
        print('ok')
    """)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_windowing_runs_without_optional_deps():
    result = run_without_optional_deps("""
        import torch
        from spectre import window_scan
        crops, grid = window_scan(torch.rand(1, 256, 128, 64) * 3000 - 1200)
        assert crops.shape[0] == 2 and grid == (2, 1, 1)
        print('ok')
    """)
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout


def test_load_ct_raises_an_actionable_error_without_nibabel():
    result = run_without_optional_deps("""
        import spectre
        try:
            spectre.load_ct('nope.nii.gz')
        except ImportError as e:
            print('IMPORTERROR:', e)
    """)
    assert result.returncode == 0, result.stderr
    assert "spectre-fm[inference]" in result.stdout, (
        "the error must tell the user which extra to install"
    )


def test_cli_help_works_without_optional_deps():
    result = run_without_optional_deps("""
        from spectre.cli import main
        raise SystemExit(main(['list-models']))
    """)
    assert result.returncode == 0, result.stderr
    assert "spectre-large" in result.stdout


@pytest.mark.parametrize("module", ["spectre.model", "spectre.windowing", "spectre.presets"])
def test_core_modules_import_without_optional_deps(module):
    result = run_without_optional_deps(f"""
        import {module}
        print('ok')
    """)
    assert result.returncode == 0, f"{module} needs an optional dependency:\n{result.stderr}"
