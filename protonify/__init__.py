"""
Protonify - Protonation state prediction for molecules at a given pH.

Simple usage:
    from protonify import predict_protonation
    result = predict_protonation("CCO", ph=7.4)
    print(result["smiles"])  # Most probable protonated SMILES

Environment Variables:
    PROTONIFY_MODEL_PATH: Path to the model file. If set, this path will be used
                          instead of downloading or looking in the cache.
"""

import os
import sys
import urllib.request
from pathlib import Path
from importlib.resources import files
from typing import Union, List, Dict, Any, Optional

# Default model URL (GitHub Releases)
DEFAULT_MODEL_URL = "https://github.com/pabloPauling/protonify/releases/download/v0.1.0/t_dwar_v_novartis_a_b.pt"
DEFAULT_MODEL_NAME = "t_dwar_v_novartis_a_b.pt"
CACHE_DIR = Path.home() / ".cache" / "protonify"

# Environment variable name for model path
MODEL_PATH_ENV_VAR = "PROTONIFY_MODEL_PATH"

# Lazy loading to avoid slow imports on module load
_predictor = None
_predictor_model_path = None
_templates = None


def get_cache_dir() -> Path:
    """Get the cache directory, creating it if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR


def download_model(url: str = DEFAULT_MODEL_URL, filename: str = DEFAULT_MODEL_NAME, force: bool = False) -> Path:
    """
    Download model from URL to cache directory.

    Args:
        url: URL to download from
        filename: Name to save the file as
        force: Re-download even if file exists

    Returns:
        Path to the downloaded model
    """
    cache_dir = get_cache_dir()
    model_path = cache_dir / filename

    if model_path.exists() and not force:
        return model_path

    print(f"Downloading model from {url}...")
    print(f"This may take a few minutes (model is ~500MB)")

    try:
        urllib.request.urlretrieve(url, model_path, _download_progress)
        print("\nDownload complete!")
    except Exception as e:
        if model_path.exists():
            model_path.unlink()
        raise RuntimeError(f"Failed to download model: {e}")

    return model_path


def _download_progress(block_num, block_size, total_size):
    """Show download progress."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        print(f"\rProgress: {percent:.1f}% ({mb_downloaded:.1f}/{mb_total:.1f} MB)", end="", flush=True)


def get_bundled_model_path() -> Optional[str]:
    """Check if model is bundled with the package."""
    try:
        model_path = files("protonify.models").joinpath(DEFAULT_MODEL_NAME)
        # Check if it's a real file (not just a resource reference)
        if hasattr(model_path, 'is_file') and model_path.is_file():
            return str(model_path)
        # For older Python/importlib versions
        model_str = str(model_path)
        if os.path.exists(model_str):
            return model_str
    except (TypeError, FileNotFoundError):
        pass
    return None


def get_model_path(model_path: Optional[str] = None) -> str:
    """
    Get the model path, downloading if necessary.

    Priority:
    1. Explicit model_path argument
    2. PROTONIFY_MODEL_PATH environment variable
    3. Bundled model in package (protonify/models/)
    4. Cached model in ~/.cache/protonify/
    5. Download from GitHub Releases

    Args:
        model_path: Optional explicit path to model file

    Returns:
        Path to the model file
    """
    # 1. Explicit path
    if model_path:
        if os.path.exists(model_path):
            return model_path
        raise FileNotFoundError(f"Model not found: {model_path}")

    # 2. Environment variable
    env_path = os.environ.get("PROTONIFY_MODEL_PATH")
    if env_path and os.path.exists(env_path):
        return env_path

    # 3. Check bundled model in package
    bundled = get_bundled_model_path()
    if bundled:
        return bundled

    # 4. Check cache
    cached_model = CACHE_DIR / DEFAULT_MODEL_NAME
    if cached_model.exists():
        return str(cached_model)

    # 5. Download
    return str(download_model())


def ensure_model(model_path: Optional[str] = None, set_env: bool = True) -> str:
    """
    Ensure the model is available, downloading if necessary.

    This function should be called before using the API to ensure the model
    is downloaded. It's especially useful when installing the package or
    in scripts that need to guarantee the model is available.

    Args:
        model_path: Optional explicit path to model file
        set_env: If True, set PROTONIFY_MODEL_PATH environment variable
                 to the resolved model path (default: True)

    Returns:
        Path to the model file

    Example:
        >>> from protonify import ensure_model
        >>> model_path = ensure_model()  # Downloads if needed, sets env var
        >>> print(f"Model at: {model_path}")

        # Or during installation:
        >>> ensure_model()  # Guarantees model is ready for API use
    """
    resolved_path = get_model_path(model_path)

    if set_env:
        os.environ[MODEL_PATH_ENV_VAR] = resolved_path

    return resolved_path


def _get_predictor(model_path: Optional[str] = None, use_gpu: bool = True):
    """Lazily load the predictor (only once per model path + device combo)."""
    global _predictor, _predictor_model_path

    resolved_path = get_model_path(model_path)
    cache_key = (resolved_path, use_gpu)

    # Return cached predictor if same model and device
    if _predictor is not None and _predictor_model_path == cache_key:
        return _predictor

    from protonify.FreeEnergyPredictor import FreeEnergyPredictor
    _predictor = FreeEnergyPredictor(resolved_path, batch_size=32, remove_hs=False, use_gpu=use_gpu)
    _predictor_model_path = cache_key
    return _predictor


def _get_templates(template_type: str = "smart"):
    """Lazily load templates (cached)."""
    global _templates
    if _templates is None:
        _templates = {}
    if template_type not in _templates:
        from protonify.pka_utils import read_template
        filename = "simple_smarts_pattern.tsv" if template_type == "simple" else "smarts_pattern.tsv"
        template_path = str(files("protonify.data").joinpath(filename))
        _templates[template_type] = read_template(template_path)
    return _templates[template_type]


def predict_protonation(
    smiles: str,
    ph: Union[float, List[float]] = 7.4,
    template: str = "smart",
    skip_tautomers: bool = True,
    model_path: Optional[str] = None,
    use_gpu: bool = True,
    quiet: bool = True,
) -> Dict[str, Any]:
    """
    Predict the most probable protonation state for a molecule at a given pH.

    Args:
        smiles: SMILES string of the molecule
        ph: pH value or list of pH values (default: 7.4)
        template: Template type, "simple" or "smart" (default: "smart")
        skip_tautomers: Skip tautomer enumeration for speed (default: True)
        model_path: Optional path to custom model file (default: auto-download)
        use_gpu: Use GPU for inference if available (default: True, falls back to CPU)
        quiet: Suppress all logging and print output (default: True)

    Returns:
        Dictionary with:
            - smiles: Most probable protonated SMILES
            - charge: Formal charge
            - probability: Probability of this state
            - ph: pH value used
            - full_distribution: All microstates with probabilities (if single pH)

    Example:
        >>> from protonify import predict_protonation
        >>> result = predict_protonation("CCO", ph=7.4)
        >>> print(result["smiles"])

        # Use GPU
        >>> result = predict_protonation("CCO", ph=7.4, use_gpu=True)

        # Show verbose output
        >>> result = predict_protonation("CCO", ph=7.4, quiet=False)
    """
    import logging
    import warnings

    # IMPORTANT: Resolve model path BEFORE entering quiet mode
    # This ensures that if the model needs to be downloaded, the user sees progress
    resolved_model_path = get_model_path(model_path)

    # Set environment variable so subsequent calls find the model immediately
    os.environ[MODEL_PATH_ENV_VAR] = resolved_model_path

    # Setup quiet mode AFTER model is resolved
    if quiet:
        logging.disable(logging.CRITICAL)
        warnings.filterwarnings('ignore')
        # Redirect stdout/stderr to suppress print statements
        import io
        _stdout = sys.stdout
        _stderr = sys.stderr
        sys.stdout = io.StringIO()
        sys.stderr = io.StringIO()

    try:
        from protonify.pka_utils import predict_ensemble_free_energy
        from protonify.ph_utils import process_single_ph

        # Use the already-resolved model path
        predictor = _get_predictor(resolved_model_path, use_gpu=use_gpu)
        template_a2b, template_b2a = _get_templates(template)

        ensemble_free_energy = predict_ensemble_free_energy(
            smiles, template_a2b, template_b2a, predictor, skip_tautomers=skip_tautomers
        )

        # Handle single pH or list of pH values
        if isinstance(ph, (int, float)):
            result = process_single_ph(ph, ensemble_free_energy, smiles)
            return {
                "smiles": result["max_microstate"],
                "charge": result["max_charge"],
                "probability": result["probability"],
                "ph": ph,
                "full_distribution": result["full_distribution"],
            }
        else:
            # Multiple pH values
            results = []
            for p in ph:
                result = process_single_ph(p, ensemble_free_energy, smiles)
                results.append({
                    "smiles": result["max_microstate"],
                    "charge": result["max_charge"],
                    "probability": result["probability"],
                    "ph": p,
                })
            return results

    finally:
        # Restore stdout/stderr if quiet mode was enabled
        if quiet:
            sys.stdout = _stdout
            sys.stderr = _stderr
            logging.disable(logging.NOTSET)


__all__ = [
    "predict_protonation",
    "download_model",
    "get_model_path",
    "get_bundled_model_path",
    "ensure_model",
    "MODEL_PATH_ENV_VAR",
    "CACHE_DIR",
    "DEFAULT_MODEL_NAME",
]
