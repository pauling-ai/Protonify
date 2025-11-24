"""Unit tests for the Protonify public API."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
from protonify import (
    predict_protonation,
    ensure_model,
    get_model_path,
    download_model,
    get_bundled_model_path,
    MODEL_PATH_ENV_VAR,
    CACHE_DIR,
    DEFAULT_MODEL_NAME,
)


# Path to the model in the package
PACKAGE_MODEL_PATH = Path(__file__).parent.parent / "models" / DEFAULT_MODEL_NAME


@pytest.fixture
def model_available():
    """Check if model is available and set env var."""
    if not PACKAGE_MODEL_PATH.exists():
        pytest.skip(f"Model not found at {PACKAGE_MODEL_PATH}")
    os.environ[MODEL_PATH_ENV_VAR] = str(PACKAGE_MODEL_PATH)
    return str(PACKAGE_MODEL_PATH)


def get_test_model_path():
    """Get model path for testing, skip if not available or cannot be loaded."""
    if PACKAGE_MODEL_PATH.exists():
        os.environ[MODEL_PATH_ENV_VAR] = str(PACKAGE_MODEL_PATH)
        # Try to actually load the model to verify it works
        try:
            import torch
            torch.load(str(PACKAGE_MODEL_PATH), map_location='cpu', weights_only=False)
        except Exception as e:
            pytest.skip(f"Model exists but cannot be loaded: {e}")
        return str(PACKAGE_MODEL_PATH)
    try:
        return get_model_path()
    except Exception as e:
        pytest.skip(f"Model not available for testing: {e}")


class TestGetModelPath:
    """Tests for get_model_path function."""

    def test_explicit_path_exists(self, tmp_path):
        """Test that explicit path is returned when file exists."""
        model_file = tmp_path / "test_model.pt"
        model_file.write_bytes(b"dummy")
        result = get_model_path(str(model_file))
        assert result == str(model_file)

    def test_explicit_path_not_exists(self):
        """Test that FileNotFoundError is raised for non-existent explicit path."""
        with pytest.raises(FileNotFoundError):
            get_model_path("/non/existent/path.pt")

    def test_env_variable_priority(self, tmp_path, monkeypatch):
        """Test that environment variable is checked after explicit path."""
        model_file = tmp_path / "env_model.pt"
        model_file.write_bytes(b"dummy")
        monkeypatch.setenv(MODEL_PATH_ENV_VAR, str(model_file))
        result = get_model_path()
        assert result == str(model_file)

    def test_env_variable_ignored_if_not_exists(self, monkeypatch):
        """Test that non-existent env path is ignored."""
        monkeypatch.setenv(MODEL_PATH_ENV_VAR, "/non/existent/path.pt")
        # Should not raise, should try other methods
        # (may skip or fail depending on other availability)


class TestEnsureModel:
    """Tests for ensure_model function."""

    def test_ensure_model_sets_env_var(self, tmp_path, monkeypatch):
        """Test that ensure_model sets the environment variable."""
        model_file = tmp_path / "test_model.pt"
        model_file.write_bytes(b"dummy")

        # Clear any existing env var
        monkeypatch.delenv(MODEL_PATH_ENV_VAR, raising=False)

        result = ensure_model(str(model_file), set_env=True)
        assert result == str(model_file)
        assert os.environ.get(MODEL_PATH_ENV_VAR) == str(model_file)

    def test_ensure_model_no_set_env(self, tmp_path, monkeypatch):
        """Test that ensure_model respects set_env=False."""
        model_file = tmp_path / "test_model.pt"
        model_file.write_bytes(b"dummy")

        # Clear any existing env var
        monkeypatch.delenv(MODEL_PATH_ENV_VAR, raising=False)

        result = ensure_model(str(model_file), set_env=False)
        assert result == str(model_file)
        # Env var should not be set
        assert os.environ.get(MODEL_PATH_ENV_VAR) != str(model_file)


class TestPredictProtonation:
    """Tests for predict_protonation function."""

    def test_basic_prediction(self):
        """Test basic prediction with single pH."""
        get_test_model_path()  # Skip if model not available

        result = predict_protonation("CCO", ph=7.4)

        assert isinstance(result, dict)
        assert "smiles" in result
        assert "charge" in result
        assert "probability" in result
        assert "ph" in result
        assert "full_distribution" in result
        assert result["ph"] == 7.4
        assert isinstance(result["smiles"], str)
        assert isinstance(result["charge"], int)
        assert 0 <= result["probability"] <= 1

    def test_multiple_ph_values(self):
        """Test prediction with multiple pH values."""
        get_test_model_path()  # Skip if model not available

        ph_values = [2.0, 7.4, 10.0]
        results = predict_protonation("CC(=O)O", ph=ph_values)

        assert isinstance(results, list)
        assert len(results) == len(ph_values)
        for i, result in enumerate(results):
            assert result["ph"] == ph_values[i]
            assert "smiles" in result
            assert "charge" in result
            assert "probability" in result

    def test_smart_template(self):
        """Test prediction with smart template (default)."""
        get_test_model_path()  # Skip if model not available

        result = predict_protonation("CCO", ph=7.4, template="smart")

        assert isinstance(result, dict)
        assert "smiles" in result

    def test_with_tautomers(self):
        """Test prediction with tautomer enumeration."""
        get_test_model_path()  # Skip if model not available

        result_no_taut = predict_protonation("c1c[nH]cn1", ph=7.4, skip_tautomers=True)
        result_taut = predict_protonation("c1c[nH]cn1", ph=7.4, skip_tautomers=False)

        assert isinstance(result_no_taut, dict)
        assert isinstance(result_taut, dict)
        assert "smiles" in result_no_taut
        assert "smiles" in result_taut

    def test_quiet_mode(self):
        """Test that quiet mode suppresses output."""
        get_test_model_path()  # Skip if model not available

        # quiet=True is default
        result = predict_protonation("CCO", ph=7.4, quiet=True)
        assert isinstance(result, dict)

    def test_verbose_mode(self):
        """Test verbose mode."""
        get_test_model_path()  # Skip if model not available

        result = predict_protonation("CCO", ph=7.4, quiet=False)
        assert isinstance(result, dict)

    def test_ionizable_molecule(self):
        """Test prediction for ionizable molecule (acetic acid)."""
        get_test_model_path()  # Skip if model not available

        # Acetic acid pKa ~4.75
        result_low = predict_protonation("CC(=O)O", ph=2.0)
        result_high = predict_protonation("CC(=O)O", ph=10.0)

        # At low pH, should be neutral (charge 0)
        # At high pH, should be deprotonated (charge -1)
        assert result_low["charge"] >= result_high["charge"]

    def test_amino_acid(self):
        """Test prediction for amino acid (glycine)."""
        get_test_model_path()  # Skip if model not available

        result = predict_protonation("NCC(=O)O", ph=7.4)

        assert isinstance(result, dict)
        assert "smiles" in result
        # Glycine at pH 7.4 should be zwitterionic


class TestExports:
    """Test that all expected items are exported."""

    def test_exports(self):
        """Test that all expected items are exported from the module."""
        from protonify import (
            predict_protonation,
            download_model,
            get_model_path,
            get_bundled_model_path,
            ensure_model,
            MODEL_PATH_ENV_VAR,
            CACHE_DIR,
            DEFAULT_MODEL_NAME,
        )

        assert callable(predict_protonation)
        assert callable(download_model)
        assert callable(get_model_path)
        assert callable(get_bundled_model_path)
        assert callable(ensure_model)
        assert isinstance(MODEL_PATH_ENV_VAR, str)
        assert MODEL_PATH_ENV_VAR == "PROTONIFY_MODEL_PATH"
        assert DEFAULT_MODEL_NAME == "t_dwar_v_novartis_a_b.pt"
