"""Tests for tautomer enumeration limits and truncation reporting (issue #5)."""

import logging
import sys
import warnings
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
from rdkit.Chem import MolFromSmiles, CanonSmiles, MolToSmiles
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumeratorStatus

import protonify
from protonify import pka_utils
from protonify.pka_utils import (
    DEFAULT_MAX_TAUTOMERS,
    TRUNCATED_MAX_TAUTOMERS,
    TRUNCATED_MAX_TRANSFORMS,
    TautomerLimitWarning,
    describe_tautomer_truncations,
    enumerate_tautomers,
    enumerate_template,
    get_ensemble,
    predict_ensemble_free_energy,
    read_template,
)

# Cyanuric acid: exactly 5 tautomers
MANY_TAUTOMERS = "Oc1nc(O)nc(O)n1"
N_TAUTOMERS = 5


def _limit_warnings(caught):
    return [w for w in caught if issubclass(w.category, TautomerLimitWarning)]


@pytest.fixture(scope="module")
def templates():
    from importlib.resources import files
    return read_template(str(files("protonify.data").joinpath("smarts_pattern.tsv")))


class FakePredictor:
    """Stands in for FreeEnergyPredictor: every microstate gets DfGm = 0."""

    def predict(self, smis):
        return {s: 0.0 for s in smis}


@pytest.fixture
def fake_model(monkeypatch):
    """Run predict_protonation without loading or downloading the model."""
    monkeypatch.setenv(protonify.MODEL_PATH_ENV_VAR, "unused")
    monkeypatch.setattr(protonify, "get_model_path", lambda model_path=None: "unused")
    monkeypatch.setattr(protonify, "_get_predictor", lambda *a, **k: FakePredictor())


class TestEnumerateTautomers:
    def test_complete_enumeration_does_not_warn(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tauts = enumerate_tautomers(MolFromSmiles(MANY_TAUTOMERS), max_tautomers=100)
        assert len(tauts) == N_TAUTOMERS
        assert not _limit_warnings(caught)

    def test_max_tautomers_is_respected(self):
        with pytest.warns(TautomerLimitWarning, match="more than 3 tautomers"):
            tauts = enumerate_tautomers(MolFromSmiles(MANY_TAUTOMERS), max_tautomers=3)
        assert len(tauts) == 3

    def test_exactly_at_limit_is_not_truncated(self):
        """RDKit reports MaxTautomersReached when count == cap; that must not count."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            tauts = enumerate_tautomers(MolFromSmiles(MANY_TAUTOMERS), max_tautomers=N_TAUTOMERS)
        assert len(tauts) == N_TAUTOMERS
        assert not _limit_warnings(caught)

    def test_one_below_limit_is_truncated(self):
        with pytest.warns(TautomerLimitWarning):
            tauts = enumerate_tautomers(MolFromSmiles(MANY_TAUTOMERS), max_tautomers=N_TAUTOMERS - 1)
        assert len(tauts) == N_TAUTOMERS - 1

    def test_different_limits_do_not_interfere(self):
        mol = MolFromSmiles(MANY_TAUTOMERS)
        with pytest.warns(TautomerLimitWarning):
            assert len(enumerate_tautomers(mol, max_tautomers=2)) == 2
        assert len(enumerate_tautomers(mol, max_tautomers=100)) == N_TAUTOMERS
        with pytest.warns(TautomerLimitWarning):
            assert len(enumerate_tautomers(mol, max_tautomers=2)) == 2

    def test_default_limit(self):
        assert DEFAULT_MAX_TAUTOMERS == 50
        assert len(enumerate_tautomers(MolFromSmiles(MANY_TAUTOMERS))) == N_TAUTOMERS

    @pytest.mark.parametrize("bad", [0, -1, 2.5, "10", True])
    def test_invalid_limit_raises(self, bad):
        with pytest.raises(ValueError):
            enumerate_tautomers(MolFromSmiles("CCO"), max_tautomers=bad)

    def test_none_mol(self):
        assert enumerate_tautomers(None) == []

    def test_max_transforms_reason(self, monkeypatch):
        """Hitting RDKit's transform limit is reported as such, not as max_tautomers."""
        mol = MolFromSmiles(MANY_TAUTOMERS)

        class FakeResult(list):
            status = TautomerEnumeratorStatus.MaxTransformsReached

        fake_enum = SimpleNamespace(Enumerate=lambda m: FakeResult([m]))
        monkeypatch.setattr(pka_utils, "_get_tautomer_enumerator", lambda cap: fake_enum)
        tauts, reason = pka_utils._enumerate_tautomers_checked(mol, 50)
        assert reason == TRUNCATED_MAX_TRANSFORMS
        assert len(tauts) == 1
        with pytest.warns(TautomerLimitWarning, match="transform limit"):
            enumerate_tautomers(mol, max_tautomers=50)


class TestDescribeTruncations:
    def test_counts_and_advice_per_reason(self):
        msg = describe_tautomer_truncations(
            {"A": TRUNCATED_MAX_TAUTOMERS, "B": TRUNCATED_MAX_TAUTOMERS, "C": TRUNCATED_MAX_TRANSFORMS},
            max_tautomers=7, option="--max-tautomers",
        )
        assert "2 molecule(s) had more than 7 tautomers (increase --max-tautomers)" in msg
        assert "1 molecule(s) hit RDKit's internal tautomer transform limit (not controlled by --max-tautomers)" in msg

    def test_other_status(self):
        assert "status Canceled" in describe_tautomer_truncations({"A": "Canceled"}, 50)


class TestPipeline:
    def test_enumerate_template_passes_limit(self, templates):
        a2b, b2a = templates
        calls = []

        def fake(mol, max_tautomers):
            calls.append(max_tautomers)
            return [mol], None

        with patch.object(pka_utils, "_enumerate_tautomers_checked", side_effect=fake):
            enumerate_template("CC(=O)O", a2b, b2a, mode="a2b", max_tautomers=7)
        assert calls and set(calls) == {7}

    def test_enumerate_template_skip_ignores_limit(self, templates):
        a2b, b2a = templates
        with patch.object(pka_utils, "_enumerate_tautomers_checked") as mocked:
            enumerate_template("CC(=O)O", a2b, b2a, mode="a2b", skip_tautomers=True, max_tautomers=0)
        mocked.assert_not_called()

    def test_get_ensemble_warns_once_with_distinct_molecules(self, templates):
        a2b, b2a = templates
        truncations = {}
        get_ensemble(MANY_TAUTOMERS, a2b, b2a, max_tautomers=1, truncations=truncations)
        assert truncations and set(truncations.values()) == {TRUNCATED_MAX_TAUTOMERS}

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            get_ensemble(MANY_TAUTOMERS, a2b, b2a, max_tautomers=1)
        limit = _limit_warnings(caught)
        assert len(limit) == 1
        assert f"{len(truncations)} molecule(s)" in str(limit[0].message)

    def test_truncations_out_param_suppresses_warning(self, templates):
        a2b, b2a = templates
        truncations = {}
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            predict_ensemble_free_energy(MANY_TAUTOMERS, a2b, b2a, FakePredictor(),
                                         max_tautomers=1, truncations=truncations)
        assert truncations
        assert not _limit_warnings(caught)
        # keys are canonical SMILES
        assert all(CanonSmiles(s) == s for s in truncations)

    def test_get_ensemble_limit_changes_result(self, templates):
        a2b, b2a = templates
        with pytest.warns(TautomerLimitWarning):
            small = get_ensemble(MANY_TAUTOMERS, a2b, b2a, max_tautomers=1)
        truncations = {}
        large = get_ensemble(MANY_TAUTOMERS, a2b, b2a, max_tautomers=1000, truncations=truncations)
        assert truncations == {}
        n_small = sum(len(v) for v in small.values())
        n_large = sum(len(v) for v in large.values())
        assert n_large > n_small
        expected = {CanonSmiles(MolToSmiles(m)) for m in enumerate_tautomers(MolFromSmiles(MANY_TAUTOMERS), 1000)}
        assert expected <= set(large[0])


class TestPredictProtonationReporting:
    def test_truncation_reported_in_quiet_mode(self, fake_model):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = protonify.predict_protonation(MANY_TAUTOMERS, skip_tautomers=False, max_tautomers=1)
        assert result["tautomer_limit_reached"] is True
        assert set(result["tautomer_truncations"].values()) == {TRUNCATED_MAX_TAUTOMERS}
        limit = _limit_warnings(caught)
        assert len(limit) == 1
        assert limit[0].filename == __file__  # attributed to the caller

    def test_no_truncation(self, fake_model):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = protonify.predict_protonation(MANY_TAUTOMERS, skip_tautomers=False, max_tautomers=1000)
        assert result["tautomer_limit_reached"] is False
        assert result["tautomer_truncations"] == {}
        assert not _limit_warnings(caught)

    def test_quiet_mode_restores_global_state(self, fake_model):
        filters_before = list(warnings.filters)
        logging_before = logging.root.manager.disable
        stdout_before, stderr_before = sys.stdout, sys.stderr
        protonify.predict_protonation("CC(=O)O", quiet=True)
        assert warnings.filters == filters_before
        assert logging.root.manager.disable == logging_before
        assert sys.stdout is stdout_before and sys.stderr is stderr_before

    def test_quiet_mode_restores_state_on_error(self, fake_model):
        filters_before = list(warnings.filters)
        stdout_before = sys.stdout
        with pytest.raises(Exception):
            protonify.predict_protonation("not a smiles", quiet=True)
        assert warnings.filters == filters_before
        assert sys.stdout is stdout_before

    def test_quiet_mode_preserves_callers_logging_disable(self, fake_model):
        logging.disable(logging.WARNING)
        try:
            protonify.predict_protonation("CC(=O)O", quiet=True)
            assert logging.root.manager.disable == logging.WARNING
        finally:
            logging.disable(logging.NOTSET)

    def test_max_tautomers_ignored_when_skipping(self, fake_model):
        result = protonify.predict_protonation("CC(=O)O", skip_tautomers=True, max_tautomers=0)
        assert result["tautomer_limit_reached"] is False

    def test_max_tautomers_validated_when_enumerating(self, fake_model):
        with pytest.raises(ValueError):
            protonify.predict_protonation("CC(=O)O", skip_tautomers=False, max_tautomers=0)

    @pytest.mark.parametrize("ph", [np.float32(7.4), np.float64(7.4), np.int64(7), 7, 7.4])
    def test_scalar_ph_types(self, fake_model, ph):
        result = protonify.predict_protonation("CC(=O)O", ph=ph)
        assert isinstance(result, dict)
        assert result["ph"] == ph

    def test_list_ph(self, fake_model):
        results = protonify.predict_protonation("CC(=O)O", ph=np.array([3.0, 7.4]))
        assert isinstance(results, list) and len(results) == 2
        for r in results:
            assert "full_distribution" in r and "free_energies" in r


class TestCli:
    def _run(self, argv, monkeypatch):
        from protonify import __main__ as cli
        monkeypatch.setattr(cli, "ensure_model", lambda path=None: "unused")
        monkeypatch.setattr(cli, "FreeEnergyPredictor", lambda *a, **k: FakePredictor())
        monkeypatch.setattr(sys, "argv", ["protonify"] + argv)
        return cli.main()

    def test_truncation_warning_counts_distinct_molecules(self, monkeypatch, capsys, templates):
        a2b, b2a = templates
        truncations = {}
        get_ensemble(MANY_TAUTOMERS, a2b, b2a, max_tautomers=2, truncations=truncations)

        self._run(["--smiles", MANY_TAUTOMERS, "--ph", "7.4", "--template", "smart",
                   "--enumerate-tautomers", "--max-tautomers", "2", "-q"], monkeypatch)
        out, err = capsys.readouterr()
        assert err.count("WARNING:") == 1
        assert f"{len(truncations)} molecule(s) had more than 2 tautomers (increase --max-tautomers)" in err
        assert "WARNING" not in out

    def test_no_warning_when_complete(self, monkeypatch, capsys):
        self._run(["--smiles", MANY_TAUTOMERS, "--ph", "7.4", "--template", "smart",
                   "--enumerate-tautomers", "--max-tautomers", "1000", "-q"], monkeypatch)
        assert "WARNING" not in capsys.readouterr().err

    def test_max_tautomers_requires_enumerate(self, monkeypatch, capsys):
        with pytest.raises(SystemExit):
            self._run(["--smiles", "CCO", "--ph", "7.4", "--template", "smart", "--max-tautomers", "10"], monkeypatch)
        assert "--max-tautomers requires --enumerate-tautomers" in capsys.readouterr().err

    @pytest.mark.parametrize("bad", ["0", "-3", "abc", "2.5"])
    def test_max_tautomers_must_be_positive_int(self, bad, monkeypatch, capsys):
        with pytest.raises(SystemExit):
            self._run(["--smiles", "CCO", "--ph", "7.4", "--template", "smart",
                       "--enumerate-tautomers", "--max-tautomers", bad], monkeypatch)
        assert "must be a positive integer" in capsys.readouterr().err
