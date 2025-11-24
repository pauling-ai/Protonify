#!/usr/bin/env python3
"""
Test script to verify Protonify API works correctly.
Runs several examples from the README.
"""

from protonify import predict_protonation

def test_basic_prediction():
    """Basic prediction with a single pH."""
    print("=" * 60)
    print("TEST 1: Basic prediction (ethanol at pH 7.4)")
    print("=" * 60)

    result = predict_protonation("CCO", ph=7.4)

    print(f"  Most probable SMILES: {result['smiles']}")
    print(f"  Formal charge:        {result['charge']}")
    print(f"  Probability:          {result['probability']:.4f}")
    print()

def test_multiple_ph():
    """Prediction with multiple pH values."""
    print("=" * 60)
    print("TEST 2: Multiple pH values (ethanol)")
    print("=" * 60)

    results = predict_protonation("CCO", ph=[7.0, 7.4, 8.0])

    for r in results:
        print(f"  pH {r['ph']}: {r['smiles']} (charge={r['charge']}, prob={r['probability']:.4f})")
    print()

def test_different_templates():
    """Test with different templates (simple vs smart)."""
    print("=" * 60)
    print("TEST 3: Template comparison (glycine)")
    print("=" * 60)

    smiles = "NCC(=O)O"  # Glycine - amino acid with amino and carboxyl groups
    ph = 7.4

    result_simple = predict_protonation(smiles, ph=ph, template="simple")
    result_smart = predict_protonation(smiles, ph=ph, template="smart")

    print(f"  Template 'simple': {result_simple['smiles']} (charge={result_simple['charge']})")
    print(f"  Template 'smart':  {result_smart['smiles']} (charge={result_smart['charge']})")
    print()

def test_ionizable_molecule():
    """Test with ionizable molecule at different pH."""
    print("=" * 60)
    print("TEST 4: Ionizable molecule (acetic acid) at different pH")
    print("=" * 60)

    smiles = "CC(=O)O"  # Acetic acid (pKa ~4.75)
    ph_values = [2.0, 4.75, 7.0, 10.0]

    results = predict_protonation(smiles, ph=ph_values)

    for r in results:
        print(f"  pH {r['ph']:5.2f}: {r['smiles']:15s} (charge={r['charge']:+d}, prob={r['probability']:.4f})")
    print()

def test_drug_molecule():
    """Test with a drug-like molecule."""
    print("=" * 60)
    print("TEST 5: Pharmaceutical molecule (aspirin)")
    print("=" * 60)

    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    ph = 7.4  # Physiological pH

    result = predict_protonation(smiles, ph=ph)

    print(f"  Original SMILES:     {smiles}")
    print(f"  SMILES at pH 7.4:    {result['smiles']}")
    print(f"  Charge:              {result['charge']}")
    print(f"  Probability:         {result['probability']:.4f}")
    print()

def test_with_tautomers():
    """Test with tautomer enumeration enabled."""
    print("=" * 60)
    print("TEST 6: With tautomer enumeration (imidazole)")
    print("=" * 60)

    smiles = "c1c[nH]cn1"  # Imidazole
    ph = 7.4

    # Without tautomers (faster)
    result_no_taut = predict_protonation(smiles, ph=ph, skip_tautomers=True)
    print(f"  Without tautomers: {result_no_taut['smiles']} (charge={result_no_taut['charge']})")

    # With tautomers (more thorough)
    result_taut = predict_protonation(smiles, ph=ph, skip_tautomers=False)
    print(f"  With tautomers:    {result_taut['smiles']} (charge={result_taut['charge']})")
    print()

def main():
    """Run all tests."""
    print("\n" + "#" * 60)
    print("# PROTONIFY API TESTS")
    print("#" * 60 + "\n")

    try:
        test_basic_prediction()
        test_multiple_ph()
        test_different_templates()
        test_ionizable_molecule()
        test_drug_molecule()
        test_with_tautomers()

        print("=" * 60)
        print("ALL TESTS COMPLETED SUCCESSFULLY")
        print("=" * 60)

    except Exception as e:
        print(f"\nERROR during tests: {e}")
        raise

if __name__ == "__main__":
    main()
