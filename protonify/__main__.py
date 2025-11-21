import os
import sys
import logging
import warnings
import argparse
from importlib.resources import files
from rdkit import RDLogger
from protonify import get_model_path
from protonify.pka_utils import FreeEnergyPredictor, read_template, predict_ensemble_free_energy
from protonify.ph_utils import process_single_ph, parse_ph_values

# Disable RDKit warnings and general Python warnings
RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings(action='ignore')

# Configure logging
logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger("protonify.inference")
logging.disable(50)  # Disable all logging below CRITICAL


def get_template_path(template_type: str) -> str:
    """Get the path to the template file from package data."""
    filename = "simple_smarts_pattern.tsv" if template_type == "simple" else "smarts_pattern.tsv"
    return str(files("protonify.data").joinpath(filename))


def main():
    parser = argparse.ArgumentParser(description="Prediction of free energy and microstate distribution for a given SMILES.")
    parser.add_argument('--template', choices=['simple', 'smart'], required=True, help='Template type to use (simple or smart)')
    parser.add_argument('--ph', type=parse_ph_values, required=True,
                       help='pH level(s) to calculate the distribution. Can be a single value or comma-separated list (e.g., "7.4" or "7.0,7.4,8.0")')
    parser.add_argument('--smiles', type=str, required=True, help='SMILES of the molecule to analyze')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the pretrained model (defaults to auto-download)')
    parser.add_argument('--enumerate-tautomers', action='store_true', help='Enumerate tautomers (slower but more thorough). Default: skip tautomers for speed.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output, only print final SMILES result')
    args = parser.parse_args()

    # Get model path (auto-downloads if needed)
    model_path = get_model_path(args.model_path)

    predictor = FreeEnergyPredictor(
        model_path,
        batch_size=32,
        remove_hs=False,
        use_gpu=args.gpu
    )
    if not args.quiet:
        print(f"FreeEnergyPredictor loaded from {model_path}")
        print(f"Using rdkit for conformer generation")

    # Load templates from package data
    template_path = get_template_path(args.template)
    template_a2b, template_b2a = read_template(template_path)

    smi_drug = args.smiles
    if not args.quiet:
        print(f"Processing SMILES: {smi_drug}")

    # Calculate ensemble free energy once for all pH values
    skip_tautomers = not args.enumerate_tautomers
    if not args.quiet:
        print("Calculating ensemble free energy...")
        if skip_tautomers:
            print("Tautomer enumeration: DISABLED (fast mode)")
        else:
            print("Tautomer enumeration: ENABLED")

    drug_ensemble_free_energy = predict_ensemble_free_energy(smi_drug, template_a2b, template_b2a, predictor, skip_tautomers=skip_tautomers)

    if not args.quiet:
        print("Free energy of each macrostate (by charge):")
        for q, macrostate in drug_ensemble_free_energy.items():
            print(f"Charge {q}:")
            for microstate, energy in macrostate:
                print(f"  {microstate}: {energy:.4f}")

    # Process each pH value and store results
    results = {}
    for ph in args.ph:  # args.ph is already a list of floats from parse_ph_values
        if not args.quiet:
            print(f"\nCalculating distribution at pH={ph}...")
        result = process_single_ph(ph, drug_ensemble_free_energy, smi_drug)
        results[ph] = result

        if args.quiet:
            # Quiet mode: only print SMILES
            print(result['max_microstate'])
        else:
            # Verbose mode: print detailed results
            print(f"\nResults for pH={ph}:")
            print(f"Most probable microstate:")
            print(f"SMILES: {result['max_microstate']}")
            print(f"Charge: {result['max_charge']}")
            print(f"Probability: {result['probability']:.4f}")

            # Print full distribution
            print("\nFull distribution:")
            for q, macrostate in result['full_distribution'].items():
                print(f"Charge {q}:")
                for microstate, fraction in macrostate:
                    print(f"  {microstate}: {fraction:.4f}")

    logger.info("protonify tool finished successfully.")
    return results

if __name__ == "__main__":
    main()
