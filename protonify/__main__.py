import os
import sys
import logging
import warnings
import argparse
from importlib.resources import files
from rdkit import RDLogger
from protonify import get_model_path, ensure_model, download_model, MODEL_PATH_ENV_VAR, CACHE_DIR
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


def cmd_download_model(args):
    """Download the model to cache directory."""
    print(f"Protonify Model Downloader")
    print(f"=" * 40)

    # Show where model will be stored
    print(f"Cache directory: {CACHE_DIR}")

    try:
        model_path = download_model(force=args.force)
        print(f"\nModel downloaded successfully!")
        print(f"Model path: {model_path}")
        print(f"\nTo use this model, you can either:")
        print(f"  1. Do nothing - protonify will find it automatically")
        print(f"  2. Set environment variable:")
        print(f"     export {MODEL_PATH_ENV_VAR}={model_path}")
        return 0
    except Exception as e:
        print(f"\nError downloading model: {e}", file=sys.stderr)
        return 1


def cmd_show_model_path(args):
    """Show where the model is located."""
    print(f"Protonify Model Information")
    print(f"=" * 40)

    # Check environment variable
    env_path = os.environ.get(MODEL_PATH_ENV_VAR)
    if env_path:
        print(f"Environment variable ({MODEL_PATH_ENV_VAR}): {env_path}")
        if os.path.exists(env_path):
            print(f"  Status: EXISTS")
        else:
            print(f"  Status: NOT FOUND")
    else:
        print(f"Environment variable ({MODEL_PATH_ENV_VAR}): Not set")

    print(f"\nCache directory: {CACHE_DIR}")
    cached_model = CACHE_DIR / "t_dwar_v_novartis_a_b.pt"
    if cached_model.exists():
        print(f"Cached model: {cached_model}")
        print(f"  Status: EXISTS")
    else:
        print(f"Cached model: Not downloaded")

    # Try to resolve the model path
    try:
        resolved = get_model_path()
        print(f"\nResolved model path: {resolved}")
    except FileNotFoundError:
        print(f"\nNo model found. Run 'protonify download-model' to download.")

    return 0


def cmd_predict(args):
    """Run protonation prediction (main command)."""
    # Get model path (auto-downloads if needed) and set env var
    model_path = ensure_model(args.model_path)

    predictor = FreeEnergyPredictor(
        model_path,
        batch_size=32,
        remove_hs=False,
        use_gpu=not args.cpu
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


def main():
    parser = argparse.ArgumentParser(
        description="Protonify - Protonation state prediction for molecules at a given pH.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict protonation state
  protonify --smiles "CCO" --ph 7.4 --template smart

  # Download model first (recommended for first use)
  protonify download-model

  # Show model location
  protonify model-info
"""
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Download model subcommand
    download_parser = subparsers.add_parser(
        'download-model',
        help='Download the model to cache directory'
    )
    download_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download even if model exists'
    )
    download_parser.set_defaults(func=cmd_download_model)

    # Model info subcommand
    info_parser = subparsers.add_parser(
        'model-info',
        help='Show model location and status'
    )
    info_parser.set_defaults(func=cmd_show_model_path)

    # Main prediction arguments (backwards compatible - no subcommand needed)
    parser.add_argument('--template', choices=['simple', 'smart'], help='Template type to use (simple or smart)')
    parser.add_argument('--ph', type=parse_ph_values,
                       help='pH level(s) to calculate the distribution. Can be a single value or comma-separated list (e.g., "7.4" or "7.0,7.4,8.0")')
    parser.add_argument('--smiles', type=str, help='SMILES of the molecule to analyze')
    parser.add_argument('--model-path', type=str, default=None, help='Path to the pretrained model (defaults to auto-download)')
    parser.add_argument('--enumerate-tautomers', action='store_true', help='Enumerate tautomers (slower but more thorough). Default: skip tautomers for speed.')
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference (GPU is used by default when available)')
    parser.add_argument('--quiet', '-q', action='store_true', help='Suppress verbose output, only print final SMILES result')

    args = parser.parse_args()

    # Handle subcommands
    if args.command:
        return args.func(args)

    # Backwards compatibility: if no subcommand, expect prediction arguments
    if not args.smiles or not args.ph or not args.template:
        parser.print_help()
        print("\nError: --smiles, --ph, and --template are required for prediction.")
        print("Use 'protonify download-model' to download the model first.")
        return 1

    return cmd_predict(args)


if __name__ == "__main__":
    sys.exit(main() or 0)
