import argparse
from typing import Dict, List, Tuple, Union

from protonify.pka_utils import calc_distribution

def process_single_ph(ph: float, drug_ensemble_free_energy: Dict, smiles: str) -> Dict:
    """Process a single pH value and return the results.
    
    Args:
        ph: pH value to calculate distribution
        drug_ensemble_free_energy: Pre-calculated ensemble free energy
        smiles: Original SMILES string (for reference)
        
    Returns:
        Dictionary containing the results for this pH value
    """
    distribution = calc_distribution(drug_ensemble_free_energy, ph)
    
    # Find the microstate with the highest probability
    max_fraction = -1
    max_microstate = None
    max_charge = None
    for q, macrostate in distribution.items():
        for microstate, fraction in macrostate:
            if fraction > max_fraction:
                max_fraction = fraction
                max_microstate = microstate
                max_charge = q
    
    return {
        'original_smiles': smiles,
        'ph': ph,
        'max_microstate': max_microstate,
        'max_charge': max_charge,
        'probability': max_fraction,
        'full_distribution': distribution
    }

def parse_ph_values(ph_str: str) -> List[float]:
    """Parse a comma-separated string of pH values into a list of floats.
    
    Args:
        ph_str: Comma-separated string of pH values
        
    Returns:
        List of float pH values
        
    Raises:
        argparse.ArgumentTypeError: If any value cannot be converted to float
    """
    try:
        return [float(ph.strip()) for ph in ph_str.split(',')]
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid pH value in list: {ph_str}. Each value must be a valid float.") 