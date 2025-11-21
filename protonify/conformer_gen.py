import numpy as np
from scipy.spatial import distance_matrix
from rdkit import Chem
from rdkit.Chem import AllChem
import logging
from typing import List, Tuple, Any
from protonify.chem_dicts import DICT, DICT_CHARGE
from protonify.dictionary import Dictionary

logger = logging.getLogger("unimol_free_energy.inference")

class InvalidSmilesError(Exception):
    """Custom exception for invalid SMILES strings."""

class ConformerGen(object):
    '''
    This class designed to generate conformers for molecules represented as SMILES strings using provided parameters and configurations.
    Uses RDKit for conformer generation.
    '''
    def __init__(self, **params):
        """
        Initializes the neural network model based on the provided model name and parameters.

        :param model_name: (str) The name of the model to initialize.
        :param params: Additional parameters for model configuration including:
            - seed: Random seed for reproducibility (default: 42)
            - max_atoms: Maximum number of atoms (default: 256)
            - mode: 'fast' or 'heavy' for RDKit
            - remove_hs: Whether to remove hydrogen atoms (default: False)
            - 3d_generation_timeout: Timeout in seconds for 3D generation

        :return: An instance of the specified neural network model.
        """
        self._init_features(**params)

    def _init_features(self, **params):
        """
        Initializes the features of the ConformerGen object based on provided parameters.

        :param params: Arbitrary keyword arguments for feature configuration.
                      These can include the random seed, maximum number of atoms, data type,
                      generation mode, and whether to remove hydrogens.
        """
        self.seed = params.get('seed', 42)
        self.max_atoms = params.get('max_atoms', 256)
        self.data_type = params.get('data_type', 'molecule')
        self.mode = params.get('mode', 'fast')
        self.remove_hs = params.get('remove_hs', False)
        self.dict_dir = params.get('dict_dir', 'dict')
        self.dictionary = Dictionary.load_from_str(DICT)
        self.dictionary.add_symbol("[MASK]", is_special=True)
        self.charge_dictionary = Dictionary.load_from_str(DICT_CHARGE)
        self.charge_dictionary.add_symbol("[MASK]", is_special=True)

    def single_process(self, smiles):
        """
        Processes a single SMILES string to generate conformers using RDKit.

        :param smiles: (str) The SMILES string representing the molecule.
        :return: A unimolecular data representation (dictionary) of the molecule.
        """
        atoms, coordinates, charges = inner_smi2coords(
            smiles,
            seed=self.seed,
            mode=self.mode,
            remove_hs=self.remove_hs
        )

        return coords2unimol(atoms, coordinates, charges, self.dictionary, self.charge_dictionary,
                            self.max_atoms, remove_hs=self.remove_hs)

    def transform_raw(self, atoms_list, coordinates_list, charges_list):

        inputs = []
        for atoms, coordinates, charges in zip(atoms_list, coordinates_list, charges_list):
            inputs.append(coords2unimol(atoms, coordinates, charges, self.dictionary, self.charge_dictionary, self.max_atoms, remove_hs=self.remove_hs))
        return inputs

    def transform(self, smiles_list):
        logger.info('Start generating conformers...')
        inputs = [self.single_process(item) for item in smiles_list]
        failed_cnt = np.mean([(item['src_coord']==0.0).all() for item in inputs])
        logger.info('Succeed to generate conformers for {:.2f}% of molecules.'.format((1-failed_cnt)*100))
        failed_3d_cnt = np.mean([(item['src_coord'][:,2]==0.0).all() for item in inputs])
        logger.info('Succeed to generate 3d conformers for {:.2f}% of molecules.'.format((1-failed_3d_cnt)*100))
        return inputs


def inner_smi2coords(smi, seed=42, mode='fast', remove_hs=True):
    '''
    Convert SMILES string into 3D coordinates using RDKit.

    Args:
        smi (str): SMILES string of the molecule
        seed (int): Random seed for conformer generation
        mode (str): 'fast' or 'heavy' generation mode
        remove_hs (bool): Whether to remove hydrogens

    Returns:
        tuple: (atoms, coordinates, charges)
    '''
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        logger.error(f"Invalid SMILES string: {smi}")
        raise InvalidSmilesError(f"Invalid SMILES string: {smi}")

    mol = AllChem.AddHs(mol)
    atoms, charges = [], []
    for atom in mol.GetAtoms():
        atoms.append(atom.GetSymbol())
        charges.append(atom.GetFormalCharge())

    assert len(atoms)>0, 'No atoms in molecule: {}'.format(smi)

    # Generate 3D coordinates with RDKit
    res = AllChem.EmbedMolecule(mol, randomSeed=seed)
    if res == 0:
        try:
            AllChem.MMFFOptimizeMolecule(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        except:
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    elif res == -1 and mode == 'heavy':
        AllChem.EmbedMolecule(mol, maxAttempts=5000, randomSeed=seed)
        try:
            AllChem.MMFFOptimizeMolecule(mol)
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
        except:
            coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    else:
        AllChem.Compute2DCoords(mol)
        coordinates = mol.GetConformer().GetPositions().astype(np.float32)

    assert len(atoms) == len(coordinates), "coordinates shape is not align with {}".format(smi)

    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        charges_no_h = [charges[i] for i in idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with {}".format(smi)
        return atoms_no_h, coordinates_no_h, charges_no_h
    else:
        return atoms, coordinates, charges

def inner_coords(atoms, coordinates, charges, remove_hs=True):
    """
    Processes a list of atoms and their corresponding coordinates to remove hydrogen atoms if specified.
    This function takes a list of atom symbols and their corresponding coordinates and optionally removes hydrogen atoms from the output. It includes assertions to ensure the integrity of the data and uses numpy for efficient processing of the coordinates. 

    :param atoms: (list) A list of atom symbols (e.g., ['C', 'H', 'O']).
    :param coordinates: (list of tuples or list of lists) Coordinates corresponding to each atom in the `atoms` list.
    :param remove_hs: (bool, optional) A flag to indicate whether hydrogen atoms should be removed from the output.
                      Defaults to True.
    
    :return: A tuple containing two elements; the filtered list of atom symbols and their corresponding coordinates.
             If `remove_hs` is False, the original lists are returned.
    
    :raises AssertionError: If the length of `atoms` list does not match the length of `coordinates` list.
    """
    assert len(atoms) == len(coordinates), "coordinates shape is not align atoms"
    coordinates = np.array(coordinates).astype(np.float32)
    if remove_hs:
        idx = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms_no_h = [atom for atom in atoms if atom != 'H']
        coordinates_no_h = coordinates[idx]
        charges_no_h = [charges[i] for i in idx]
        assert len(atoms_no_h) == len(coordinates_no_h), "coordinates shape is not align with atoms"
        return atoms_no_h, coordinates_no_h, charges_no_h
    else:
        return atoms, coordinates, charges

def coords2unimol(atoms, coordinates, charges, dictionary, charge_dictionary, max_atoms=256, remove_hs=True, **params):
    """
    Converts atom symbols and coordinates into a unified molecular representation.

    :param atoms: (list) List of atom symbols.
    :param coordinates: (ndarray) Array of atomic coordinates.
    :param dictionary: (Dictionary) An object that maps atom symbols to unique integers.
    :param max_atoms: (int) The maximum number of atoms to consider for the molecule.
    :param remove_hs: (bool) Whether to remove hydrogen atoms from the representation.
    :param params: Additional parameters.

    :return: A dictionary containing the molecular representation with tokens, distances, coordinates, and edge types.
    """
    atoms, coordinates, charges = inner_coords(atoms, coordinates, charges, remove_hs=remove_hs)
    atoms = np.array(atoms)
    coordinates = np.array(coordinates).astype(np.float32)
    charges = np.array(charges).astype(str)
    # cropping atoms and coordinates
    if len(atoms) > max_atoms:
        idx = np.random.choice(len(atoms), max_atoms, replace=False)
        atoms = atoms[idx]
        coordinates = coordinates[idx]
        charges = charges[idx]
    # tokens padding
    src_tokens = np.array([dictionary.bos()] + [dictionary.index(atom) for atom in atoms] + [dictionary.eos()])
    src_distance = np.zeros((len(src_tokens), len(src_tokens)))
    src_charges = np.array([charge_dictionary.bos()] + [charge_dictionary.index(charge) for charge in charges] + [charge_dictionary.eos()])
    # coordinates normalize & padding
    src_coord = coordinates - coordinates.mean(axis=0)
    src_coord = np.concatenate([np.zeros((1,3)), src_coord, np.zeros((1,3))], axis=0)
    # distance matrix
    src_distance = distance_matrix(src_coord, src_coord)
    # edge type
    src_edge_type = src_tokens.reshape(-1, 1) * len(dictionary) + src_tokens.reshape(1, -1)

    return {
            'src_tokens': src_tokens.astype(int), 
            'src_charges': src_charges.astype(int),
            'src_distance': src_distance.astype(np.float32), 
            'src_coord': src_coord.astype(np.float32), 
            'src_edge_type': src_edge_type.astype(int),
            }