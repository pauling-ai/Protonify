from protonify.FreeEnergyPredictor import FreeEnergyPredictor
from rdkit.Chem import Mol, RWMol, AddHs, RemoveHs, SanitizeMol, MolToSmiles, MolFromSmiles, CanonSmiles, MolFromSmarts
from rdkit.Chem.MolStandardize.rdMolStandardize import TautomerEnumerator, TautomerEnumeratorStatus
from rdkit.Chem.Draw import MolToImage, ReactionToImage
from rdkit.Chem.rdChemReactions import ReactionFromSmarts
from PIL import Image
import math
import warnings
from typing import Dict, List, Optional, Tuple, Union, Callable, Literal
from collections import OrderedDict, defaultdict
import pandas as pd
import numpy as np
try:
    from IPython.display import display
except ImportError:
    # Fallback for non-notebook environments
    def display(obj):
        pass
from rdkit.Chem import RemoveHs

LN10 = math.log(10)
TRANSLATE_PH = 6.504894871171601

# Default cap on tautomers generated per enumeration call, to avoid combinatorial explosion
DEFAULT_MAX_TAUTOMERS = 50

# Reasons a tautomer enumeration can be cut short
TRUNCATED_MAX_TAUTOMERS = "max_tautomers"    # the max_tautomers cap was exceeded
TRUNCATED_MAX_TRANSFORMS = "max_transforms"  # RDKit's internal transform limit was hit


class TautomerLimitWarning(UserWarning):
    """Emitted when tautomer enumeration stops early because a limit was reached.

    A truncated enumeration is otherwise indistinguishable from a complete one,
    so microstates beyond the limit could silently be missing from the ensemble.
    """


# Tautomer enumerator instances, one per cap (reused for efficiency)
_tautomer_enumerators: Dict[int, TautomerEnumerator] = {}


def _get_tautomer_enumerator(cap: int) -> TautomerEnumerator:
    enumerator = _tautomer_enumerators.get(cap)
    if enumerator is None:
        enumerator = TautomerEnumerator()
        enumerator.SetMaxTautomers(cap)
        _tautomer_enumerators[cap] = enumerator
    return enumerator


def _check_max_tautomers(max_tautomers: int) -> int:
    if isinstance(max_tautomers, bool) or not isinstance(max_tautomers, (int, np.integer)) or max_tautomers < 1:
        raise ValueError(f"max_tautomers must be a positive integer, got {max_tautomers!r}")
    return int(max_tautomers)


def _enumerate_tautomers_checked(mol: Mol, max_tautomers: int) -> Tuple[List[Mol], Optional[str]]:
    """Enumerate up to ``max_tautomers`` tautomers of ``mol``.

    Returns ``(tautomers, reason)`` where ``reason`` is ``None`` if the enumeration
    is complete, otherwise why it was cut short (``TRUNCATED_MAX_TAUTOMERS``,
    ``TRUNCATED_MAX_TRANSFORMS`` or the name of another RDKit status).
    ``max_tautomers`` must already be validated.
    """
    if mol is None:
        return [], None
    try:
        # Ask RDKit for one more than the cap: it reports MaxTautomersReached as soon
        # as the count equals its limit, even when no further tautomer exists, so only
        # an (N+1)-th tautomer proves that the enumeration was actually truncated.
        result = _get_tautomer_enumerator(max_tautomers + 1).Enumerate(mol)
        tautomers = list(result)
        status = result.status
    except Exception:
        # If enumeration fails, return the original molecule
        return [mol], None
    if len(tautomers) > max_tautomers:
        return tautomers[:max_tautomers], TRUNCATED_MAX_TAUTOMERS
    if status == TautomerEnumeratorStatus.MaxTransformsReached:
        reason = TRUNCATED_MAX_TRANSFORMS
    elif status != TautomerEnumeratorStatus.Completed:
        reason = str(status.name)
    else:
        reason = None
    return (tautomers if tautomers else [mol]), reason


def describe_tautomer_truncations(truncations: Dict[str, str], max_tautomers: int, option: str = "max_tautomers") -> str:
    """Human-readable summary of ``{smiles: reason}`` truncations, with advice per reason."""
    by_reason: Dict[str, int] = defaultdict(int)
    for reason in truncations.values():
        by_reason[reason] += 1
    parts = []
    for reason, n in sorted(by_reason.items()):
        if reason == TRUNCATED_MAX_TAUTOMERS:
            parts.append(f"{n} molecule(s) had more than {max_tautomers} tautomers (increase {option})")
        elif reason == TRUNCATED_MAX_TRANSFORMS:
            parts.append(f"{n} molecule(s) hit RDKit's internal tautomer transform limit "
                         f"(not controlled by {option})")
        else:
            parts.append(f"{n} molecule(s) stopped with RDKit status {reason}")
    return "Tautomer enumeration was truncated: " + "; ".join(parts) + ". The ensemble may be incomplete."


def enumerate_tautomers(mol: Mol, max_tautomers: int = DEFAULT_MAX_TAUTOMERS) -> List[Mol]:
    """
    Enumerate tautomers for a given molecule using RDKit's TautomerEnumerator.

    Args:
        mol: RDKit Mol object
        max_tautomers: Maximum number of tautomers to generate

    Returns:
        List of tautomer Mol objects

    Warns:
        TautomerLimitWarning: if the enumeration was truncated (more than
            ``max_tautomers`` tautomers exist, or an RDKit limit was hit).
    """
    max_tautomers = _check_max_tautomers(max_tautomers)
    tautomers, reason = _enumerate_tautomers_checked(mol, max_tautomers)
    if reason is not None:
        warnings.warn(
            describe_tautomer_truncations({MolToSmiles(mol): reason}, max_tautomers),
            TautomerLimitWarning,
            stacklevel=2,
        )
    return tautomers


def _warn_truncations(truncations: Dict[str, str], max_tautomers: int, stacklevel: int = 3) -> None:
    if truncations:
        warnings.warn(describe_tautomer_truncations(truncations, max_tautomers),
                      TautomerLimitWarning, stacklevel=stacklevel)

#Predictor setup example
# model_path = "t_dwar_v_novartis_a_b.pt"
# predictor = FreeEnergyPredictor(model_path)


def prot(mol: Mol, idx: int, mode: str) -> Mol:
    '''
    Protonate / Deprotonate a molecule at a specified site
    '''
    mw = RWMol(mol)
    if mode == "a2b":
        atom_H = mw.GetAtomWithIdx(idx)
        if atom_H.GetAtomicNum() == 1:
            atom_A = atom_H.GetNeighbors()[0]
            charge_A = atom_A.GetFormalCharge()
            atom_A.SetFormalCharge(charge_A - 1)
            mw.RemoveAtom(idx)
            mol_prot = mw.GetMol()
        else:
            charge_H = atom_H.GetFormalCharge()
            numH_H = atom_H.GetTotalNumHs()
            atom_H.SetFormalCharge(charge_H - 1)
            atom_H.SetNumExplicitHs(numH_H - 1)
            atom_H.UpdatePropertyCache()
            mol_prot = AddHs(mw)
    elif mode == "b2a":
        atom_B = mw.GetAtomWithIdx(idx)
        charge_B = atom_B.GetFormalCharge()
        atom_B.SetFormalCharge(charge_B + 1)
        numH_B = atom_B.GetNumExplicitHs()
        atom_B.SetNumExplicitHs(numH_B + 1)
        mol_prot = AddHs(mw)
    SanitizeMol(mol_prot)
    mol_prot = MolFromSmiles(MolToSmiles(mol_prot, canonical=False))
    mol_prot = AddHs(mol_prot)
    return mol_prot

def draw(mol: Mol, size=(300, 300), highlightAtoms=[]) -> Image:
    for atom in mol.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return MolToImage(mol, size=size, highlightAtoms=highlightAtoms, highlightColor=(0.8,0.8,0.8))

def predict_micro_pKa(predictor: FreeEnergyPredictor, smi: str, idx: int, mode: Literal["a2b", "b2a"]):
    mol = MolFromSmiles(smi)
    new_mol = RemoveHs(prot(mol, idx, mode))
    new_smi = MolToSmiles(new_mol)
    if mode == "a2b":
        smi_A = smi
        smi_B = new_smi
    elif mode == "b2a":
        smi_B = smi
        smi_A = new_smi
    DfGm = predictor.predict([smi_A, smi_B])
    pKa = (DfGm[smi_B] - DfGm[smi_A]) / LN10 + TRANSLATE_PH
    pKa_image = ReactionToImage(ReactionFromSmarts(f"{smi_A}>>{smi_B}", useSmiles=True))
    display(pKa_image)
    return pKa

FILTER_PATTERNS = list(map(MolFromSmarts, [
    "[#6X5]",
    "[#7X5]",
    "[#8X4]",
    "[*r]=[*r]=[*r]",
    "[#1]-[*+1]~[*-1]",
    "[#1]-[*+1]=,:[*]-,:[*-1]",
    "[#1]-[*+1]-,:[*]=,:[*-1]",
    "[*+2]",
    "[*-2]",
    "[#1]-[#8+1].[#8-1,#7-1,#6-1]",
    "[#1]-[#7+1,#8+1].[#7-1,#6-1]",
    "[#1]-[#8+1].[#8-1,#6-1]",
    "[#1]-[#7+1].[#8-1]-[C](-[C,#1])(-[C,#1])",
    # "[#6;!$([#6]-,:[*]=,:[*]);!$([#6]-,:[#7,#8,#16])]=[C](-[O,N,S]-[#1])",
    # "[#6]-,=[C](-[O,N,S])(-[O,N,S]-[#1])",
    "[OX1]=[C]-[OH2+1]",
    "[NX1,NX2H1,NX3H2]=[C]-[O]-[H]",
    "[#6-1]=[*]-[*]",
    "[cX2-1]",
    "[N+1](=O)-[O]-[H]"
]))

def sanitize_checker(smi: str, filter_patterns: List[Mol], verbose: bool=False) -> bool:
    mol = AddHs(MolFromSmiles(smi))
    for pattern in filter_patterns:
        match = mol.GetSubstructMatches(pattern)
        if match:
            if verbose:
                print(f"pattern {pattern}")
            return False
    try:
        SanitizeMol(mol)
    except:
        print("cannot sanitize")
        return False
    return True

def sanitize_filter(smis: List[str], filter_patterns: List[Mol]=FILTER_PATTERNS) -> List[str]:
    def _checker(smi):
        return sanitize_checker(smi, filter_patterns)
    return list(filter(_checker, smis))

def cnt_stereo_atom(smi: str) -> int:
    mol = MolFromSmiles(smi)
    return sum([str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()])

def stereo_filter(smis: List[str]) -> List[str]:
    filtered_smi_dict = dict()
    for smi in smis:
        nonstereo_smi = CanonSmiles(smi, useChiral=0)
        stereo_cnt = cnt_stereo_atom(smi)
        if nonstereo_smi not in filtered_smi_dict:
            filtered_smi_dict[nonstereo_smi] = (smi, stereo_cnt)
        else:
            if stereo_cnt > filtered_smi_dict[nonstereo_smi][1]:
                filtered_smi_dict[nonstereo_smi] = (smi, stereo_cnt)
    return [value[0] for value in filtered_smi_dict.values()]

def make_filter(filter_param: OrderedDict) -> Callable:
    def seq_filter(smis):
        for single_filter, param in filter_param.items():
            smis = single_filter(smis, **param)
        return smis
    return seq_filter

def match_template(template: pd.DataFrame, mol: Mol, verbose: bool=False) -> list:
    mol = AddHs(mol)
    matches = []
    for idx, name, smarts, index, acid_base in template.itertuples():
        pattern = MolFromSmarts(smarts)
        match = mol.GetSubstructMatches(pattern)
        if len(match) == 0:
            continue
        else:
            index = int(index)
            for m in match:
                matches.append(m[index])
                if verbose:
                    print(f"find index {m[index]} in pattern {name} smarts {smarts}")
    return list(set(matches))

def prot_template(template: pd.DataFrame, smi: str, mode: str) -> Tuple[List[int], List[str]]:
    mol = MolFromSmiles(smi)
    sites = match_template(template, mol)
    smis = []
    for site in sites:
        smis.append(CanonSmiles(MolToSmiles(RemoveHs(prot(mol, site, mode)))))
    return sites, list(set(smis))

def enumerate_template(smi: Union[str, List[str]], template_a2b: pd.DataFrame, template_b2a: pd.DataFrame, mode: str="A", maxiter: int=10, verbose: int=0, filter_patterns: List[Mol]=FILTER_PATTERNS, skip_tautomers: bool=False, max_tautomers: int=DEFAULT_MAX_TAUTOMERS, truncations: Optional[Dict[str, str]]=None) -> Tuple[List[str], List[str]]:
    """Enumerate protonation states (and, unless skip_tautomers, tautomers).

    ``max_tautomers`` caps each individual tautomer enumeration. If ``truncations``
    is given, it is filled with ``{canonical SMILES: reason}`` for every molecule whose
    tautomer enumeration was cut short; otherwise a single TautomerLimitWarning is
    emitted if any was.
    """
    if isinstance(smi, str):
        smis = [smi]
    else:
        smis = list(smi)
    report = truncations if truncations is not None else {}
    if skip_tautomers:
        enum_func = lambda x: [x]
    else:
        max_tautomers = _check_max_tautomers(max_tautomers)

        def enum_func(mol):
            tautomers, reason = _enumerate_tautomers_checked(mol, max_tautomers)
            if reason is not None:
                report[CanonSmiles(MolToSmiles(mol))] = reason
            return tautomers
    if mode == "a2b":
        smis_A_pool, smis_B_pool = smis, []
    elif mode == "b2a":
        smis_A_pool, smis_B_pool = [], smis
    filters = make_filter({
        sanitize_filter: {"filter_patterns": filter_patterns},
        stereo_filter: {}
    })
    pool_length_A = -1
    pool_length_B = -1
    i = 0
    while (len(smis_A_pool) != pool_length_A or len(smis_B_pool) != pool_length_B) and i < maxiter:
        pool_length_A, pool_length_B = len(smis_A_pool), len(smis_B_pool)
        if verbose > 0:
            print(f"iter {i}: {pool_length_A} acid, {pool_length_B} base")
        if verbose > 1:
            print(f"iter {i}, acid: {smis_A_pool}, base: {smis_B_pool}")
        if (mode == "a2b" and (i + 1) % 2) or (mode == "b2a" and i % 2):
            smis_A_tmp_pool = []
            for smi in smis_A_pool:
                smis_B_pool += filters(prot_template(template_a2b, smi, "a2b")[1])
                smis_A_tmp_pool += filters([CanonSmiles(MolToSmiles(mol)) for mol in enum_func(MolFromSmiles(smi))])
            smis_A_pool += smis_A_tmp_pool
        elif (mode == "b2a" and (i + 1) % 2) or (mode == "a2b" and i % 2):
            smis_B_tmp_pool = []
            for smi in smis_B_pool:
                smis_A_pool += filters(prot_template(template_b2a, smi, "b2a")[1])
                smis_B_tmp_pool += filters([CanonSmiles(MolToSmiles(mol)) for mol in enum_func(MolFromSmiles(smi))])
            smis_B_pool += smis_B_tmp_pool
        smis_A_pool = filters(smis_A_pool)
        smis_B_pool = filters(smis_B_pool)
        smis_A_pool = list(set(smis_A_pool))
        smis_B_pool = list(set(smis_B_pool))
        i += 1
    if verbose > 0:
            print(f"iter {i}: {pool_length_A} acid, {pool_length_B} base")
    if verbose > 1:
        print(f"iter {i}, acid: {smis_A_pool}, base: {smis_B_pool}")
    smis_A_pool = list(map(CanonSmiles, smis_A_pool))
    smis_B_pool = list(map(CanonSmiles, smis_B_pool))
    if truncations is None:
        _warn_truncations(report, max_tautomers)
    return smis_A_pool, smis_B_pool 

def log_sum_exp(DfGm: list[float]) -> float:
    return math.log10(sum([math.exp(-g) for g in DfGm]))

from rdkit.Chem.Draw import MolsToGridImage

def draw_macrostate(macrostate: List[str], base_name: str):
    macrostate_mols = list(map(MolFromSmiles, macrostate))
    macrostate_size = len(macrostate_mols)
    legends = [f"{i+1}-{base_name}" for i in range(macrostate_size)]
    display(MolsToGridImage(macrostate_mols, legends=legends, useSVG=True))

def predict_macro_pKa(smi: str, template_a2b: pd.DataFrame, template_b2a: pd.DataFrame, mode: Literal["a2b", "b2a"], predictor) -> float:
    macrostate_A, macrostate_B = enumerate_template(smi, template_a2b, template_b2a, mode)
    DfGm_A = predictor.predict(macrostate_A)
    DfGm_B = predictor.predict(macrostate_B)
    draw_macrostate(macrostate_A, "A")
    draw_macrostate(macrostate_B, "B")
    return log_sum_exp(DfGm_A.values()) - log_sum_exp(DfGm_B.values()) + TRANSLATE_PH

def calc_base_name(neutral_base_name: str, target_charge: int) -> str:
    if neutral_base_name.startswith("H"):
        if neutral_base_name[1:].startswith("<sub>"):
            num_H_end = neutral_base_name.find("</sub>", 6)
            num_H = int(neutral_base_name[6:num_H_end])
        else:
            num_H_end = 1
            num_H = 1
    else:
        num_H_end = 0
        num_H = 0
    target_num_H = num_H + target_charge
    assert target_num_H >= 0
    target_base_name = ""
    if target_num_H == 1:
        target_base_name += "H"
    elif target_num_H > 1:
        target_base_name += f"H<sub>{target_num_H}</sub>"
    target_base_name += "A"
    if target_charge < -1:
        target_base_name += f"<sup>{-target_charge}-</sup>"
    elif target_charge == -1:
        target_base_name += f"<sup>-</sup>"
    elif target_charge == 1:
        target_base_name += f"<sup>+</sup>"
    elif target_charge > 1:
        target_base_name += f"<sup>{target_charge}+</sup>"
    return target_base_name

from rdkit.Chem import GetFormalCharge
from typing import Dict

def get_ensemble(smi: str, template_a2b: pd.DataFrame, template_b2a: pd.DataFrame, maxiter: int=10, skip_tautomers: bool=False, max_tautomers: int=DEFAULT_MAX_TAUTOMERS, truncations: Optional[Dict[str, str]]=None) -> Dict[int, list[str]]:
    """Enumerate all protonation (and tautomeric) microstates, grouped by formal charge.

    See enumerate_template for ``max_tautomers`` and ``truncations``.
    """
    report = truncations if truncations is not None else {}
    ensemble = dict()
    q0 = GetFormalCharge(MolFromSmiles(smi))
    ensemble[q0] = [smi]
    smis_0 = [smi]
    smis_0, smis_b1 = enumerate_template(smis_0, template_a2b, template_b2a, maxiter=maxiter, mode="a2b", skip_tautomers=skip_tautomers, max_tautomers=max_tautomers, truncations=report)
    if smis_b1:
        ensemble[q0 - 1] = smis_b1
    q = q0 - 2
    while True:
        if q + 1 in ensemble:
            _, smis_b = enumerate_template(ensemble[q + 1], template_a2b, template_b2a, maxiter=maxiter, mode="a2b", skip_tautomers=skip_tautomers, max_tautomers=max_tautomers, truncations=report)
            if smis_b:
                ensemble[q] = smis_b
            else:
                break
        q -= 1
    smis_a1, smis_0 = enumerate_template(smis_0, template_a2b, template_b2a, maxiter=maxiter, mode="b2a", skip_tautomers=skip_tautomers, max_tautomers=max_tautomers, truncations=report)
    if smis_a1:
        ensemble[q0 + 1] = smis_a1
    q = q0 + 2
    while True:
        if q - 1 in ensemble:
            smis_a, _ = enumerate_template(ensemble[q - 1], template_a2b, template_b2a, maxiter=maxiter, mode="b2a", skip_tautomers=skip_tautomers, max_tautomers=max_tautomers, truncations=report)
            if smis_a:
                ensemble[q] = smis_a
            else:
                break
        q += 1
    ensemble[q0] = smis_0
    if truncations is None:
        _warn_truncations(report, max_tautomers)
    return ensemble

def get_neutral_base_name(ensemble: Dict[int, list[str]]) -> str:
    q_list = sorted(ensemble.keys())
    min_q = -int(min(q_list))
    return "A" if min_q == 0 else f"H<sub>{min_q}</sub>A"

def draw_ensemble(ensemble: Dict[int, list[str]]) -> None:
    q_list = sorted(ensemble.keys())
    neutral_base_name = get_neutral_base_name(ensemble)
    for q in q_list:
        draw_macrostate(ensemble[q], calc_base_name(neutral_base_name, q))

from collections import defaultdict
import pylab as pl

def predict_ensemble_free_energy(smi: str, template_a2b: pd.DataFrame, template_b2a: pd.DataFrame, predictor, skip_tautomers: bool=False, max_tautomers: int=DEFAULT_MAX_TAUTOMERS, truncations: Optional[Dict[str, str]]=None) -> Dict[int, list[tuple[str, float]]]:
    """Predict the free energy (DfGm, in units of RT) of every microstate, grouped by charge.

    See enumerate_template for ``max_tautomers`` and ``truncations``.
    """
    report = truncations if truncations is not None else {}
    ensemble = get_ensemble(smi, template_a2b, template_b2a, skip_tautomers=skip_tautomers, max_tautomers=max_tautomers, truncations=report)
    if truncations is None:
        _warn_truncations(report, max_tautomers)
    ensemble_free_energy = dict()
    for q, macrostate in ensemble.items():
        prediction = predictor.predict(macrostate)
        ensemble_free_energy[q] = [(microstate, prediction[microstate]) for microstate in macrostate]
    return ensemble_free_energy

def calc_distribution(ensemble_free_energy: Dict[int, list[tuple[str, float]]], pH: float) -> Dict[int, list[tuple[str, float]]]:
    ensemble_boltzmann_factor = defaultdict(list)
    partition_function = 0
    for q, macrostate_free_energy in ensemble_free_energy.items():
        for microstate, DfGm in macrostate_free_energy:
            boltzmann_factor = math.exp(-DfGm - q * LN10 * (pH - TRANSLATE_PH))
            partition_function += boltzmann_factor
            ensemble_boltzmann_factor[q].append((microstate, boltzmann_factor))
    return {
        q: [(microstate, boltzmann_factor / partition_function) for microstate, boltzmann_factor in macrostate_boltzmann_factor] 
        for q, macrostate_boltzmann_factor in ensemble_boltzmann_factor.items()
    }

def draw_distribution_pH(ensemble_free_energy: Dict[int, list[tuple[str, float]]]) -> None:
    pHs = np.linspace(0, 14, 1000)
    fractions = defaultdict(list)
    name_mapping = dict()
    ensemble = defaultdict(list)
    neutral_base_name = get_neutral_base_name(ensemble_free_energy)
    for q, macrostate in ensemble_free_energy.items():
        for i, (microstate, _) in enumerate(macrostate):
            name_mapping[microstate] = f"{i+1}-{calc_base_name(neutral_base_name, q)}"
            ensemble[q].append(microstate)
    for pH in pHs:
        distribution = calc_distribution(ensemble_free_energy, pH)
        for q, macrostate_fraction in distribution.items():
            for microstate, fraction in macrostate_fraction:
                fractions[name_mapping[microstate]].append(fraction)
    pl.figure(figsize=(14, 3), dpi=200)
    for base_name, fraction_curve in fractions.items():
        pl.plot(pHs, fraction_curve, label=base_name.replace("<sub>", "$_{").replace("</sub>", "}$").replace("<sup>", "$^{").replace("</sup>", "}$"))
    draw_ensemble(ensemble)
    pl.xlabel("pH")
    pl.ylabel("fraction")
    pl.legend()
    pl.show()

def read_template(template_file: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    template = pd.read_csv(template_file, sep="\t")
    template_a2b = template[template.Acid_or_base == "A"]
    template_b2a = template[template.Acid_or_base == "B"]
    return template_a2b, template_b2a

def predict_macro_pKa_from_macrostate(macrostate_A, macrostate_B, A_name, B_name, predictor) -> float:
    DfGm_A = predictor.predict(macrostate_A)
    DfGm_B = predictor.predict(macrostate_B)
    draw_macrostate(macrostate_A, A_name)
    draw_macrostate(macrostate_B, B_name)
    return log_sum_exp(DfGm_A.values()) - log_sum_exp(DfGm_B.values()) + TRANSLATE_PH 