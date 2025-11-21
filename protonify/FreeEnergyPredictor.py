import torch
from protonify.unimol_model import UniMolModel
from protonify.conformer_gen import ConformerGen
from protonify.mol_dataset import MolDataset
from torch.utils.data import DataLoader

class FreeEnergyPredictor(object):
    def __init__(self, model_path, batch_size=32, remove_hs=False, use_gpu=True):
        """
        Initialize the free energy predictor.

        Args:
            model_path: Path to the model checkpoint
            batch_size: Batch size for inference
            remove_hs: Whether to remove hydrogens
            use_gpu: Whether to use GPU
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and use_gpu else "cpu")
        self.model = UniMolModel(model_path, output_dim=1, remove_hs=remove_hs).to(self.device)
        self.model.eval()
        self.batch_size = batch_size
        self.remove_hs = remove_hs

        # Initialize conformer generator
        params = {
            'remove_hs': self.remove_hs
        }

        self.conformer_gen = ConformerGen(**params)

    def preprocess_data(self, smiles_list):
        # conf gen
        inputs = self.conformer_gen.transform(smiles_list)
        return inputs

    def predict(self, smiles_list):
        unimol_input = self.preprocess_data(smiles_list)
        dataset = MolDataset(unimol_input)
        dataloader = DataLoader(dataset, 
                                batch_size=self.batch_size, 
                                shuffle=False,
                                collate_fn=self.model.batch_collate_fn,
                                )

        results = {}
        for batch in dataloader:
            net_input, _ = self.decorate_torch_batch(batch)
            with torch.no_grad():
                predictions = self.model(**net_input)
                for smiles, energy in zip(smiles_list, predictions):
                    results[smiles] = energy.item()
        return results

    def predict_single(self, smiles):
        return self.predict([smiles])

    def decorate_torch_batch(self, batch):
        """
        Prepares a standard PyTorch batch of data for processing by the model. Handles tensor-based data structures.

        :param batch: The batch of tensor-based data to be processed.

        :return: A tuple of (net_input, net_target) for model processing.
        """
        net_input, net_target = batch
        if isinstance(net_input, dict):
            net_input, net_target = {
                k: v.to(self.device) for k, v in net_input.items()}, net_target.to(self.device)
        else:
            net_input, net_target = {'net_input': net_input.to(
                self.device)}, net_target.to(self.device)
        net_target = None

        return net_input, net_target
