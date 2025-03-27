import numpy as np
import torch
import uproot

import features_list

# Dizionari per le caratteristiche (ripresi da pythia_generator.py)
features_list = ["id", "status", "px", "py", "pz", "e", "m"]
features_23 = [f"{feature}_23" for feature in features_list if feature != "status"]
features_final = [f"{feature}_final" for feature in features_list if feature != "status"]

def root_to_tensors(file_path, max_seq_len=None, normalize=True):
    """
    Legge un file ROOT e converte i dati in tensori Torch.

    Args:
        file_path (str): Percorso del file ROOT.
        max_seq_len (int, opzionale): Numero massimo di particelle per evento.
        normalize (bool, opzionale): Normalizza le caratteristiche di input.

    Returns:
        tuple: (source_tensor, target_tensor)
            source_tensor (torch.Tensor): Tensore delle particelle status 23,
                                          shape (batch_size, seq_len, feature_dim).
            target_tensor (torch.Tensor): Tensore delle particelle finali,
                                          stessa shape.
    """
    with uproot.open(file_path) as file:
        tree = file["ParticleTree"]
        
        # Caricamento diretto degli array da uproot
        data_23 = tree.arrays(features_23, library="np")
        data_final = tree.arrays(features_final, library="np")
        
        # Conversione in liste di eventi (senza usare range)
        events_23 = [np.column_stack([data_23[feat] for feat in features_23]) for _ in data_23[features_23[0]]]
        events_final = [np.column_stack([data_final[feat] for feat in features_final]) for _ in data_final[features_final[0]]]

        # Normalizzazione
        if normalize:
            def normalize_features(events):
                all_data = np.vstack(events)
                means = all_data.mean(axis=0)
                stds = all_data.std(axis=0)
                return [(event - means) / stds for event in events]
            
            events_23 = normalize_features(events_23)
            events_final = normalize_features(events_final)

        # Padding e troncamento
        def pad_and_truncate(events, max_seq_len, feature_dim):
            padded = np.zeros((len(events), max_seq_len, feature_dim))
            for i, event in enumerate(events):
                seq_len = min(len(event), max_seq_len)
                padded[i, :seq_len, :] = event[:seq_len]
            return padded
        
        if max_seq_len is None:
            max_seq_len = max(len(event) for event in events_23 + events_final)
        
        feature_dim = len(features_23)
        tensor_23 = pad_and_truncate(events_23, max_seq_len, feature_dim)
        tensor_final = pad_and_truncate(events_final, max_seq_len, feature_dim)
        
        # Conversione in Torch Tensor
        source_tensor = torch.tensor(tensor_23, dtype=torch.float32)
        target_tensor = torch.tensor(tensor_final, dtype=torch.float32)
        
        return source_tensor, target_tensor

# Esempio di utilizzo
source, target = root_to_tensors("events.root", max_seq_len=100)
print("Source shape:", source.shape)
print("Target shape:", target.shape)


