"""
dataset.py — Gestione dei dati
================================

Si occupa di: caricare il testo, fare lo split train/validation,
e fornire batch casuali per il training.

Tiene insieme i dati e la logica per pescarne dei pezzetti (get_batch).
"""

import torch


def load_text(path=None, fallback=None):
    """
    Carica il testo da un file, oppure usa un testo di fallback.

    Args:
        path: percorso a un file .txt (es. "divina_commedia.txt")
        fallback: testo da usare se path e' None o il file non esiste
    """
    if path is not None:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        except FileNotFoundError:
            print(f"File '{path}' non trovato, uso il testo di fallback.")
    return fallback

class Dataset:
    """
    Contiene i dati tokenizzati e li divide in train/validation.
    Fornisce get_batch per pescare batch casuali durante il training.
    """

    def __init__(self, data_tensor, train_split, block_size, batch_size, device):
        self.block_size = block_size
        self.batch_size = batch_size
        self.device = device

        # Split: primo train_split% per training, resto per validation
        n = int(train_split * len(data_tensor))
        self.train_data = data_tensor[:n]
        self.val_data = data_tensor[n:]

    def get_batch(self, split):
        """
        Pesca un batch casuale.

        Restituisce:
            x: (batch_size, block_size) — sequenze di input
            y: (batch_size, block_size) — target (input shiftato di 1)
        """
        data = self.train_data if split == "train" else self.val_data
        # Posizioni di partenza casuali
        ix = torch.randint(len(data) - self.block_size, (self.batch_size,))
        x = torch.stack([data[i:i + self.block_size] for i in ix])
        y = torch.stack([data[i + 1:i + self.block_size + 1] for i in ix])
        return x.to(self.device), y.to(self.device)

    def summary(self):
        print(f"Training:   {len(self.train_data)} caratteri")
        print(f"Validation: {len(self.val_data)} caratteri")
