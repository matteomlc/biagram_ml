"""
tokenizer.py — Tokenizzazione a livello carattere
===================================================

Converte testo <-> numeri. Ogni carattere unico e' un token.

Incapsuliamo tutto in una classe perche' il tokenizer ha uno "stato"
(il vocabolario e le mappe di conversione) che vogliamo tenere insieme.
"""

import torch


class CharTokenizer:
    """
    Tokenizer a livello carattere.

    Costruisce il vocabolario dai caratteri unici del testo e fornisce
    i metodi encode (testo -> numeri) e decode (numeri -> testo).
    """

    def __init__(self, text):
        # Trova tutti i caratteri unici e ordinali (per riproducibilita')
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)

        # Mappe di conversione
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}  # string -> int
        self.itos = {i: ch for i, ch in enumerate(self.chars)}  # int -> string

    def encode(self, s):
        """Converte una stringa in lista di interi. 'ciao' -> [3, 12, 1, 15]"""
        return [self.stoi[c] for c in s]

    def decode(self, tokens):
        """Converte una lista di interi in stringa. [3, 12, 1, 15] -> 'ciao'"""
        return "".join([self.itos[i] for i in tokens])

    def encode_to_tensor(self, s):
        """Come encode, ma restituisce direttamente un tensore PyTorch."""
        return torch.tensor(self.encode(s), dtype=torch.long)

    def summary(self):
        """Stampa un riepilogo del vocabolario."""
        print(f"Vocabolario: {self.vocab_size} caratteri unici")
        print(f"Caratteri: {''.join(self.chars)}")
