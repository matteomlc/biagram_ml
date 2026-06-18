"""
model.py — Il modello
=======================

Contiene la definizione dell'architettura: il BigramLanguageModel.

Questo e' il file che modificherai quando passerai al Transformer:
basta sostituire/estendere questa classe, tutto il resto del progetto
(training, dati, tokenizer) resta invariato.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Il language model piu' semplice possibile.

    Architettura: una singola tabella di embedding (vocab_size x vocab_size).
    La riga di ogni token E' GIA' la previsione (i logit) del prossimo token.
    Guarda solo l'ultimo carattere — nessuna memoria del contesto piu' ampio.
    """

    def __init__(self, vocab_size):
        super().__init__()
        # L'unica componente: embedding che mappa ogni token ai logit
        # del prossimo token. Forma: (vocab_size, vocab_size)
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(self, idx, targets=None):
        """
        Forward pass.

        Args:
            idx:     (B, T) indici dei token di input
            targets: (B, T) indici dei token target (opzionale)

        Returns:
            logits: (B, T, C) punteggi sul prossimo token
            loss:   scalare se ci sono i targets, altrimenti None
        """
        # Embedding lookup: ogni token -> riga di logit
        logits = self.token_embedding_table(idx)  # (B, T, C)

        if targets is None:
            loss = None
        else:
            # cross_entropy vuole logits (N, C) e targets (N,)
            # quindi appiattiamo le prime due dimensioni
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Genera testo autoregressivamente, un token alla volta.

        Args:
            idx: (B, T) contesto iniziale
            max_new_tokens: quanti token generare

        Returns:
            (B, T + max_new_tokens) la sequenza estesa
        """
        for _ in range(max_new_tokens):
            logits, _ = self(idx)            # forward pass
            logits = logits[:, -1, :]        # solo l'ultima posizione -> (B, C)
            probs = F.softmax(logits, dim=-1)  # logit -> probabilita'
            idx_next = torch.multinomial(probs, num_samples=1)  # campiona
            idx = torch.cat((idx, idx_next), dim=1)  # concatena
        return idx

    def num_params(self):
        """Conta i parametri totali del modello."""
        return sum(p.numel() for p in self.parameters())
