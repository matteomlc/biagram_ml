"""
model.py — Il modello (PASSO 2)
================================

PASSO 2: aggiungiamo il POSITIONAL EMBEDDING.

Oltre a "quale carattere e'" (token embedding), il modello ora sa anche
"in quale posizione si trova" (positional embedding). I due vengono sommati:

    rappresentazione = embedding_del_carattere + embedding_della_posizione

Perche'? La self-attention (Passo 3) avra' bisogno di distinguere l'ordine
dei caratteri. Senza posizione, "vita" e "tavi" sarebbero indistinguibili.

Come il Passo 1, da solo questo cambia poco nei numeri: e' preparatorio.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Language model con token embedding + positional embedding + output head.
    """

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.block_size = block_size

        # Embedding del CARATTERE: "quale token sono" -> vettore n_embd
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # NOVITA': embedding della POSIZIONE: "dove mi trovo" -> vettore n_embd
        # Ha una riga per ogni posizione possibile (0 ... block_size-1)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # Output head: rappresentazione (n_embd) -> logit (vocab_size)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # Embedding del carattere: (B, T) -> (B, T, n_embd)
        tok_emb = self.token_embedding_table(idx)

        # NOVITA': embedding della posizione.
        # Creiamo gli indici di posizione [0, 1, 2, ..., T-1] e li mappiamo.
        # Risultato: (T, n_embd) — un vettore per ogni posizione.
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)  # (T, n_embd)

        # Somma carattere + posizione.
        # pos_emb (T, n_embd) viene "broadcast" su tutte le B sequenze.
        x = tok_emb + pos_emb  # (B, T, n_embd)

        # NOTA: qui, al Passo 3, andra' la self-attention (operera' su x)

        # Output head -> logit
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """
        Genera testo autoregressivamente.

        NOVITA': dobbiamo limitare il contesto a block_size token, perche'
        la tabella di posizione ha solo block_size righe. Se la sequenza
        cresce oltre, teniamo solo gli ultimi block_size caratteri.
        """
        for _ in range(max_new_tokens):
            # Taglia il contesto agli ultimi block_size token
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())