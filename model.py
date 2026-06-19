"""
model.py — Il modello (PASSO 1)
================================

PASSO 1: spezziamo il percorso diretto "token -> logit" del bigram in:
    token -> rappresentazione ricca (n_embd) -> logit

Due novita' rispetto al bigram:
  1. nn.Embedding(vocab_size, n_embd): il token diventa un vettore astratto
     di n_embd numeri, NON piu' i logit diretti.
  2. nn.Linear(n_embd, vocab_size): un "traduttore" finale che converte
     la rappresentazione in logit sul vocabolario.

ATTENZIONE: questo passo da solo NON migliora la loss. Il modello guarda
ancora un solo token. Serve a creare lo "spazio in mezzo" (la rappresentazione)
dove ai prossimi passi inseriremo la self-attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    """
    Language model con embedding ricco + output head lineare.

    Manteniamo il nome 'BigramLanguageModel' per ora perche', a livello di
    capacita', e' ancora un bigram (guarda un solo token). Lo rinomineremo
    quando aggiungeremo la self-attention e diventera' un vero mini-GPT.
    """

    def __init__(self, vocab_size, n_embd):
        super().__init__()
        # NOVITA' 1: embedding ricco. Ogni token -> vettore di n_embd numeri.
        # Prima era (vocab_size, vocab_size); ora (vocab_size, n_embd).
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)

        # NOVITA' 2: output head. Traduce la rappresentazione (n_embd numeri)
        # nei logit finali (vocab_size numeri).
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        """
        Args:
            idx:     (B, T) indici dei token di input
            targets: (B, T) indici target (opzionale)

        Returns:
            logits: (B, T, vocab_size)
            loss:   scalare se ci sono targets, altrimenti None
        """
        # idx (B, T) -> embedding -> (B, T, n_embd)
        # Questa e' la rappresentazione ricca: lo "spazio in mezzo".
        # NOTA: qui, ai prossimi passi, andranno attention e feed-forward.
        tok_emb = self.token_embedding_table(idx)  # (B, T, n_embd)

        # rappresentazione -> output head -> logit (B, T, vocab_size)
        logits = self.lm_head(tok_emb)  # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            # Appiattiamo per cross_entropy: (B*T, vocab_size) e (B*T,)
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        """Genera testo autoregressivamente (identico al bigram)."""
        for _ in range(max_new_tokens):
            logits, _ = self(idx)            # forward pass
            logits = logits[:, -1, :]        # ultima posizione -> (B, vocab_size)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())
