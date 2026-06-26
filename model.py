"""
model.py — Il modello (PASSO 3)
================================

PASSO 3: aggiungiamo la SELF-ATTENTION (un singolo head).

Finora ogni token veniva predetto guardando solo se stesso. Ora ogni token
puo' guardare i token PRECEDENTI e raccogliere informazione da loro, tramite
il meccanismo Query / Key / Value (vedi DOCUMENTAZIONE.md sez. 8).

Flusso di un head di attention:
  1. da ogni token ricava query, key, value (tre nn.Linear appresi)
  2. punteggi = query . key  (prodotto scalare tra ogni coppia di token)
  3. scala per 1/sqrt(head_size) e applica la MASCHERA CAUSALE (solo il passato)
  4. softmax -> pesi di attenzione (sommano a 1)
  5. output = media pesata dei value

Questo e' il passo in cui, per la prima volta, la val perplexity dovrebbe
SCENDERE: il modello inizia davvero a usare il contesto.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    """
    Un singolo "head" di self-attention.

    Trasforma la sequenza di vettori in ingresso (B, T, n_embd) in una nuova
    sequenza (B, T, head_size) dove ogni posizione ha assorbito informazione
    dalle posizioni precedenti.
    """

    def __init__(self, n_embd, head_size, block_size):
        super().__init__()
        # Le tre proiezioni lineari apprese. Niente bias, come da prassi.
        # Ognuna trasforma un vettore da n_embd a head_size numeri.
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key   = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)

        # La maschera causale: una matrice triangolare inferiore di 1.
        # tril = "TRiangular Lower". Serve a impedire a ogni token di guardare
        # i token FUTURI. La registriamo come "buffer" (non e' un parametro
        # apprendibile, ma deve seguire il modello su CPU/GPU).
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        # x: (B, T, n_embd) — B sequenze, T token, n_embd numeri per token
        B, T, C = x.shape

        # 1. Ricava query, key, value da ogni token
        q = self.query(x)  # (B, T, head_size)
        k = self.key(x)    # (B, T, head_size)
        v = self.value(x)  # (B, T, head_size)

        head_size = q.shape[-1]

        # 2. Punteggi di attenzione: prodotto scalare query . key.
        # Per ogni coppia di token (i, j) calcoliamo quanto la query di i
        # combacia con la key di j. In forma matriciale: q @ k trasposta.
        # k.transpose(-2, -1) scambia le ultime due dimensioni: (B, T, hs) -> (B, hs, T)
        # Risultato: (B, T, T) — per ogni token, un punteggio verso ogni altro token.
        scores = q @ k.transpose(-2, -1)  # (B, T, T)

        # 3a. Scaling: dividiamo per sqrt(head_size) per tenere la softmax "morbida"
        scores = scores * (head_size ** -0.5)

        # 3b. Maschera causale: dove tril == 0 (cioe' i token futuri), mettiamo
        # -infinito. Cosi' dopo la softmax quei pesi diventano 0: ogni token
        # puo' guardare solo se stesso e i precedenti, mai il futuro.
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        # 4. Softmax sull'ultima dimensione: i punteggi diventano pesi che
        # sommano a 1 (per ogni token, una distribuzione su quelli precedenti).
        weights = F.softmax(scores, dim=-1)  # (B, T, T)

        # 5. Media pesata dei value: ogni token diventa la combinazione pesata
        # dei value dei token a cui presta attenzione.
        out = weights @ v  # (B, T, T) @ (B, T, head_size) -> (B, T, head_size)
        return out


class BigramLanguageModel(nn.Module):
    """
    Ora con un head di self-attention tra l'embedding e l'output head.
    """

    def __init__(self, vocab_size, n_embd, block_size):
        super().__init__()
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # NOVITA': un head di self-attention. Per ora head_size = n_embd,
        # cosi' l'output dell'attention ha la stessa dimensione dell'input.
        self.sa_head = Head(n_embd, n_embd, block_size)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)            # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)          # (T, n_embd)
        x = tok_emb + pos_emb                                 # (B, T, n_embd)

        # NOVITA': la self-attention. x esce arricchito col contesto.
        x = self.sa_head(x)                                   # (B, T, n_embd)

        logits = self.lm_head(x)                              # (B, T, vocab_size)

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
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]  # contesto entro block_size
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def num_params(self):
        return sum(p.numel() for p in self.parameters())