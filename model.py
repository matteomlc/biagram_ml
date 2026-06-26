"""
model.py — Il modello (PASSO 4)
================================

PASSO 4: MULTI-HEAD ATTENTION.

Invece di un solo head di self-attention, ne usiamo diversi in PARALLELO.
Ogni head ha le sue matrici Q/K/V indipendenti e puo' specializzarsi su un
tipo di relazione diverso tra i token. Gli output dei head vengono concatenati.

Con n_embd=32 e 4 head: ogni head lavora con head_size = 32/4 = 8. Quattro
specialisti da 8 invece di un tuttofare da 32, stesso budget totale.
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


class MultiHeadAttention(nn.Module):
    """
    Piu' head di self-attention in parallelo, i cui output sono concatenati.

    Ogni head e' un'istanza della classe Head. Concatenando num_heads output
    da head_size ciascuno, si torna a num_heads * head_size = n_embd.
    """

    def __init__(self, num_heads, n_embd, head_size, block_size):
        super().__init__()
        # Una lista di head indipendenti. nn.ModuleList registra correttamente
        # i parametri di ciascun head (a differenza di una lista Python normale).
        self.heads = nn.ModuleList([
            Head(n_embd, head_size, block_size) for _ in range(num_heads)
        ])
        # Una proiezione lineare che "rimescola" gli output concatenati dei head.
        self.proj = nn.Linear(num_heads * head_size, n_embd)

    def forward(self, x):
        # Esegue ogni head e concatena i risultati lungo l'ultima dimensione.
        # Ogni head: (B, T, head_size). Concatenati: (B, T, num_heads*head_size).
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        # La proiezione finale ricombina le informazioni dei vari head.
        out = self.proj(out)  # (B, T, n_embd)
        return out


class BigramLanguageModel(nn.Module):
    """
    Ora con un head di self-attention tra l'embedding e l'output head.
    """

    def __init__(self, vocab_size, n_embd, block_size, n_head):
        super().__init__()
        self.block_size = block_size

        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)

        # NOVITA' (Passo 4): multi-head attention al posto del singolo head.
        # head_size = n_embd / n_head, cosi' la concatenazione torna a n_embd.
        head_size = n_embd // n_head
        self.sa_heads = MultiHeadAttention(n_head, n_embd, head_size, block_size)

        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)            # (B, T, n_embd)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)          # (T, n_embd)
        x = tok_emb + pos_emb                                 # (B, T, n_embd)

        # NOVITA' (Passo 4): la multi-head attention. x esce arricchito.
        x = self.sa_heads(x)                                  # (B, T, n_embd)

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