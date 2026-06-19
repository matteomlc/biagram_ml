"""
checkpoint.py — Salvataggio e caricamento del modello
======================================================

Permette di conservare il modello addestrato su disco e ricaricarlo
in seguito, senza dover riaddestrare da zero.

Salviamo un "pacchetto" che contiene:
  - i pesi del modello (state_dict)
  - il vocabolario del tokenizer (chars)
  - gli iperparametri architetturali (vocab_size, n_embd)

Perche' anche il vocabolario? Perche' i pesi codificano relazioni tra
NUMERI (token 7 -> token 12), ma quale carattere sia il numero 7 dipende
dal vocabolario. Pesi e vocabolario devono viaggiare insieme, o il modello
ricaricato produrrebbe spazzatura.
"""

import os
import torch


def save_checkpoint(model, tokenizer, n_embd, path):
    """
    Salva pesi + vocabolario + iperparametri in un unico file .pt
    """
    checkpoint = {
        "model_state": model.state_dict(),   # i pesi appresi
        "chars": tokenizer.chars,            # il vocabolario
        "vocab_size": tokenizer.vocab_size,
        "n_embd": n_embd,
    }
    torch.save(checkpoint, path)
    print(f"Modello salvato in '{path}'")


def checkpoint_exists(path):
    """Verifica se esiste gia' un salvataggio."""
    return os.path.exists(path)


def load_checkpoint(path, model_class, tokenizer_class, device):
    """
    Ricarica un modello salvato.

    Ricostruisce lo scheletro del modello con la stessa architettura,
    ci carica dentro i pesi salvati, e ricostruisce il tokenizer
    con lo stesso vocabolario.

    Returns:
        (model, tokenizer) pronti all'uso
    """
    checkpoint = torch.load(path, map_location=device)

    # Ricostruisci il tokenizer con lo stesso vocabolario.
    # Passiamo una stringa che contiene tutti i caratteri salvati, cosi'
    # CharTokenizer ricostruisce le stesse identiche mappe stoi/itos.
    tokenizer = tokenizer_class("".join(checkpoint["chars"]))

    # Ricostruisci lo scheletro del modello e caricaci i pesi
    model = model_class(checkpoint["vocab_size"], checkpoint["n_embd"])
    model.load_state_dict(checkpoint["model_state"])
    model = model.to(device)
    model.eval()  # modalita' valutazione (utile per generare subito)

    print(f"Modello caricato da '{path}'")
    return model, tokenizer