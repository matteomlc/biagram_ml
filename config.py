"""
config.py — Configurazione centrale del progetto
=================================================

Tutti gli iperparametri e le impostazioni in un unico posto.
Modifica QUI quando vuoi sperimentare (learning rate, dimensioni, ecc.)
invece di andare a caccia di numeri sparsi nel codice.
"""

import torch


# --- Iperparametri di training ---
BATCH_SIZE = 32        # Quante sequenze processiamo in parallelo per step
BLOCK_SIZE = 8         # Lunghezza del contesto (caratteri visti per predizione)
N_EMBD = 32            # Dimensione della rappresentazione di ogni token (Passo 1)
N_HEAD = 4             # Numero di head di attention in parallelo (Passo 4)
MAX_STEPS = 10000      # Numero totale di training step
EVAL_INTERVAL = 1000   # Ogni quanti step valutare e stampare la loss
EVAL_ITERS = 200       # Su quanti batch mediare la loss di valutazione
LEARNING_RATE = 1e-2   # Dimensione del passo nel gradient descent

# --- Split dei dati ---
TRAIN_SPLIT = 0.9      # Frazione di dati usata per il training (resto: validation)

# --- Riproducibilità ---
SEED = 42              # Seed per risultati riproducibili

# --- Tracciamento ---
RUN_NOTES = "Passo 5 - aggiunta feed-forward network"  # descrizione del run

# --- Salvataggio modello ---
CHECKPOINT_PATH = "modello_passo5.pt"   # dove salvare/caricare i pesi del modello

# --- Generazione ---
GEN_TOKENS_BEFORE = 200   # Caratteri da generare prima del training (gibberish)
GEN_TOKENS_AFTER = 500    # Caratteri da generare dopo il training


def get_device():
    """
    Sceglie automaticamente il miglior device disponibile.
      - "mps"  = Metal Performance Shaders (Mac Apple Silicon)
      - "cuda" = GPU NVIDIA
      - "cpu"  = fallback universale
    """
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"