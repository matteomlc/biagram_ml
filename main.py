"""
main.py — Punto di ingresso
=============================

Orchestra tutti i moduli: prepara dati e tokenizer, crea il modello,
addestra, e valuta. Lancia con:

    python main.py

Il flusso e' lineare e leggibile perche' tutta la complessita' e'
"nascosta" nei moduli specializzati.
"""

import torch

import config
from data import SAMPLE_TEXT
from dataset import Dataset
from tokenizer import CharTokenizer
from model import BigramLanguageModel
from trainer import train
import evaluate

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

def main():
    # --- Setup ---
    device = config.get_device()
    torch.manual_seed(config.SEED)
    print(f"Device: {device}\n")

    # --- Dati e tokenizer ---
    # Per usare un testo tuo: load_text(path="tuo_file.txt", fallback=SAMPLE_TEXT)
    text = load_text(path=None, fallback=SAMPLE_TEXT)
    tokenizer = CharTokenizer(text)
    tokenizer.summary()

    data = tokenizer.encode_to_tensor(text)
    dataset = Dataset(
        data_tensor=data,
        train_split=config.TRAIN_SPLIT,
        block_size=config.BLOCK_SIZE,
        batch_size=config.BATCH_SIZE,
        device=device,
    )
    dataset.summary()

    # --- Modello ---
    model = BigramLanguageModel(tokenizer.vocab_size).to(device)
    print(f"\nParametri del modello: {model.num_params():,}\n")

    # --- Generazione PRIMA del training (pesi casuali) ---
    print("=" * 60)
    print("PRIMA DEL TRAINING (pesi casuali):")
    print("=" * 60)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, config.GEN_TOKENS_BEFORE)
    print(tokenizer.decode(out[0].tolist()), "\n")

    # --- Training ---
    print("=" * 60)
    print("TRAINING")
    print("=" * 60)
    first_loss = train(
        model=model,
        dataset=dataset,
        max_steps=config.MAX_STEPS,
        eval_interval=config.EVAL_INTERVAL,
        eval_iters=config.EVAL_ITERS,
        learning_rate=config.LEARNING_RATE,
    )

    # --- Generazione DOPO il training ---
    print("\n" + "=" * 60)
    print("DOPO IL TRAINING:")
    print("=" * 60)
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, config.GEN_TOKENS_AFTER)
    print(tokenizer.decode(out[0].tolist()))

    # --- Valutazione ---
    print("\n" + "=" * 60)
    print("VALUTAZIONE")
    print("=" * 60)
    evaluate.sanity_check(tokenizer.vocab_size, first_loss)
    evaluate.perplexity(model, dataset, config.EVAL_ITERS)
    for c in ["q", "c", "a", " ", "z"]:
        evaluate.next_char(model, tokenizer, c, device)


if __name__ == "__main__":
    main()
