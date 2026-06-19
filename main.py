"""
main.py — Punto di ingresso
=============================

Orchestra tutti i moduli. Lancia con:

    python main.py

All'avvio, se esiste gia' un modello salvato, chiede se caricarlo
(saltando il training) o riaddestrare da zero. A fine training, i pesi
vengono sempre salvati su disco.
"""

import torch

import config
from data import SAMPLE_TEXT
from dataset import load_text, Dataset
from tokenizer import CharTokenizer
from model import BigramLanguageModel
from trainer import train
from logger import RunLogger
import checkpoint
import evaluate


def ask_load_or_train():
    """
    Se esiste un checkpoint, chiede all'utente cosa fare.
    Returns True se si vuole CARICARE, False se si vuole RIADDESTRARE.
    """
    if not checkpoint.checkpoint_exists(config.CHECKPOINT_PATH):
        return False  # nessun salvataggio: si addestra per forza

    print(f"Trovato un modello salvato in '{config.CHECKPOINT_PATH}'.")
    risposta = input("Vuoi [C]aricarlo o [R]iaddestrare da zero? [C/R]: ").strip().lower()
    return risposta != "r"  # qualsiasi cosa diversa da 'r' -> carica


def generate_sample(model, tokenizer, device, n_tokens):
    """Genera un campione di testo partendo da un contesto vuoto."""
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(context, n_tokens)
    return tokenizer.decode(out[0].tolist())


def main():
    # --- Setup ---
    device = config.get_device()
    torch.manual_seed(config.SEED)
    print(f"Device: {device}\n")

    # ========================================================================
    # RAMO A — CARICAMENTO: il modello esiste e l'utente vuole caricarlo
    # ========================================================================
    if ask_load_or_train():
        model, tokenizer = checkpoint.load_checkpoint(
            path=config.CHECKPOINT_PATH,
            model_class=BigramLanguageModel,
            tokenizer_class=CharTokenizer,
            device=device,
        )
        print(f"Parametri del modello: {model.num_params():,}\n")

        # Genera direttamente, senza riaddestrare
        print("=" * 60)
        print("GENERAZIONE (modello caricato):")
        print("=" * 60)
        print(generate_sample(model, tokenizer, device, config.GEN_TOKENS_AFTER))
        return

    # ========================================================================
    # RAMO B — TRAINING: nessun salvataggio, o l'utente vuole riaddestrare
    # ========================================================================

    # --- Dati e tokenizer ---
    text = load_text(path="commedia.txt", fallback=SAMPLE_TEXT)
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
    model = BigramLanguageModel(tokenizer.vocab_size, config.N_EMBD, config.BLOCK_SIZE).to(device)
    print(f"\nParametri del modello: {model.num_params():,}\n")

    # --- Logger ---
    logger = RunLogger(config_dict={
        "BATCH_SIZE": config.BATCH_SIZE,
        "BLOCK_SIZE": config.BLOCK_SIZE,
        "N_EMBD": config.N_EMBD,
        "MAX_STEPS": config.MAX_STEPS,
        "LEARNING_RATE": config.LEARNING_RATE,
        "vocab_size": tokenizer.vocab_size,
    })
    logger.set_notes(config.RUN_NOTES)
    logger.set_n_params(model.num_params())

    # --- Generazione PRIMA del training ---
    print("=" * 60)
    print("PRIMA DEL TRAINING (pesi casuali):")
    print("=" * 60)
    print(generate_sample(model, tokenizer, device, config.GEN_TOKENS_BEFORE), "\n")

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
        logger=logger,
    )

    # --- Generazione DOPO il training ---
    print("\n" + "=" * 60)
    print("DOPO IL TRAINING:")
    print("=" * 60)
    sample = generate_sample(model, tokenizer, device, config.GEN_TOKENS_AFTER)
    print(sample)
    logger.set_sample(sample)

    # --- Valutazione ---
    print("\n" + "=" * 60)
    print("VALUTAZIONE")
    print("=" * 60)
    evaluate.sanity_check(tokenizer.vocab_size, first_loss)
    ppl = evaluate.perplexity(model, dataset, config.EVAL_ITERS)
    logger.set_perplexity(ppl)
    for c in ["q", "c", "a", " ", "z"]:
        evaluate.next_char(model, tokenizer, c, device)

    # --- Salvataggio: log + pesi del modello ---
    logger.save()
    checkpoint.save_checkpoint(model, tokenizer, config.N_EMBD, config.BLOCK_SIZE, config.CHECKPOINT_PATH)


if __name__ == "__main__":
    main()