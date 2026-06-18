# Bigram Language Model

Un language model a livello carattere, costruito da zero in PyTorch.
Prevede il prossimo carattere basandosi sull'ultimo carattere visto.

È il punto di partenza per costruire un Transformer: la struttura del
progetto è pensata per essere estesa (basterà sostituire `model.py`).

## Come lanciarlo

```bash
pip install torch
python main.py
```

Funziona su Mac Apple Silicon (MPS), GPU NVIDIA (CUDA) o CPU — il device
viene scelto automaticamente.

## Struttura del progetto

Ogni file ha una responsabilità chiara:

| File           | Responsabilità                                              |
|----------------|-------------------------------------------------------------|
| `main.py`      | Punto di ingresso: orchestra tutti i moduli                 |
| `config.py`    | Iperparametri e selezione del device (modifica qui)         |
| `data.py`      | Il testo di esempio (Dante)                                 |
| `tokenizer.py` | Conversione testo ↔ numeri (`CharTokenizer`)                |
| `dataset.py`   | Caricamento dati, split train/val, batching                 |
| `model.py`     | L'architettura (`BigramLanguageModel`)                      |
| `trainer.py`   | Il training loop e la stima della loss                      |
| `evaluate.py`  | Test formali: sanity check, perplexity, test qualitativo    |

## Sperimentare

- **Iperparametri**: modifica i valori in `config.py` (learning rate,
  numero di step, dimensione del batch...).
- **Testo tuo**: salva un file `.txt` e in `main.py` cambia
  `load_text(path=None, ...)` in `load_text(path="tuo_file.txt", ...)`.
  Per un test più ricco, scarica la Divina Commedia completa da Project
  Gutenberg.

## Prossimo passo: il Transformer

Per trasformare questo bigram in un mini-GPT si modifica solo `model.py`,
aggiungendo, un pezzo alla volta:

1. Embedding ricco (`vocab_size, n_embd`) + layer di output
2. Positional embedding
3. Self-attention
4. Feed-forward network
5. Stack di N transformer block

Il resto del progetto (dati, tokenizer, trainer, evaluate) resta invariato:
è esattamente il vantaggio di aver separato le responsabilità.
