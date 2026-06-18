"""
evaluate.py — Valutazione del modello
=======================================

I test formali per capire quanto e' buono il modello:
  1. sanity_check   — la loss iniziale e' quella attesa?
  2. perplexity     — quanto e' "confuso" il modello in media?
  3. next_char      — test qualitativo dei pattern appresi
"""

import math
import torch


def sanity_check(vocab_size, measured_initial_loss):
    """
    La loss iniziale (pesi casuali) dovrebbe essere ~log(vocab_size).
    Se e' molto diversa, c'e' probabilmente un bug.
    """
    expected = math.log(vocab_size)
    print(f"Loss iniziale attesa : {expected:.4f}")
    print(f"Loss iniziale misurata: {measured_initial_loss:.4f}")
    if abs(expected - measured_initial_loss) < 0.5:
        print("PASS — inizializzazione corretta\n")
    else:
        print("ATTENZIONE — loss iniziale lontana dall'attesa, possibile bug\n")


@torch.no_grad()
def perplexity(model, dataset, eval_iters):
    """
    Perplexity = e^loss. "Tra quante opzioni il modello esita in media."
    Piu' bassa = meglio. Vicina a vocab_size = sta tirando a caso.
    """
    model.eval()
    results = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        mean_loss = losses.mean().item()
        ppl = math.exp(mean_loss)
        results[split] = ppl
        print(f"{split:5s} | loss: {mean_loss:.4f} | perplexity: {ppl:.2f}")
    model.train()

    gap = results["val"] - results["train"]
    print(f"\nGap val-train: {gap:.2f}")
    if gap > results["train"] * 0.5:
        print("Segnale di OVERFITTING\n")
    else:
        print("Generalizzazione ragionevole\n")
    return results


@torch.no_grad()
def next_char(model, tokenizer, char, device, top_k=5):
    """
    Test qualitativo: dato un carattere, mostra i prossimi piu' probabili.
    Per un bigram dovrebbe riflettere i pattern dell'italiano (es. q -> u).
    """
    if char not in tokenizer.stoi:
        print(f"'{char}' non e' nel vocabolario")
        return
    idx = torch.tensor([[tokenizer.stoi[char]]], device=device)
    logits, _ = model(idx)
    probs = torch.softmax(logits[0, -1, :], dim=-1)
    top_probs, top_idx = torch.topk(probs, top_k)
    print(f"Dopo '{char}' il modello prevede:")
    for p, i in zip(top_probs, top_idx):
        ch = tokenizer.itos[i.item()]
        display = repr(ch) if ch in (" ", "\n") else ch
        print(f"   {display:6s} -> {p.item()*100:5.1f}%")
    print()
