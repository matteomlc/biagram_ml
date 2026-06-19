"""
trainer.py — Il ciclo di addestramento
========================================

Contiene la logica di training: il loop forward -> loss -> backward -> update,
e la funzione di valutazione della loss.

Separare il trainer dal modello significa che lo stesso trainer puo'
addestrare QUALSIASI modello (bigram, transformer, ...) senza modifiche.
"""

import torch


@torch.no_grad()
def estimate_loss(model, dataset, eval_iters):
    """
    Calcola la loss media su piu' batch, per train e validation.
    Mediare su molti batch da' una stima piu' stabile.
    """
    out = {}
    model.eval()  # modalita' valutazione
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = dataset.get_batch(split)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()  # torna in modalita' training
    return out


def train(model, dataset, max_steps, eval_interval, eval_iters, learning_rate,
          logger=None):
    """
    Addestra il modello.

    Args:
        logger: un RunLogger opzionale. Se fornito, registra le loss
                ad ogni eval_interval per il tracciamento su file.

    Returns:
        first_loss: la loss al primo step (utile per il sanity check)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    first_loss = None

    for step in range(max_steps):

        # Valutazione periodica
        if step % eval_interval == 0:
            losses = estimate_loss(model, dataset, eval_iters)
            print(f"Step {step:5d} | train loss: {losses['train']:.4f} "
                  f"| val loss: {losses['val']:.4f}")
            if logger is not None:
                logger.log_loss(step, losses['train'], losses['val'])

        # --- Il training step: il cuore di tutto ---
        xb, yb = dataset.get_batch("train")   # 1. campiona un batch
        logits, loss = model(xb, yb)          # 2. forward pass
        optimizer.zero_grad(set_to_none=True)  # 3. azzera i gradienti
        loss.backward()                        # 4. backward pass
        optimizer.step()                       # 5. aggiorna i pesi

        if first_loss is None:
            first_loss = loss.item()

    # Loss finale
    losses = estimate_loss(model, dataset, eval_iters)
    print(f"Step {max_steps:5d} | train loss: {losses['train']:.4f} "
          f"| val loss: {losses['val']:.4f}")
    if logger is not None:
        logger.log_loss(max_steps, losses['train'], losses['val'])

    return first_loss