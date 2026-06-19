"""
logger.py — Tracciamento degli esperimenti
============================================

Registra ogni run di training su file, per tenere uno storico
dell'evoluzione del modello. Produce tre output (nella cartella logs/):

  1. history.txt          — storico cumulativo: ogni run aggiunto in coda
  2. run_<timestamp>.txt  — file dettagliato del singolo run
  3. run_<timestamp>.csv  — le loss step-per-step (per farci grafici)

L'idea: incapsuliamo lo stato del log (i percorsi dei file, le loss
raccolte) in una classe RunLogger, creata all'inizio di ogni run.
"""

import os
import csv
from datetime import datetime


class RunLogger:
    """
    Raccoglie le informazioni di un run e le scrive su file.

    Uso tipico:
        logger = RunLogger(config_dict)      # all'inizio del run
        logger.log_loss(step, train, val)    # durante il training
        logger.set_perplexity(...)           # a fine valutazione
        logger.set_sample(testo_generato)    # campione di testo
        logger.save()                        # scrive tutto su file
    """

    def __init__(self, config_dict, log_dir="logs"):
        # Timestamp che identifica univocamente questo run
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.config = config_dict
        self.log_dir = log_dir

        # Stato raccolto durante il run
        self.loss_history = []      # lista di (step, train_loss, val_loss)
        self.perplexity = None      # dict {"train": ..., "val": ...}
        self.sample = ""            # campione di testo generato
        self.n_params = None        # numero di parametri del modello
        self.notes = ""             # nota descrittiva del run (es. "Passo 1")

        # Crea la cartella dei log se non esiste
        os.makedirs(self.log_dir, exist_ok=True)

        # Percorsi dei file
        self.history_path = os.path.join(self.log_dir, "history.txt")
        self.run_txt_path = os.path.join(self.log_dir, f"run_{self.timestamp}.txt")
        self.run_csv_path = os.path.join(self.log_dir, f"run_{self.timestamp}.csv")

    # --- Metodi per raccogliere informazioni durante il run ---

    def set_notes(self, notes):
        self.notes = notes

    def set_n_params(self, n):
        self.n_params = n

    def log_loss(self, step, train_loss, val_loss):
        """Registra le loss a un dato step. Chiamato durante il training."""
        self.loss_history.append((step, train_loss, val_loss))

    def set_perplexity(self, perplexity_dict):
        self.perplexity = perplexity_dict

    def set_sample(self, text):
        self.sample = text

    # --- Scrittura su file ---

    def _build_report(self):
        """Costruisce il testo del report (usato sia per history che per run.txt)."""
        lines = []
        lines.append("=" * 60)
        lines.append(f"RUN: {self.timestamp}")
        if self.notes:
            lines.append(f"NOTE: {self.notes}")
        lines.append("=" * 60)

        # Configurazione
        lines.append("\n--- Configurazione ---")
        for key, value in self.config.items():
            lines.append(f"  {key}: {value}")
        if self.n_params is not None:
            lines.append(f"  parametri modello: {self.n_params:,}")

        # Andamento della loss
        lines.append("\n--- Andamento loss ---")
        for step, train, val in self.loss_history:
            lines.append(f"  step {step:6d} | train {train:.4f} | val {val:.4f}")

        # Perplexity finale
        if self.perplexity is not None:
            lines.append("\n--- Perplexity finale ---")
            for split, ppl in self.perplexity.items():
                lines.append(f"  {split}: {ppl:.2f}")

        # Campione di testo
        if self.sample:
            lines.append("\n--- Campione generato ---")
            lines.append(self.sample)

        lines.append("\n")
        return "\n".join(lines)

    def save(self):
        """Scrive tutti e tre i file di output."""
        report = self._build_report()

        # 1. File dettagliato del singolo run
        with open(self.run_txt_path, "w", encoding="utf-8") as f:
            f.write(report)

        # 2. Storico cumulativo (append in coda)
        with open(self.history_path, "a", encoding="utf-8") as f:
            f.write(report)

        # 3. CSV delle loss (per grafici)
        with open(self.run_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["step", "train_loss", "val_loss"])
            writer.writerows(self.loss_history)

        print(f"\nLog salvati nella cartella '{self.log_dir}/':")
        print(f"  - {self.run_txt_path}")
        print(f"  - {self.run_csv_path}")
        print(f"  - {self.history_path} (aggiornato)")