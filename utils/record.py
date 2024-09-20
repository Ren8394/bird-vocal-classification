from pathlib import Path

import numpy as np

def write_loss_history(epoch, loss, filename=None):
    log_entry = f"{epoch}\t{loss:.6f}\n"

    with open(filename, "a") as f:
        f.write(log_entry)