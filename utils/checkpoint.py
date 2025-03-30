from pathlib import Path
import os

import torch


def save_checkpoint(model, optimizer, epoch, best_loss, filename="checkpoint.pth.tar"):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "best_loss": best_loss,
    }
    torch.save(state, filename)


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar", device="cpu"):
    checkpoint = torch.load(filename, map_location=device)

    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    epoch = checkpoint["epoch"]
    best_loss = checkpoint["best_loss"]

    return model, optimizer, epoch, best_loss


def save_and_cleanup_weights(model, out_path, max_checkpoints=3):
    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), out_path)
    else:
        torch.save(model.state_dict(), out_path)

    # Get list of all weights files and sort by modification time
    weights_files = sorted(Path(out_path).parent.glob(
        "*.pth"), key=os.path.getmtime)

    # If number of weights files exceeds max_checkpoints, remove the oldest ones
    while len(weights_files) > max_checkpoints:
        oldest_weight = weights_files.pop(0)
        oldest_weight.unlink(missing_ok=True)
