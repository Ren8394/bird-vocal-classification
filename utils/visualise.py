from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.style.use("seaborn-v0_8-paper")
COLOR_LIST = ["navy", "orange", "#800020", "lightgreen", "lightyellow", "lightblue", "lightgray", "darkgray"]

def plot_loss_history(loss_history, filename=None):
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, color=COLOR_LIST[0])
    # last 1 epoch and its loss
    plt.scatter(len(loss_history)-1, loss_history[-1], color=COLOR_LIST[-1], marker="o")
    plt.annotate(f"{loss_history[-1]:.6f}", (len(loss_history)-1, loss_history[-1]), textcoords="offset points", xytext=(-5, 11), ha="center")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    title = "Loss History" if filename is None else Path(filename).stem
    plt.title(title)
    plt.grid()
    if filename is not None:
        Path.mkdir(Path.cwd().joinpath(Path(filename).parent), exist_ok=True, parents=True)
        plt.savefig(Path.cwd().joinpath(filename))
    else:
        plt.show()

def plot_gt_mask_pred(gt, mask, pred, filename=None):
    assert gt.shape == mask.shape == pred.shape
    
    duration = gt.shape[1]
    feq_band = gt.shape[-1]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(23, 7), sharex=False, sharey=True)
    for i, spec in enumerate((gt, mask, pred)):
        axs[i].imshow(spec.T, aspect="auto", origin="lower", cmap="viridis")
        axs[i].set_title("Ground Truth" if i == 0 else "Masked" if i == 1 else "Prediction")
        axs[i].set_xlabel("Time Frames")
        axs[i].set_ylabel("Frequency Bands")
    plt.tight_layout()
    if filename is not None:
        Path.mkdir(Path.cwd().joinpath(Path(filename).parent), exist_ok=True, parents=True)
        plt.savefig(Path.cwd().joinpath(filename))
    else:
        plt.show()

def plot_loss(train_loss_path, val_loss_path=None):
    train_loss = np.loadtxt(train_loss_path, delimiter="\t")
    _, ax = plt.subplots(figsize=(16, 6))
    ax.plot(train_loss[:, 0], train_loss[:, 1], label="train loss", color="#808000")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    plt.title("Train Loss")

    if val_loss_path is not None:
        val_loss = np.loadtxt(val_loss_path)
        ax.plot(val_loss[:, 0], val_loss[:, 1], label="val loss", color="#000080")
        plt.title("Train and Validation Loss")

    ax.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{train_loss_path.parent}/loss.jpg")

if __name__ == "__main__":
    plot_loss(
        train_loss_path=Path("./results/pretrain/MAE-Bert/p3-3-5/lr0005_wd0001_b512_e32/train_loss.txt"),
        val_loss_path=None
    )