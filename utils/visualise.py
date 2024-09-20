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

def plot_gt_mask_pred(gt, mask, pred, paste, filename=None):
    assert gt.shape == mask.shape == pred.shape == paste.shape
    dim = gt.shape[0]
    duration = 100 # points

    # plot (dim rows, 4 cols)
    fig, axs = plt.subplots(nrows=dim, ncols=4, figsize=(17, 9), sharex=True, sharey=True)
    for i in range(dim):
        axs[i, 0].plot(gt[i][:duration], color=COLOR_LIST[i])
        axs[i, 1].plot(mask[i][:duration], color=COLOR_LIST[i])
        axs[i, 2].plot(pred[i][:duration], color=COLOR_LIST[i])
        axs[i, 3].plot(paste[i][:duration], color=COLOR_LIST[i])
    # set titles
    axs[0, 0].set_title("Original")
    axs[0, 1].set_title("Masked")
    axs[0, 2].set_title("Reconstructed")
    axs[0, 3].set_title("Reconstruction + Visible")
    # set labels
    axs[dim-1, 0].set_xlabel("Time")
    axs[dim-1, 1].set_xlabel("Time")
    axs[dim-1, 2].set_xlabel("Time")
    axs[dim-1, 3].set_xlabel("Time")

    title = "Ground Truth, Masked, Reconstructed, Reconstruction + Visible" if filename is None else Path(filename).stem
    fig.suptitle(title)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
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