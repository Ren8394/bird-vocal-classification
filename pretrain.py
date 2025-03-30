from utils.visualise import plot_gt_mask_pred
from utils.checkpoint import save_checkpoint, load_checkpoint, save_and_cleanup_weights
from utils.record import write_loss_history
from nets.models import MAE_Swin
from datasets.twbird import TWBird
import torch
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
warnings.filterwarnings("ignore")

torch.random.manual_seed(2024)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("-w", "--window_size", type=float, default=3.0)
    parser.add_argument("-hp", "--hop_length", type=float, default=0.5)

    # Model & Dataset
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--weight", type=str)
    parser.add_argument("-mr", "--mask_ratio", type=float, default=0.8)

    # Hyperparameters
    parser.add_argument("-e", "--epochs", type=int, default=32)
    parser.add_argument("-b", "--batch_size", type=int, default=512)
    parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.0001)

    # Decision
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")

    return parser.parse_args()


def train(args):

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=DEVICE)
        model.load_state_dict(checkpoint["state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"]
        best_loss = checkpoint["best_loss"]
    else:
        start_epoch = 0
        best_loss = np.inf
        Path(
            f"./results/{model_record}/{result_record}/train_loss.txt").unlink(missing_ok=True)
        Path(
            f"./results/{model_record}/{result_record}/val_loss.txt").unlink(missing_ok=True)

    # model training
    for epoch in trange(start_epoch, args.epochs, desc="Epoch"):
        train_loss = 0
        model.train()
        for _, (x, _) in tqdm(enumerate(train_dataloder), desc="Training", leave=False, total=len(train_dataloder)):
            x = x.to(DEVICE)
            optimizer.zero_grad()
            _, _, _, loss = model(x=x.float(), mask_ratio=args.mask_ratio)
            loss = loss.mean()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        scheduler.step()
        write_loss_history(
            epoch=epoch, loss=train_loss/len(train_dataloder),
            filename=f"./results/{model_record}/{result_record}/train_loss.txt"
        )

        val_loss = 0
        model.eval()
        for _, (x, _) in tqdm(enumerate(val_dataloder), desc="Validation", leave=False, total=len(val_dataloder)):
            x = x.to(DEVICE)
            _, _, _, loss = model(x=x.float(), mask_ratio=args.mask_ratio)
            loss = loss.mean()
            val_loss += loss.item()

        write_loss_history(
            epoch=epoch, loss=val_loss/len(val_dataloder),
            filename=f"./results/{model_record}/{result_record}/val_loss.txt"
        )
        if val_loss/len(val_dataloder) < best_loss:
            best_loss = val_loss/len(val_dataloder)
            save_and_cleanup_weights(
                model, f"/media/birdsong/disk02/bird-vocal-classification/weights/{model_record}/{result_record}/best_e{epoch}.pth")
        save_checkpoint(
            model, optimizer, epoch, best_loss,
            filename=f"/media/birdsong/disk02/bird-vocal-classification/ckpt/{model_record}/{result_record}/best.pth.tar"
        )


@torch.no_grad()
def inference(args):
    if args.weight:
        weight = torch.load(args.weight, map_location=DEVICE)
        model.load_state_dict(weight)
    else:
        try:
            weight_path = sorted(Path(
                f"/media/birdsong/disk02/bird-vocal-classification/weights/{model_record}/{result_record}").glob("*.pth"))[-1]
            weight = torch.load(weight_path, map_location=DEVICE)
            model.load_state_dict(weight)
        except:
            print("No weight found.")

    # get used model
    model_to_use = model.module if isinstance(
        model, torch.nn.DataParallel) else model

    # model inference
    for i, (x, _) in tqdm(enumerate(test_dataloder), desc="Inference", total=len(test_dataloder)):
        if i % 100000 != 0:
            continue
        x = x.to(DEVICE)
        pred, mask, _, _ = model(x=x.float(), mask_ratio=args.mask_ratio)

        # reconstruct spectrogram w/ prediction and mask
        pred = model_to_use.unpatchify(pred)
        mask = mask.unsqueeze(-1).repeat(
            1, 1, model_to_use.patch_embed.patch_size[0] *
            model_to_use.patch_embed.patch_size[1] * 1
        )
        mask = model_to_use.unpatchify(mask)
        x_masked = x * (1 - mask)
        x_reconstructed = pred * mask + x_masked

        # visualise
        x = x.squeeze(dim=0).cpu().detach().numpy()
        x_masked = x_masked.squeeze(dim=0).cpu().detach().numpy()
        x_reconstructed = x_reconstructed.squeeze(dim=0).cpu().detach().numpy()

        plot_gt_mask_pred(
            gt=x[0], mask=x_masked[0], pred=x_reconstructed[0],
            filename=f"./results/{model_record}/{result_record}/inference_{i//100000}.jpg"
        )


if __name__ == "__main__":
    args = parse_args()
    model_record = \
        f"pretrain/AudioMAE/mr{str(args.mask_ratio).replace('.', '')}" + \
        f"/w{str(args.window_size).replace('.', '')}" + \
        f"_h{str(args.hop_length).replace('.', '')}"
    result_record = \
        f"lr{str(args.learning_rate).split('.')[-1]}_wd{str(args.weight_decay).split('.')[-1]}_b{args.batch_size}_e{args.epochs}"

    # AudioMAE Base Version
    if args.window_size == 3.0:
        in_shape = (1, 320, 128)
    elif args.window_size == 1.0:
        in_shape = (1, 128, 128)

    model = MAE_Swin(
        in_shape=in_shape, patch_size=(16, 16),
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16
    ).to(DEVICE)

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # dataloader
    train_dataloder = DataLoader(
        TWBird(src_file="./data/pretrain/train.txt", labeled=False,
               window_size=args.window_size, hop_length=args.hop_length),
        batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    val_dataloder = DataLoader(
        TWBird(src_file="./data/pretrain/val.txt", labeled=False,
               window_size=args.window_size, hop_length=args.hop_length),
        batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    test_dataloder = DataLoader(
        TWBird(src_file="./data/pretrain/test.txt", labeled=False,
               window_size=args.window_size, hop_length=args.hop_length),
        batch_size=1, num_workers=4, pin_memory=True
    )

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs)

    if args.train:
        Path(
            f"./results/{model_record}/{result_record}").mkdir(exist_ok=True, parents=True)
        Path(f"/media/birdsong/disk02/bird-vocal-classification/weights/{model_record}/{result_record}").mkdir(
            exist_ok=True, parents=True)
        Path(f"/media/birdsong/disk02/bird-vocal-classification/ckpt/{model_record}/{result_record}").mkdir(
            exist_ok=True, parents=True)
        with open(f"./results/{model_record}/{result_record}/gradient.txt", "w") as f:
            for name, param in model.named_parameters():
                f.write(
                    f"Pretrain Parameter: {name}, Requires Grad: {param.requires_grad}\n")
        train(args)
        print(f"Finish training. {model_record}")
    if args.inference:
        inference(args)
