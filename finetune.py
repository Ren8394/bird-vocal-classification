import argparse
import os
import warnings
import shutil
from datetime import datetime
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import classification_report, f1_score

from datasets.twbird import TWBird
from nets.models import MAE_Swin
from nets.losses import FocalLoss, DiceLoss
from utils.record import write_loss_history
from utils.checkpoint import save_checkpoint, load_checkpoint, save_and_cleanup_weights

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.manual_seed(21)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# TWBIRD_LABELS = [
#     "YB-S", "HA-S", "CR-S", "ML-S", "AA-S",
#     "AC-S", "NV-S", "LS-S", "PM-S", "PAL-S",
#     "PN-S", "FH-S", "HAC-S", "BG-S", "SB-S",
#     "ME-S", "GB-S", "PS-S", "PA-S", "SE-S",
#     "AM-S", "HS-S", "MH-S", "DI-S", "TM-S",
#     "TS-S", "PNI-S", "IP-S", "BS-S", "PC-S",
#     "RG-S", "LS-C", "HA-C", "AM-C", "TR-C",
#     "CM-C", "OS-S", "FH-C", "HL-C", "NJ-S",
#     "YB-C", "ACO-C", "PM-C", "DL-C", "SE-C",
#     "TI-S", "DI-C", "PMU-S", "EZ-S", "CP-S",
#     "TM-C", "HL-C", "SC-S"
# ]
TWBIRD_LABELS = [
    "YB-S", "HA-S", "CR-S", "ML-S", "AA-S",
    "AC-S", "NV-S", "LS-S", "PM-S", "PAL-S",
    "PN-S", "FH-S", "HAC-S", "BG-S", "SB-S",
    "ME-S", "GB-S", "PS-S", "PA-S", "SE-S",
    "AM-S", "HS-S", "MH-S", "DI-S", "TM-S",
    "TS-S", "PNI-S", "IP-S", "BS-S", "PC-S",
    "RG-S"
]
with_nota = False
print(f"#TWBIRD_LABELS: {len(TWBIRD_LABELS)}, NOTA: {with_nota}\n")


# class MLP(nn.Module):
#     def __init__(self, in_features, out_features):
#         super(MLP, self).__init__()
#         self.out_features = out_features
#         self.fc1 = nn.Linear(in_features, 256)
#         self.fc2 = nn.Linear(256, 128)
#         self.fc3 = nn.Linear(128, out_features)

#         self.bn1 = nn.BatchNorm1d(256)
#         self.bn2 = nn.BatchNorm1d(128)

#         self.drop = nn.Dropout(0.2)
#         self.relu = nn.ReLU()

#     def forward(self, x):
#         x = self.drop(self.relu(self.bn1(self.fc1(x))))
#         x = self.drop(self.relu(self.bn2(self.fc2(x))))
#         x = self.fc3(x)
#         return x

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.out_features = out_features
        self.head = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.head(x)
        return x


class CustomLoss(nn.Module):
    def __init__(self, alpha=0.25):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.focal = FocalLoss()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # loss = self.alpha * \
        #     self.focal(inputs, targets) + (1 - self.alpha) * \
        #     self.dice(inputs, targets)
        loss = \
            self.alpha * self.focal(inputs, targets) + \
            (1 - self.alpha) * self.bce(inputs, targets)
        return loss


def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("-wl", "--window_size", type=float, default=3.0)
    parser.add_argument("-hp", "--hop_length", type=float, default=0.5)

    # Model & Dataset
    parser.add_argument("-pw", "--weight", type=str, required=True,
                        help="Path to the pre-trained model")
    parser.add_argument("-dw", "--decoder_weight", type=str)
    parser.add_argument("--mask_t_ratio", type=float, default=0.2)
    parser.add_argument("--mask_f_ratio", type=float, default=0.2)

    # Loss
    parser.add_argument("-a", "--alpha", type=float, default=0.50)

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
    # dataloader
    train_dataloder = DataLoader(
        TWBird(
            src_file="./data/finetune/train.txt",
            labeled=True,
            window_size=args.window_size,
            hop_length=args.hop_length,
            status="train",  # status="train"
            with_nota=with_nota),
        batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    val_dataloder = DataLoader(
        TWBird(
            src_file="./data/finetune/val.txt",
            labeled=True,
            window_size=args.window_size,
            hop_length=args.hop_length,
            status="inference",
            with_nota=with_nota),
        batch_size=args.batch_size, num_workers=8, pin_memory=True
    )

    # Load pretrain model
    state_dict = torch.load(args.weight, map_location=DEVICE)
    # for state_dict w/ and w/o DataParallel
    state_dict = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(state_dict, strict=False)
    # freeze pretrain model
    for param in model.parameters():
        param.requires_grad = False
    record_structure_gredient(model, classifier, output_dir["result"])

    # criterion & optimizer
    optimizer = torch.optim.AdamW(
        classifier.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay)
    criterion = CustomLoss(alpha=args.alpha)
    # scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=20)

    # Initialize the training process
    stale = 0
    best_loss = np.inf
    Path(output_dir["result"]+"/train_loss.txt").unlink(missing_ok=True)
    Path(output_dir["result"]+"/val_loss.txt").unlink(missing_ok=True)

    model.eval()
    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss = 0
        classifier.train()
        for _, (x, y) in tqdm(enumerate(train_dataloder), desc="Training", leave=False, total=len(train_dataloder)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            latent = model.forward_feature(
                x=x.float(),
                mask_col_ratio=args.mask_t_ratio,
                mask_row_ratio=args.mask_f_ratio)
            output = classifier(latent)

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
        write_loss_history(
            epoch=epoch,
            loss=train_loss/len(train_dataloder),
            filename=output_dir["result"]+"/train_loss.txt"
        )

        val_loss = 0
        classifier.eval()
        for _, (x, y) in tqdm(enumerate(val_dataloder), desc="Validation", leave=False, total=len(val_dataloder)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            latent = model.forward_feature(
                x=x.float(),
                mask_col_ratio=args.mask_t_ratio,
                mask_row_ratio=args.mask_f_ratio)
            output = classifier(latent)

            loss = criterion(output, y)
            val_loss += loss.item()
        write_loss_history(
            epoch=epoch,
            loss=val_loss/len(val_dataloder),
            filename=output_dir["result"]+"/val_loss.txt"
        )
        if val_loss/len(val_dataloder) < best_loss:
            best_loss = val_loss/len(val_dataloder)
            torch.save(
                model.state_dict(),
                output_dir["weight"]+"/pretrain_best.pth")
            torch.save(
                classifier.state_dict(),
                output_dir["weight"]+"/classifier_best.pth")
            stale = 0
        else:
            stale += 1
            if stale > 50:
                for file in ["train_loss.txt", "val_loss.txt"]:
                    with open(f"{output_dir['result']}/{file}", "a") as f:
                        f.write(f"Early Stopping at Epoch {epoch}\n")
                break
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_loss,
            filename=output_dir["ckpt"]+"/checkpoint.pth"
        )
        scheduler.step(val_loss/len(val_dataloder))


@torch.no_grad()
def inference(args):
    # dataloader
    val_dataloder = DataLoader(
        TWBird(
            src_file="./data/finetune/val.txt",
            labeled=True,
            window_size=args.window_size,
            hop_length=args.hop_length,
            status="inference",
            with_nota=with_nota),
        batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    test_dataloder = DataLoader(
        TWBird(
            src_file="./data/finetune/test.txt",
            labeled=True,
            window_size=args.window_size,
            hop_length=args.hop_length,
            status="inference",
            with_nota=with_nota),
        batch_size=1, num_workers=4
    )

    model.load_state_dict(
        torch.load(
            output_dir["weight"]+"/pretrain_best.pth",
            map_location=DEVICE))
    classifier.load_state_dict(
        torch.load(
            output_dir["weight"]+"/classifier_best.pth",
            map_location=DEVICE))

    # Calculate probability threshold using validation set
    model.eval()
    classifier.eval()
    val_pred, val_true = [], []
    for _, (x, y) in tqdm(enumerate(val_dataloder), desc="Validation", total=len(val_dataloder)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        latent = model.forward_feature(
            x=x.float(),
            mask_col_ratio=args.mask_t_ratio,
            mask_row_ratio=args.mask_f_ratio)
        output = classifier(latent)

        val_true.append((y == 1).cpu().numpy())
        val_pred.append(torch.sigmoid(output).cpu().numpy())
    val_true = np.concatenate(val_true, axis=0)
    val_pred = np.concatenate(val_pred, axis=0)
    thresholds = calculate_threshold(val_true, val_pred)
    with open(output_dir["result"]+"/thresholds.txt", "w") as f:
        for i, threshold in enumerate(thresholds):
            f.write(f"{TWBIRD_LABELS[i]}: {threshold:.2f}\n")

    # Inference on test set
    classifier.eval()
    pred_species, true_species = [], []
    for _, (x, y) in tqdm(enumerate(test_dataloder), desc="Testing", total=len(test_dataloder)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        latent = model.forward_feature(
            x=x.float(),
            mask_col_ratio=args.mask_t_ratio,
            mask_row_ratio=args.mask_f_ratio)
        output = classifier(latent)

        # Multi-label classification
        # thershold_prob = 0.5
        true_species.append((y == 1).cpu().numpy())
        # pred_species.append(
        #     (torch.sigmoid(output) > thershold_prob).cpu().numpy())
        prob = torch.sigmoid(output).cpu().numpy()
        pred_binary = np.zeros_like(prob)
        for i, threshold in enumerate(thresholds):
            pred_binary[:, i] = (prob[:, i] >= threshold).astype(int)
        pred_species.append(pred_binary)

    true_species = np.concatenate(true_species, axis=0)
    pred_species = np.concatenate(pred_species, axis=0)
    np.savetxt(
        output_dir["result"]+"/true_species.txt",
        true_species, fmt="%d")
    np.savetxt(
        output_dir["result"]+"/pred_species.txt",
        pred_species, fmt="%d")
    if with_nota:
        report = classification_report(
            true_species,
            pred_species,
            target_names=TWBIRD_LABELS+["NOTA"],
            zero_division=0)
    else:
        report = classification_report(
            true_species,
            pred_species,
            target_names=TWBIRD_LABELS,
            zero_division=0)
    with open(output_dir["result"]+"/classification_report.txt", "w") as f:
        f.write(report)


def calculate_threshold(truth, pred, step=0.05):
    n_species = truth.shape[1]
    best_thresholds = []

    for i in range(n_species):
        best_threshold = 0.0
        best_f1 = 0.0
        for threshold in np.arange(0.0, 1.0 + step, step):
            pred_bin = (pred[:, i] >= threshold).astype(int)
            current_f1 = f1_score(truth[:, i], pred_bin, zero_division=0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                best_threshold = threshold
        best_thresholds.append(best_threshold)
    return best_thresholds


def float2str(num):
    return str(num).replace(".", "")


def record_args(args, output_dir):
    with open(Path(output_dir).joinpath("args.txt"), "w") as f:
        f.write("="*50 + "\n")
        f.write("Arguments:\n")
        f.write("-"*50 + "\n")
        for arg, value in vars(args).items():
            f.write(f"| {arg:12s}: {value}\n")
        f.write("="*50)


def record_structure_gredient(pretrain, classifier, output_dir):
    with open(Path(output_dir).joinpath("gradient.txt"), "w") as f:
        for name, param in pretrain.named_parameters():
            f.write(
                f"Pretrain Parameter: {name}, Requires Grad: {param.requires_grad}\n")
        f.write("="*50+"\n")
        for name, param in classifier.named_parameters():
            f.write(
                f"Finetune Parameter: {name}, Requires Grad: {param.requires_grad}\n")


if __name__ == "__main__":
    current_time = datetime.now()
    timestamp = current_time.strftime("%Y%m%d_%H%M")

    args = parse_args()

    model_record = \
        "AudioMAE_LP/" + \
        f"/w{float2str(args.window_size)}_h{float2str(args.hop_length)}"
    result_record = \
        f"a{float2str(args.alpha)}_lr{float2str(args.learning_rate)}_wd{float2str(args.weight_decay)}_b{args.batch_size}_e{args.epochs}" + \
        f"/{timestamp}"
    output_dir = {
        "result": f"./results/{model_record}/{result_record}",
        "weight": f"./weights/{model_record}/{result_record}",
        "ckpt": f"./ckpt/{model_record}/{result_record}"
    }

    # pretrain model
    if args.window_size == 3.0:
        in_shape = (1, 320, 128)
    elif args.window_size == 1.0:
        in_shape = (1, 128, 128)

    model = MAE_Swin(
        in_shape=in_shape, patch_size=(16, 16),
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        use_mask_2d=True
    )
    if with_nota:
        classifier = MLP(in_features=768, out_features=len(TWBIRD_LABELS)+1)
    else:
        classifier = MLP(in_features=768, out_features=len(TWBIRD_LABELS))
    print(f"Classifier: {classifier.out_features}")

    model.to(DEVICE)
    classifier.to(DEVICE)
    if args.train:
        try:
            Path(output_dir["result"]).mkdir(exist_ok=True, parents=True)
            Path(output_dir["weight"]).mkdir(exist_ok=True, parents=True)
            Path(output_dir["ckpt"]).mkdir(exist_ok=True, parents=True)

            # Save the arguments record
            record_args(args, output_dir["result"])

            # Start training
            train(args)
            print(f"Finish training. {model_record}")
        except (Exception, KeyboardInterrupt) as e:
            shutil.rmtree(f"./results/{model_record}/{result_record}")
            shutil.rmtree(f"./weights/{model_record}/{result_record}")
            print(f"The training process failed due to {e}")

    if args.inference:
        inference(args)
