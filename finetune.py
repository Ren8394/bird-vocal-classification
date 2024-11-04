import argparse
import os
import warnings
from itertools import chain
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

from datasets.twbird import TWBird
from nets.models import MAE_Swin
from utils.record import write_loss_history
from utils.checkpoint import save_checkpoint, load_checkpoint, save_and_cleanup_weights

warnings.filterwarnings("ignore")
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

torch.manual_seed(21)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TWBIRD_LABELS = [
    "YB-S", "HA-S", "CR-S", "ML-S", "AA-S", 
    "AC-S", "NV-S", "LS-S", "PM-S", "PAL-S", 
    "PN-S", "FH-S", "HAC-S", "BG-S", "SB-S", 
    "ME-S", "GB-S", "PS-S", "PA-S", "SE-S", 
    "AM-S", "HS-S", "MH-S", "DI-S", "TM-S", 
    "TS-S", "PNI-S", "IP-S", "BS-S", "PC-S", 
    "RG-S", "LS-C", "HA-C", "AM-C", "TR-C", 
    "CM-C", "OS-S", "FH-C", "HL-C", "NJ-S", 
    "YB-C", "ACO-C", "PM-C", "DL-C", "SE-C", 
    "TI-S", "DI-C", "PMU-S", "EZ-S", "CP-S", 
    "TM-C", "HL-S", "SC-S"
]
with_nota = True
print(f"#TWBIRD_LABELS: {len(TWBIRD_LABELS)}, NOTA: {with_nota}\n")

class MLP(nn.Module):
    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, out_features)

        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(128)

        self.drop = nn.Dropout(0.2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.drop(self.relu(self.bn1(self.fc1(x))))
        x = self.drop(self.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)
        return x

def parse_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument("-w", "--window_size", type=float, default=3.0)
    parser.add_argument("-hp", "--hop_length", type=float, default=0.5)

    # Model & Dataset
    parser.add_argument("--weight", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument("--mask_t_ratio", type=float, default=0.2)
    parser.add_argument("--mask_f_ratio", type=float, default=0.2)

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
    model.load_state_dict(torch.load(args.weight, map_location=DEVICE), strict=False)

    best_loss = np.inf
    Path(f"./results/{model_record}/{result_record}/train_loss.txt").unlink(missing_ok=True)
    Path(f"./results/{model_record}/{result_record}/val_loss.txt").unlink(missing_ok=True)
    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss = 0
        classifier.train()
        for _, (x, y) in tqdm(enumerate(train_dataloder), desc="Training", leave=False, total=len(train_dataloder)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            latent = model.forward_feature(x.float(), mask_col_ratio=args.mask_t_ratio, mask_row_ratio=args.mask_f_ratio)
            output = classifier(latent)

            optimizer.zero_grad()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        write_loss_history(
            epoch=epoch, loss=train_loss/len(train_dataloder),
            filename=f"./results/{model_record}/{result_record}/train_loss.txt"
        )

        val_loss = 0
        classifier.eval()
        for _, (x, y) in tqdm(enumerate(val_dataloder), desc="Validation", leave=False, total=len(val_dataloder)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            latent = model.forward_feature(x.float(), mask_col_ratio=args.mask_t_ratio, mask_row_ratio=args.mask_f_ratio)
            output = classifier(latent)

            loss = criterion(output, y)
            val_loss += loss.item()
        write_loss_history(
            epoch=epoch, loss=val_loss/len(val_dataloder),
            filename=f"./results/{model_record}/{result_record}/val_loss.txt"
        )
        if val_loss/len(val_dataloder) < best_loss:
            best_loss = val_loss/len(val_dataloder)
            torch.save(model.state_dict(), f"./weights/{model_record}/{result_record}/pretrain_best.pth")
            torch.save(classifier.state_dict(), f"./weights/{model_record}/{result_record}/classifier_best.pth")
        save_checkpoint(
            model, optimizer, epoch, best_loss,
            filename=f"./ckpt/{model_record}/{result_record}/best.pth"
        )

@torch.no_grad()
def inference(args):
    model.load_state_dict(torch.load(f"./weights/{model_record}/{result_record}/pretrain_best.pth", map_location=DEVICE))
    classifier.load_state_dict(torch.load(f"./weights/{model_record}/{result_record}/classifier_best.pth", map_location=DEVICE))

    classifier.eval()
    pred_species, true_species = [], []
    for _, (x, y) in tqdm(enumerate(test_dataloder), desc="Testing", total=len(test_dataloder)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        latent = model.forward_feature(x.float(), mask_col_ratio=args.mask_t_ratio, mask_row_ratio=args.mask_f_ratio)
        output = classifier(latent)

        # Multi-label classification
        thershold_prob = 1 / len(TWBIRD_LABELS)
        true_species.append((y > thershold_prob).cpu().numpy())
        pred_species.append((torch.sigmoid(output) > thershold_prob).cpu().numpy())
    
    true_species = np.concatenate(true_species, axis=0)
    pred_species = np.concatenate(pred_species, axis=0)
    np.savetxt(f"results/{model_record}/{result_record}/true_species.txt", true_species, fmt="%d")
    np.savetxt(f"results/{model_record}/{result_record}/pred_species.txt", pred_species, fmt="%d")
    if with_nota:
        report = classification_report(true_species, pred_species, target_names=TWBIRD_LABELS+["NOTA"], zero_division=0)
    else:
        report = classification_report(true_species, pred_species, target_names=TWBIRD_LABELS, zero_division=0)
    with open(f"results/{model_record}/{result_record}/classification_report.txt", "w") as f:
        f.write(report)        

if __name__ == "__main__":
    args = parse_args()

    model_record = \
        f"linearProbing/AudioMAE/mr_t{str(args.mask_t_ratio).replace('.', '')}f{str(args.mask_f_ratio).replace('.', '')}" + \
        f"/w{str(args.window_size).replace('.', '')}" + \
        f"_h{str(args.hop_length).replace('.', '')}"
    result_record = \
        f"lr{str(args.learning_rate).split('.')[-1]}_wd{str(args.weight_decay).split('.')[-1]}_b{args.batch_size}_e{args.epochs}"

    # pretrain model
    if args.window_size == 3.0 and args.hop_length == 0.5:
        in_shape = (1, 320, 128)
    elif args.window_size == 1.0 and args.hop_length == 0.5:
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
    model.to(DEVICE)
    classifier.to(DEVICE)

    # dataloader
    train_dataloder = DataLoader(
        TWBird(src_file="./data/finetune/train.txt", labeled=True, window_size=args.window_size, hop_length=args.hop_length), 
        batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    val_dataloder = DataLoader(
        TWBird(src_file="./data/finetune/val.txt", labeled=True, window_size=args.window_size, hop_length=args.hop_length), 
        batch_size=args.batch_size, num_workers=8, pin_memory=True
    )
    test_dataloder = DataLoader(
        TWBird(src_file="./data/finetune/test.txt", labeled=True, window_size=args.window_size, hop_length=args.hop_length), 
        batch_size=1, num_workers=4
    )

    # criterion & optimizer
    # freeze pretrain model
    for param in model.parameters():
        param.requires_grad = False
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    
    if args.train:
        Path(f"./results/{model_record}/{result_record}").mkdir(exist_ok=True, parents=True)
        Path(f"./weights/{model_record}/{result_record}").mkdir(exist_ok=True, parents=True)
        Path(f"./ckpt/{model_record}/{result_record}").mkdir(exist_ok=True, parents=True)
        with open(f"./results/{model_record}/{result_record}/gradient.txt", "w") as f:
            for name, param in model.named_parameters():
                f.write(f"Pretrain Parameter: {name}, Requires Grad: {param.requires_grad}\n")
            f.write("-"*67+"\n")
            for name, param in classifier.named_parameters():
                f.write(f"Finetune Parameter: {name}, Requires Grad: {param.requires_grad}\n")
        train(args)
        print(f"Finish training. {model_record}")
    if args.inference:
        inference(args)