from itertools import chain
from pathlib import Path
import argparse
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

import numpy as np
from tqdm import trange, tqdm
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from sklearn.metrics import classification_report

from datasets.twbird import TWBird
from nets.models import MAE_Swin
from utils.record import write_loss_history
from utils.checkpoint import save_checkpoint, load_checkpoint, save_and_cleanup_weights

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
    # Model & Dataset
    parser.add_argument("--weight", type=str, required=True, help="Path to the pre-trained model")
    parser.add_argument(
        "--classifier", type=str, required=True,
        choices=["MLP"]
    )
    parser.add_argument("--mask_ratio", type=float, default=0.8)

    # Hyperparameters
    parser.add_argument("--epochs", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--weight_decay", type=float, default=0.0001)

    # Decision
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--inference", action="store_true")

    return parser.parse_args()

def train(args):
    model.load_state_dict(torch.load(args.weight, map_location=DEVICE), strict=False)

    Path(f"./results/{model_record}/{result_record}/train_loss.txt").unlink(missing_ok=True)
    Path(f"./results/{model_record}/{result_record}/val_loss.txt").unlink(missing_ok=True)
    best_loss = np.inf
    for epoch in trange(args.epochs, desc="Epoch"):
        train_loss = 0
        classifier.train()
        for _, (x, y) in tqdm(enumerate(train_dataloder), desc="Training", leave=False, total=len(train_dataloder)):
            x, y = x.to(DEVICE), y.to(DEVICE)
            latent = model.forward_feature(x.float(), mask_ratio=0.0)
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
            latent = model.forward_feature(x.float(), mask_ratio=0.0)
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
    pred_acts, true_acts = [], []
    for _, (x, y) in tqdm(enumerate(test_dataloder), desc="Testing", total=len(test_dataloder)):
        x, y = x.to(DEVICE), y.to(DEVICE)
        latent = model.forward_feature(x.float())
        output = classifier(latent)

        true_act = y.argmax(dim=1).cpu().tolist()
        pred_act = torch.softmax(output, dim=1).argmax(dim=1).cpu().tolist()
        true_acts.extend(true_act)
        pred_acts.extend(pred_act)

    if args.dataset in ["PAMAP2", "PAMAP2W", "PAMAP2C"]:
        labels = PAMAP2_LABEL
    elif args.dataset in ["USCHAD"]:
        labels = USCHAD_LABEL
    report = classification_report(true_acts, pred_acts, target_names=labels, zero_division=0)
    print(report)
    with open(f"results/{model_record}/{result_record}/classification_report.txt", "w") as f:
        f.write(report)        

if __name__ == "__main__":
    args = parse_args()
    model_record = f"finetune/AudioMAE/mr{str(args.mask_ratio).replace('.', '')}"
    result_record = f"lr{str(args.lr).split('.')[-1]}_wd{str(args.weight_decay).split('.')[-1]}_b{args.batch_size}_e{args.epochs}"

    # pretrain model
    model = MAE_Swin(
        in_shape=(1, 320, 128), patch_size=(16, 16),
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16
    ).to(DEVICE)
    classifier = MLP(in_features=768, out_features=)


    # select model
    model, classifier = select_model(args)
    model.to(DEVICE)
    classifier.to(DEVICE)

    # dataloader
    datasets = select_dataset(args)
    train_dataloder = DataLoader(datasets["train"], batch_size=args.batch_size, shuffle=True)
    val_dataloder = DataLoader(datasets["val"], batch_size=args.batch_size, shuffle=False)
    test_dataloder = DataLoader(datasets["test"], batch_size=1, shuffle=False)

    # criterion & optimizer
    combined_parameters = chain(model.parameters(), classifier.parameters())
    optimizer = torch.optim.AdamW(combined_parameters, lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
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



    

    # AudioMAE Base Version
    model = MAE_Swin(
        in_shape=(1, 320, 128), patch_size=(16, 16),
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16
    ).to(DEVICE)

    # multi-gpu
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # dataloader
    train_dataloder = DataLoader(TWBird(src_file="./data/pretrain/train.txt", labeled=False), batch_size=args.batch_size, num_workers=4, pin_memory=True)
    val_dataloder = DataLoader(TWBird(src_file="./data/pretrain/val.txt", labeled=False), batch_size=args.batch_size, num_workers=4, pin_memory=True)
    test_dataloder = DataLoader(TWBird(src_file="./data/pretrain/test.txt", labeled=False), batch_size=args.batch_size, num_workers=4, pin_memory=True)

    # optimizer & scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    if args.train:
        Path(f"./results/{model_record}/{result_record}").mkdir(exist_ok=True, parents=True)
        Path(f"/media/birdsong/disk02/bird-vocal-classification/weights/{model_record}/{result_record}").mkdir(exist_ok=True, parents=True)
        Path(f"/media/birdsong/disk02/bird-vocal-classification/ckpt/{model_record}/{result_record}").mkdir(exist_ok=True, parents=True)
        with open(f"./results/{model_record}/{result_record}/gradient.txt", "w") as f:
            for name, param in model.named_parameters():
                f.write(f"Pretrain Parameter: {name}, Requires Grad: {param.requires_grad}\n")
        train(args)
        print(f"Finish training. {model_record}")
    if args.inference:
        inference()