import os
import argparse
import csv
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

from dataset_CNN import Dataset
from model_CNN import CNNGenerator


def ssim_loss(img1, img2):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=3, stride=1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=3, stride=1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1_mu2

    ssim_n = (2 * mu1_mu2 + c1) * (2 * sigma12 + c2)
    ssim_d = (mu1_sq + mu2_sq + c1) * (sigma1_sq + sigma2_sq + c2)
    ssim_map = ssim_n / ssim_d

    loss = torch.clamp((1 - ssim_map) / 2, 0, 1)
    return loss.mean()


def custom_collate(batch):
    inputs = [sample[("color", -1)] for sample in batch]
    targets = []
    for sample in batch:
        for key, value in sample.items():
            if key[0] == "gt":
                targets.append(value)
                break
    inputs = torch.stack(inputs, dim=0)
    targets = torch.stack(targets, dim=0)
    return {"color": inputs, "gt": targets}


def main():
    parser = argparse.ArgumentParser(description="Train a CNN auto-encoder style model with L1 + SSIM loss")
    parser.add_argument("--data_root", type=str, default="data_net")
    parser.add_argument("--split_file", type=str, default="split/train.txt")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--save_folder", type=str, default="checkpoints/CNN/")
    parser.add_argument("--log_dir", type=str, default="tf-logs-cnn")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose([transforms.ToTensor()])

    dataset = Dataset(
        root_dir=args.data_root,
        split_file=args.split_file,
        height=args.height,
        width=args.width,
        transform=transform,
        target_transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=custom_collate,
    )

    model = CNNGenerator(in_channels=3, out_channels=3).to(device)
    l1_loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.save_folder, exist_ok=True)
    writer = SummaryWriter(logdir=args.log_dir)

    csv_path = os.path.join(args.save_folder, "losses.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "type",
        "global_step",
        "epoch",
        "batch_idx",
        "loss_total",
        "loss_l1",
        "loss_ssim",
        "timestamp",
    ])
    csv_file.flush()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            inputs = batch["color"].to(device)
            targets = batch["gt"].to(device)

            optimizer.zero_grad()
            outputs = model(inputs)

            loss_l1 = l1_loss_fn(outputs, targets)
            loss_ssim = ssim_loss(outputs, targets)
            loss = args.l1_weight * loss_l1 + args.ssim_weight * loss_ssim

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            csv_writer.writerow([
                "batch",
                global_step,
                epoch + 1,
                batch_idx,
                loss.item(),
                loss_l1.item(),
                loss_ssim.item(),
                datetime.datetime.now().isoformat(),
            ])
            csv_file.flush()
            global_step += 1

        epoch_loss = running_loss / len(dataset)
        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch + 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {epoch_loss:.4f}")
        csv_writer.writerow([
            "epoch",
            global_step,
            epoch + 1,
            -1,
            epoch_loss,
            "",
            "",
            datetime.datetime.now().isoformat(),
        ])
        csv_file.flush()

        model.eval()
        with torch.no_grad():
            sample = next(iter(dataloader))
            inputs = sample["color"].to(device)
            targets = sample["gt"].to(device)
            outputs = model(inputs)
            outputs = torch.clamp(outputs, 0, 1)

            grid_in = utils.make_grid(inputs, nrow=4, normalize=True)
            grid_out = utils.make_grid(outputs, nrow=4, normalize=True)
            grid_gt = utils.make_grid(targets, nrow=4, normalize=True)
            writer.add_image("Input", grid_in, epoch + 1)
            writer.add_image("Output", grid_out, epoch + 1)
            writer.add_image("GroundTruth", grid_gt, epoch + 1)

        if (epoch + 1) % max(1, args.save_interval) == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_folder, f"epoch_{epoch + 1}.pth")
            torch.save(model.state_dict(), ckpt_path)

    writer.close()
    csv_file.close()


if __name__ == "__main__":
    main()

