import argparse
import csv
import datetime
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

from dataset_CycleGAN import CycleGANDataset
from model_CycleGAN import build_discriminator, build_generator


def set_requires_grad(models, requires_grad: bool) -> None:
    if not isinstance(models, (list, tuple)):
        models = [models]
    for model in models:
        for param in model.parameters():
            param.requires_grad = requires_grad


def gan_loss(prediction: torch.Tensor, target_is_real: bool) -> torch.Tensor:
    target = torch.ones_like(prediction) if target_is_real else torch.zeros_like(prediction)
    return nn.MSELoss()(prediction, target)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CycleGAN model")
    parser.add_argument("--data_root", type=str, default="data_net")
    parser.add_argument("--split_a", type=str, default="split/trainA.txt")
    parser.add_argument("--split_b", type=str, default="split/trainB.txt")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--beta1", type=float, default=0.5)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--lambda_cycle", type=float, default=10.0)
    parser.add_argument("--lambda_identity", type=float, default=5.0)
    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--save_folder", type=str, default="checkpoints/CycleGAN/")
    parser.add_argument("--log_dir", type=str, default="tf-logs-cyclegan")
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    transform = transforms.Compose(
        [
            transforms.Resize((args.height, args.width)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    dataset = CycleGANDataset(
        root_dir=args.data_root,
        split_file_a=args.split_a,
        split_file_b=args.split_b,
        transform_a=transform,
        transform_b=transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    netG_AB = build_generator(3, 3)
    netG_BA = build_generator(3, 3)
    netD_A = build_discriminator(3)
    netD_B = build_discriminator(3)

    netG_AB.to(device)
    netG_BA.to(device)
    netD_A.to(device)
    netD_B.to(device)

    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    optimizer_G = optim.Adam(
        list(netG_AB.parameters()) + list(netG_BA.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )
    optimizer_D = optim.Adam(
        list(netD_A.parameters()) + list(netD_B.parameters()),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
    )

    os.makedirs(args.save_folder, exist_ok=True)
    writer = SummaryWriter(logdir=args.log_dir)

    csv_path = os.path.join(args.save_folder, "losses.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(
            [
                "type",
                "global_step",
                "epoch",
                "batch_idx",
                "loss_G",
                "loss_D",
                "loss_GAN",
                "loss_cycle",
                "loss_identity",
                "timestamp",
            ]
        )

    global_step = 0
    for epoch in range(args.epochs):
        loss_g_running = 0.0
        loss_d_running = 0.0
        for batch_idx, batch in enumerate(dataloader):
            real_A = batch["A"].to(device)
            real_B = batch["B"].to(device)

            # --------------------
            #  Train Generators
            # --------------------
            set_requires_grad([netD_A, netD_B], False)
            optimizer_G.zero_grad()

            idt_A = netG_BA(real_A)
            idt_B = netG_AB(real_B)
            loss_idt_A = criterion_identity(idt_A, real_A) * args.lambda_identity
            loss_idt_B = criterion_identity(idt_B, real_B) * args.lambda_identity

            fake_B = netG_AB(real_A)
            pred_fake_B = netD_B(fake_B)
            loss_G_AB = gan_loss(pred_fake_B, True)

            fake_A = netG_BA(real_B)
            pred_fake_A = netD_A(fake_A)
            loss_G_BA = gan_loss(pred_fake_A, True)

            rec_A = netG_BA(fake_B)
            rec_B = netG_AB(fake_A)
            loss_cycle_A = criterion_cycle(rec_A, real_A) * args.lambda_cycle
            loss_cycle_B = criterion_cycle(rec_B, real_B) * args.lambda_cycle

            loss_G = (
                loss_G_AB
                + loss_G_BA
                + loss_cycle_A
                + loss_cycle_B
                + loss_idt_A
                + loss_idt_B
            )
            loss_G.backward()
            optimizer_G.step()

            # --------------------
            #  Train Discriminators
            # --------------------
            set_requires_grad([netD_A, netD_B], True)
            optimizer_D.zero_grad()

            # Discriminator A
            pred_real_A = netD_A(real_A)
            loss_D_real_A = gan_loss(pred_real_A, True)
            pred_fake_A = netD_A(fake_A.detach())
            loss_D_fake_A = gan_loss(pred_fake_A, False)
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5

            # Discriminator B
            pred_real_B = netD_B(real_B)
            loss_D_real_B = gan_loss(pred_real_B, True)
            pred_fake_B = netD_B(fake_B.detach())
            loss_D_fake_B = gan_loss(pred_fake_B, False)
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5

            loss_D = loss_D_A + loss_D_B
            loss_D.backward()
            optimizer_D.step()

            loss_g_running += loss_G.item() * real_A.size(0)
            loss_d_running += loss_D.item() * real_A.size(0)

            writer.add_scalar("Loss/G", loss_G.item(), global_step)
            writer.add_scalar("Loss/D", loss_D.item(), global_step)
            writer.add_scalar(
                "Loss/GAN",
                (loss_G_AB + loss_G_BA).item(),
                global_step,
            )
            writer.add_scalar(
                "Loss/Cycle",
                (loss_cycle_A + loss_cycle_B).item(),
                global_step,
            )
            writer.add_scalar(
                "Loss/Identity",
                (loss_idt_A + loss_idt_B).item(),
                global_step,
            )

            with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(
                    [
                        "batch",
                        global_step,
                        epoch + 1,
                        batch_idx,
                        loss_G.item(),
                        loss_D.item(),
                        (loss_G_AB + loss_G_BA).item(),
                        (loss_cycle_A + loss_cycle_B).item(),
                        (loss_idt_A + loss_idt_B).item(),
                        datetime.datetime.now().isoformat(),
                    ]
                )

            global_step += 1

        avg_loss_g = loss_g_running / len(dataloader.dataset)
        avg_loss_d = loss_d_running / len(dataloader.dataset)
        writer.add_scalar("Loss/G_epoch", avg_loss_g, epoch + 1)
        writer.add_scalar("Loss/D_epoch", avg_loss_d, epoch + 1)
        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Loss_G: {avg_loss_g:.4f} Loss_D: {avg_loss_d:.4f}"
        )

        with open(csv_path, "a", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                [
                    "epoch",
                    global_step,
                    epoch + 1,
                    -1,
                    avg_loss_g,
                    avg_loss_d,
                    "",
                    "",
                    "",
                    datetime.datetime.now().isoformat(),
                ]
            )

        # log image grids
        netG_AB.eval()
        netG_BA.eval()
        with torch.no_grad():
            sample = next(iter(dataloader))
            real_A = sample["A"].to(device)
            real_B = sample["B"].to(device)
            fake_B = netG_AB(real_A)
            fake_A = netG_BA(real_B)

            def denorm(x: torch.Tensor) -> torch.Tensor:
                return (x * 0.5) + 0.5

            grid_real_A = utils.make_grid(denorm(real_A), normalize=True)
            grid_real_B = utils.make_grid(denorm(real_B), normalize=True)
            grid_fake_B = utils.make_grid(denorm(fake_B), normalize=True)
            grid_fake_A = utils.make_grid(denorm(fake_A), normalize=True)

            writer.add_image("A/real", grid_real_A, epoch + 1)
            writer.add_image("B/real", grid_real_B, epoch + 1)
            writer.add_image("A/to_B", grid_fake_B, epoch + 1)
            writer.add_image("B/to_A", grid_fake_A, epoch + 1)

        netG_AB.train()
        netG_BA.train()

        if (epoch + 1) % max(1, args.save_interval) == 0 or (epoch + 1) == args.epochs:
            torch.save(netG_AB.state_dict(), os.path.join(args.save_folder, f"netG_AB_epoch_{epoch + 1}.pth"))
            torch.save(netG_BA.state_dict(), os.path.join(args.save_folder, f"netG_BA_epoch_{epoch + 1}.pth"))
            torch.save(netD_A.state_dict(), os.path.join(args.save_folder, f"netD_A_epoch_{epoch + 1}.pth"))
            torch.save(netD_B.state_dict(), os.path.join(args.save_folder, f"netD_B_epoch_{epoch + 1}.pth"))

    writer.close()


if __name__ == "__main__":
    main()
