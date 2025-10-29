import os
import argparse
import csv
import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tensorboardX import SummaryWriter

from dataset_DDPM import Dataset
from model_DDPM import SimpleNoisePredictor, get_diffusion_params, forward_diffusion_sample


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


def sample_from_model(model, cond, betas, alphas, alphas_cumprod):
    model.eval()
    device = cond.device
    timesteps = betas.shape[0]

    alphas_cumprod_prev = torch.cat(
        [torch.ones(1, device=device), alphas_cumprod[:-1]]
    )
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_one_minus_cumprod = torch.sqrt(1.0 - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    x = torch.randn_like(cond)
    for t in reversed(range(timesteps)):
        t_batch = torch.full((cond.size(0),), t, device=device, dtype=torch.long)
        beta_t = betas[t]
        sqrt_recip_alpha_t = sqrt_recip_alphas[t]
        sqrt_one_minus_cumprod_t = sqrt_one_minus_cumprod[t]

        model_output = model(x, t_batch, cond)
        model_mean = (
            sqrt_recip_alpha_t * (x - beta_t / sqrt_one_minus_cumprod_t * model_output)
        )

        if t > 0:
            noise = torch.randn_like(x)
            variance = posterior_variance[t]
            x = model_mean + torch.sqrt(variance) * noise
        else:
            x = model_mean

    model.train()
    return x


def main():
    parser = argparse.ArgumentParser(description="Train a conditional DDPM noise predictor")
    parser.add_argument("--data_root", type=str, default="data_net")
    parser.add_argument("--split_file", type=str, default="split/train.txt")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--timesteps", type=int, default=20)
    parser.add_argument("--save_interval", type=int, default=100)
    parser.add_argument("--save_folder", type=str, default="checkpoints/DDPM/")
    parser.add_argument("--log_dir", type=str, default="tf-logs-ddpm")
    parser.add_argument("--sample_interval", type=int, default=50, help="how often (epochs) to run ancestral sampling")
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

    model = SimpleNoisePredictor(channels=3).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    betas, alphas, alphas_cumprod = get_diffusion_params(args.timesteps, device)

    os.makedirs(args.save_folder, exist_ok=True)
    writer = SummaryWriter(logdir=args.log_dir)

    csv_path = os.path.join(args.save_folder, "losses.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["type", "global_step", "epoch", "batch_idx", "loss", "timestamp"])
    csv_file.flush()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch_idx, batch in enumerate(dataloader):
            cond = batch["color"].to(device)
            target = batch["gt"].to(device)

            cond = cond * 2 - 1
            target = target * 2 - 1

            t = torch.randint(0, args.timesteps, (cond.size(0),), device=device).long()

            noisy_target, noise = forward_diffusion_sample(target, t, alphas_cumprod, device)
            pred_noise = model(noisy_target, t, cond)

            loss = F.mse_loss(pred_noise, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * cond.size(0)
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            csv_writer.writerow([
                "batch",
                global_step,
                epoch + 1,
                batch_idx,
                loss.item(),
                datetime.datetime.now().isoformat(),
            ])
            csv_file.flush()
            global_step += 1

        epoch_loss = running_loss / len(dataset)
        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch + 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}] Loss: {epoch_loss:.6f}")
        csv_writer.writerow([
            "epoch",
            global_step,
            epoch + 1,
            -1,
            epoch_loss,
            datetime.datetime.now().isoformat(),
        ])
        csv_file.flush()

        if (epoch + 1) % max(1, args.save_interval) == 0 or (epoch + 1) == args.epochs:
            ckpt_path = os.path.join(args.save_folder, f"epoch_{epoch + 1}.pth")
            torch.save({
                "model": model.state_dict(),
                "betas": betas.cpu(),
                "alphas": alphas.cpu(),
                "alphas_cumprod": alphas_cumprod.cpu(),
            }, ckpt_path)

        if (epoch + 1) % max(1, args.sample_interval) == 0:
            with torch.no_grad():
                sample_batch = next(iter(dataloader))
                cond = sample_batch["color"].to(device)
                cond_norm = cond * 2 - 1
                samples = sample_from_model(model, cond_norm, betas, alphas, alphas_cumprod)
                samples = torch.clamp((samples + 1) / 2, 0, 1)
                cond_grid = utils.make_grid(cond, nrow=4, normalize=True)
                sample_grid = utils.make_grid(samples, nrow=4, normalize=True)
                writer.add_image("Cond", cond_grid, epoch + 1)
                writer.add_image("DDPM_Sample", sample_grid, epoch + 1)

    writer.close()
    csv_file.close()


if __name__ == "__main__":
    main()

