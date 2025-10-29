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


def compute_soft_iou(pred, gt, eps=1e-6):
    if pred.size(1) > 1:
        pred = pred.mean(dim=1, keepdim=True)
    if gt.size(1) > 1:
        gt = gt.mean(dim=1, keepdim=True)

    intersection = (pred * gt).view(pred.size(0), -1).sum(dim=1)
    union = (pred + gt - pred * gt).view(pred.size(0), -1).sum(dim=1)
    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def compute_rmse(pred, gt):
    mse = F.mse_loss(pred, gt, reduction="mean")
    return torch.sqrt(mse + 1e-8)


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
    parser.add_argument("--noise_weight", type=float, default=1.0)
    parser.add_argument("--l1_weight", type=float, default=1.0)
    parser.add_argument("--ssim_weight", type=float, default=0.1)
    parser.add_argument("--rmse_weight", type=float, default=0.0)
    parser.add_argument("--iou_weight", type=float, default=0.0)
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
    csv_writer.writerow(
        [
            "type",
            "global_step",
            "epoch",
            "batch_idx",
            "loss_total",
            "loss_noise",
            "loss_l1",
            "loss_ssim",
            "loss_rmse",
            "loss_iou",
            "metric_ssim",
            "metric_iou",
            "metric_rmse",
            "timestamp",
        ]
    )
    csv_file.flush()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_total = 0.0
        running_noise = 0.0
        running_l1 = 0.0
        running_ssim_loss = 0.0
        running_rmse_loss = 0.0
        running_iou_loss = 0.0
        running_ssim_metric = 0.0
        running_iou_metric = 0.0
        running_rmse_metric = 0.0

        for batch_idx, batch in enumerate(dataloader):
            cond = batch["color"].to(device)
            target = batch["gt"].to(device)

            cond = cond * 2 - 1
            target = target * 2 - 1

            t = torch.randint(0, args.timesteps, (cond.size(0),), device=device).long()

            noisy_target, noise = forward_diffusion_sample(target, t, alphas_cumprod, device)
            pred_noise = model(noisy_target, t, cond)

            loss_noise = F.mse_loss(pred_noise, noise)

            alphas_cumprod_t = alphas_cumprod[t].view(-1, 1, 1, 1)
            sqrt_alphas_cumprod_t = torch.sqrt(alphas_cumprod_t)
            sqrt_one_minus_cumprod_t = torch.sqrt(1.0 - alphas_cumprod_t)

            recon = (noisy_target - sqrt_one_minus_cumprod_t * pred_noise) / sqrt_alphas_cumprod_t
            recon = torch.clamp(recon, -1.0, 1.0)

            recon_01 = torch.clamp((recon + 1) / 2, 0, 1)
            target_01 = torch.clamp((target + 1) / 2, 0, 1)

            loss_l1 = F.l1_loss(recon_01, target_01)
            loss_ssim = ssim_loss(recon_01, target_01)
            rmse_val = compute_rmse(recon_01, target_01)
            loss_rmse = rmse_val
            soft_iou = compute_soft_iou(recon_01, target_01)
            loss_iou = 1.0 - soft_iou

            total_loss = (
                args.noise_weight * loss_noise
                + args.l1_weight * loss_l1
                + args.ssim_weight * loss_ssim
                + args.rmse_weight * loss_rmse
                + args.iou_weight * loss_iou
            )

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_size = cond.size(0)
            running_total += total_loss.item() * batch_size
            running_noise += loss_noise.item() * batch_size
            running_l1 += loss_l1.item() * batch_size
            running_ssim_loss += loss_ssim.item() * batch_size
            running_rmse_loss += loss_rmse.item() * batch_size
            running_iou_loss += loss_iou.item() * batch_size

            metric_ssim = 1.0 - 2.0 * loss_ssim.detach()
            running_ssim_metric += metric_ssim.item() * batch_size
            running_iou_metric += soft_iou.detach().item() * batch_size
            running_rmse_metric += rmse_val.detach().item() * batch_size

            writer.add_scalar("Loss/train_batch", total_loss.item(), global_step)
            writer.add_scalar("Loss/noise", loss_noise.item(), global_step)
            writer.add_scalar("Loss/l1", loss_l1.item(), global_step)
            writer.add_scalar("Loss/ssim", loss_ssim.item(), global_step)
            writer.add_scalar("Loss/rmse", loss_rmse.item(), global_step)
            writer.add_scalar("Loss/iou", loss_iou.item(), global_step)
            writer.add_scalar("Metric/SSIM", metric_ssim.item(), global_step)
            writer.add_scalar("Metric/IoU", soft_iou.detach().item(), global_step)
            writer.add_scalar("Metric/RMSE", rmse_val.detach().item(), global_step)
            csv_writer.writerow([
                "batch",
                global_step,
                epoch + 1,
                batch_idx,
                total_loss.item(),
                loss_noise.item(),
                loss_l1.item(),
                loss_ssim.item(),
                loss_rmse.item(),
                loss_iou.item(),
                metric_ssim.item(),
                soft_iou.detach().item(),
                rmse_val.detach().item(),
                datetime.datetime.now().isoformat(),
            ])
            csv_file.flush()
            global_step += 1

        dataset_size = len(dataset)
        epoch_loss = running_total / dataset_size
        epoch_noise = running_noise / dataset_size
        epoch_l1 = running_l1 / dataset_size
        epoch_ssim_loss = running_ssim_loss / dataset_size
        epoch_rmse_loss = running_rmse_loss / dataset_size
        epoch_iou_loss = running_iou_loss / dataset_size
        epoch_ssim_metric = running_ssim_metric / dataset_size
        epoch_iou_metric = running_iou_metric / dataset_size
        epoch_rmse_metric = running_rmse_metric / dataset_size

        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch + 1)
        writer.add_scalar("Loss/noise_epoch", epoch_noise, epoch + 1)
        writer.add_scalar("Loss/l1_epoch", epoch_l1, epoch + 1)
        writer.add_scalar("Loss/ssim_epoch", epoch_ssim_loss, epoch + 1)
        writer.add_scalar("Loss/rmse_epoch", epoch_rmse_loss, epoch + 1)
        writer.add_scalar("Loss/iou_epoch", epoch_iou_loss, epoch + 1)
        writer.add_scalar("Metric/SSIM_epoch", epoch_ssim_metric, epoch + 1)
        writer.add_scalar("Metric/IoU_epoch", epoch_iou_metric, epoch + 1)
        writer.add_scalar("Metric/RMSE_epoch", epoch_rmse_metric, epoch + 1)

        print(
            f"Epoch [{epoch + 1}/{args.epochs}] Loss: {epoch_loss:.6f} "
            f"(noise={epoch_noise:.6f}, l1={epoch_l1:.6f}, ssim={epoch_ssim_loss:.6f}, "
            f"rmse={epoch_rmse_loss:.6f}, iou={epoch_iou_loss:.6f}; "
            f"SSIM={epoch_ssim_metric:.4f}, IoU={epoch_iou_metric:.4f}, RMSE={epoch_rmse_metric:.4f})"
        )
        csv_writer.writerow([
            "epoch",
            global_step,
            epoch + 1,
            -1,
            epoch_loss,
            epoch_noise,
            epoch_l1,
            epoch_ssim_loss,
            epoch_rmse_loss,
            epoch_iou_loss,
            epoch_ssim_metric,
            epoch_iou_metric,
            epoch_rmse_metric,
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

