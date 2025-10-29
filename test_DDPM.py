import os
import argparse

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from dataset_DDPM import Dataset
from model_DDPM import SimpleNoisePredictor, get_diffusion_params


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


def compute_iou(pred, gt, threshold=0.5):
    if pred.size(1) > 1:
        pred = pred.mean(dim=1, keepdim=True)
    if gt.size(1) > 1:
        gt = gt.mean(dim=1, keepdim=True)

    pred_bin = (pred > threshold).float()
    gt_bin = (gt > threshold).float()
    intersection = (pred_bin * gt_bin).view(pred.size(0), -1).sum(dim=1)
    union = (pred_bin + gt_bin - pred_bin * gt_bin).view(pred.size(0), -1).sum(dim=1)
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


def compute_rmse(pred, gt):
    mse = F.mse_loss(pred, gt, reduction="mean")
    return torch.sqrt(mse)


def sample_from_model(model, cond, betas, alphas, alphas_cumprod):
    model.eval()
    device = cond.device
    timesteps = betas.shape[0]

    alphas_cumprod_prev = torch.cat([
        torch.ones(1, device=device),
        alphas_cumprod[:-1],
    ])
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
    parser = argparse.ArgumentParser(description="Evaluate a conditional DDPM noise predictor")
    parser.add_argument("--data_root", type=str, default="data_net")
    parser.add_argument("--split_file", type=str, default="split/test.txt")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--timesteps", type=int, default=100, help="used when checkpoint does not store schedule")
    parser.add_argument("--results_dir", type=str, default="results/DDPM")
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
        shuffle=False,
        num_workers=4,
        collate_fn=custom_collate,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
        betas = checkpoint["betas"].to(device)
        alphas = checkpoint["alphas"].to(device)
        alphas_cumprod = checkpoint["alphas_cumprod"].to(device)
    else:
        state_dict = checkpoint
        betas, alphas, alphas_cumprod = get_diffusion_params(args.timesteps, device)

    model = SimpleNoisePredictor(channels=3)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    os.makedirs(args.results_dir, exist_ok=True)

    total_ssim = 0.0
    total_iou = 0.0
    total_rmse = 0.0
    num_batches = 0
    sample_idx = 0

    with torch.no_grad():
        for batch in dataloader:
            cond = batch["color"].to(device)
            target = batch["gt"].to(device)

            cond_norm = cond * 2 - 1
            samples = sample_from_model(model, cond_norm, betas, alphas, alphas_cumprod)
            outputs = torch.clamp((samples + 1) / 2, 0, 1)

            loss_ssim = ssim_loss(outputs, target)
            batch_ssim = 1 - 2 * loss_ssim
            batch_iou = compute_iou(outputs, target)
            batch_rmse = compute_rmse(outputs, target)

            total_ssim += batch_ssim.item()
            total_iou += batch_iou.item()
            total_rmse += batch_rmse.item()
            num_batches += 1

            for i in range(outputs.size(0)):
                save_path = os.path.join(args.results_dir, f"pred_{sample_idx:05d}.png")
                utils.save_image(outputs[i], save_path, normalize=True)
                sample_idx += 1

    avg_ssim = total_ssim / max(1, num_batches)
    avg_iou = total_iou / max(1, num_batches)
    avg_rmse = total_rmse / max(1, num_batches)

    print(f"Test Metrics: SSIM: {avg_ssim:.4f}, IoU: {avg_iou:.4f}, RMSE: {avg_rmse:.4f}")


if __name__ == "__main__":
    main()

