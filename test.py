import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms, utils

# 导入自定义的数据集和模型
from dataset import Dataset   # 文件名：myunet_dataset.py
from model import UNet4Layer      # 文件名：unet_model.py

def ssim_loss(img1, img2):
    """
    计算结构相似性（SSIM）损失，返回 (1 - SSIM)/2 的均值。
    假设 img1 和 img2 的形状均为 (N, C, H, W)，像素值范围 [0,1]
    """
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    mu1 = F.avg_pool2d(img1, kernel_size=3, stride=1, padding=1)
    mu2 = F.avg_pool2d(img2, kernel_size=3, stride=1, padding=1)

    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.avg_pool2d(img1 * img1, kernel_size=3, stride=1, padding=1) - mu1_sq
    sigma2_sq = F.avg_pool2d(img2 * img2, kernel_size=3, stride=1, padding=1) - mu2_sq
    sigma12 = F.avg_pool2d(img1 * img2, kernel_size=3, stride=1, padding=1) - mu1_mu2

    ssim_n = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    ssim_d = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim_map = ssim_n / ssim_d

    loss = torch.clamp((1 - ssim_map) / 2, 0, 1)
    return loss.mean()

def compute_iou(pred, gt, threshold=0.5):
    """
    计算 IoU (Intersection over Union)
    pred, gt: [N, C, H, W]；先将每个像素通过阈值转换为二值图像，再计算 IoU
    """
    # 如果多通道，先转为灰度
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
    """
    计算 RMSE (Root Mean Square Error)
    """
    mse = F.mse_loss(pred, gt, reduction='mean')
    return torch.sqrt(mse)

def custom_collate(batch):
    """
    自定义 collate 函数，将每个样本的字典合并成一个 batch。
    每个样本返回的字典中包含：
      ("color", -1): 输入图像张量
      ("gt", folder_id, -1): 真值图像张量（folder_id 各不相同）
    本函数将所有样本的输入图像堆叠到一起，真值图像也堆叠到一起，
    返回的字典键固定为 "color" 和 "gt"。
    """
    input_imgs = [sample[("color", -1)] for sample in batch]
    gt_imgs = []
    for sample in batch:
        for key, value in sample.items():
            if key[0] == "gt":
                gt_imgs.append(value)
                break
    input_imgs = torch.stack(input_imgs, dim=0)
    gt_imgs = torch.stack(gt_imgs, dim=0)
    return {"color": input_imgs, "gt": gt_imgs}

def main():
    parser = argparse.ArgumentParser(description="Test UNet on MyUNetDataset with SSIM, IoU and RMSE metrics")
    parser.add_argument("--data_root", type=str, default="data_net", help="数据根目录")
    parser.add_argument("--split_file", type=str, default="split/test.txt", help="测试集split文件")
    parser.add_argument("--height", type=int, default=256, help="图像高度")
    parser.add_argument("--width", type=int, default=256, help="图像宽度")
    parser.add_argument("--batch_size", type=int, default=8, help="batch大小")
    parser.add_argument("--checkpoint", type=str, required=True, help="加载的模型检查点路径")
    parser.add_argument("--results_dir", type=str, default="results/U-Net", help="保存预测图片的目录")
    args = parser.parse_args()

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 定义预处理操作（这里只转换为Tensor）
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 创建测试数据集和 DataLoader
    test_dataset = Dataset(root_dir=args.data_root,
                               split_file=args.split_file,
                               height=args.height,
                               width=args.width,
                               transform=transform,
                               target_transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=4, collate_fn=custom_collate)

    # 创建模型并加载检查点
    model = UNet4Layer(in_channels=3, out_channels=3)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model = model.to(device)
    model.eval()

    total_ssim = 0.0
    total_iou = 0.0
    total_rmse = 0.0
    num_batches = 0

    # 创建结果保存文件夹
    os.makedirs(args.results_dir, exist_ok=True)
    sample_count = 0

    with torch.no_grad():
        for batch in test_loader:
            input_img = batch["color"].to(device)
            gt_img = batch["gt"].to(device)
            output = model(input_img)
            # 将输出限定在 [0,1] 内
            output = torch.clamp(output, 0, 1)

            # 计算 SSIM：loss_ssim = (1-SSIM)/2，因此 SSIM = 1-2*loss_ssim
            loss_ssim = ssim_loss(output, gt_img)
            batch_ssim = 1 - 2 * loss_ssim

            batch_iou = compute_iou(output, gt_img, threshold=0.5)
            batch_rmse = compute_rmse(output, gt_img)

            total_ssim += batch_ssim.item()
            total_iou += batch_iou.item()
            total_rmse += batch_rmse.item()
            num_batches += 1

            # 保存每个样本的预测图片
            for i in range(output.size(0)):
                save_path = os.path.join(args.results_dir, f"pred_{sample_count:05d}.png")
                utils.save_image(output[i], save_path, normalize=True)
                sample_count += 1

        avg_ssim = total_ssim / num_batches
        avg_iou = total_iou / num_batches
        avg_rmse = total_rmse / num_batches

    print(f"Test Metrics: SSIM: {avg_ssim:.4f}, IoU: {avg_iou:.4f}, RMSE: {avg_rmse:.4f}")

if __name__ == "__main__":
    main()