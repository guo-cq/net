import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, utils
import torch.nn.functional as F
from tensorboardX import SummaryWriter  # 使用 tensorboardX 的 SummaryWriter
import csv
import datetime

# 导入自定义的数据集和模型
from dataset import Dataset   # 文件名：myunet_dataset.py
from model import UNet4Layer      # 文件名：unet_model.py

def ssim_loss(img1, img2):
    """
    计算结构相似性（SSIM）损失，返回 (1 - SSIM)/2 的均值作为损失值。
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

def custom_collate(batch):
    """
    自定义 collate 函数，将每个样本的字典合并成一个 batch。
    每个样本返回的字典中包含：
      ("color", -1): 输入图像张量
      ("gt", folder_id, -1): 真值图像张量（folder_id 各不相同）
    本函数将所有样本的输入图像堆叠到一起，真值图像也堆叠到一起，
    返回的字典键固定为 "color" 和 "gt"。
    """
    # 收集所有输入图像
    input_imgs = [sample[("color", -1)] for sample in batch]
    # 收集所有真值图像（每个样本只有一个 "gt" 键）
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
    parser = argparse.ArgumentParser(description="Train UNet on MyUNetDataset with L1 + SSIM loss using tensorboardX")
    parser.add_argument("--data_root", type=str, default="data_net", help="根目录，包含所有数据")
    parser.add_argument("--split_file", type=str, default="split/train.txt", help="训练集split文件")
    parser.add_argument("--height", type=int, default=256, help="图像高度")
    parser.add_argument("--width", type=int, default=256, help="图像宽度")
    parser.add_argument("--batch_size", type=int, default=32, help="batch大小")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-4, help="学习率")
    parser.add_argument("--save_interval", type=int, default=20, help="每隔多少个 epoch 保存一次模型（默认20）")
    parser.add_argument("--save_folder", type=str, default="checkpoints/U-Net/", help="模型保存路径")
    parser.add_argument("--l1_weight", type=float, default=1.0, help="L1 loss权重")
    parser.add_argument("--ssim_weight", type=float, default=0.1, help="SSIM loss权重")
    parser.add_argument("--log_dir", type=str, default="tf-logs-unet", help="TensorBoardX日志目录")
    args = parser.parse_args()

    # 选择设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # 定义预处理操作（这里只转换为Tensor，可根据需要添加 Normalize 等）
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # 创建数据集和 DataLoader，使用自定义 collate 函数
    train_dataset = Dataset(root_dir=args.data_root,
                                split_file=args.split_file,
                                height=args.height,
                                width=args.width,
                                transform=transform,
                                target_transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=custom_collate)

    # 创建模型并转移到设备
    model = UNet4Layer(in_channels=3, out_channels=3)
    model = model.to(device)

    # 定义损失函数和优化器
    l1_loss_fn = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # 创建模型保存文件夹
    os.makedirs(args.save_folder, exist_ok=True)

    # 创建 TensorBoardX SummaryWriter
    writer = SummaryWriter(logdir=args.log_dir)

    # 打开 CSV 文件记录训练过程中的 loss（每个 batch 与每个 epoch 汇总）
    os.makedirs(args.save_folder, exist_ok=True)
    losses_csv = os.path.join(args.save_folder, 'losses.csv')
    loss_file = open(losses_csv, 'w', newline='', encoding='utf-8')
    loss_writer = csv.writer(loss_file)
    # header: type,batch/global_step,epoch,batch_idx,loss_total,loss_l1,loss_ssim,timestamp
    loss_writer.writerow(['type', 'global_step', 'epoch', 'batch_idx', 'loss_total', 'loss_l1', 'loss_ssim', 'timestamp'])
    loss_file.flush()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            input_img = batch["color"].to(device)
            gt_img = batch["gt"].to(device)

            optimizer.zero_grad()
            output = model(input_img)

            loss_l1 = l1_loss_fn(output, gt_img)
            loss_ssim = ssim_loss(output, gt_img)
            loss = args.l1_weight * loss_l1 + args.ssim_weight * loss_ssim

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * input_img.size(0)
            writer.add_scalar("Loss/train_batch", loss.item(), global_step)
            # 记录 batch 级别的 loss 到 CSV
            try:
                loss_writer.writerow(['batch', global_step, epoch+1, 0, loss.item(), loss_l1.item(), loss_ssim.item(), datetime.datetime.now().isoformat()])
                loss_file.flush()
            except Exception:
                pass
            global_step += 1

        epoch_loss = running_loss / len(train_dataset)
        print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {epoch_loss:.4f}")
        writer.add_scalar("Loss/train_epoch", epoch_loss, epoch+1)
        # 记录 epoch 级别的汇总 loss
        try:
            loss_writer.writerow(['epoch', global_step, epoch+1, -1, epoch_loss, '', '', datetime.datetime.now().isoformat()])
            loss_file.flush()
        except Exception:
            pass

        # 每个 epoch 结束后，展示一个 batch 的预测结果
        model.eval()
        with torch.no_grad():
            sample = next(iter(train_loader))
            sample_input = sample["color"].to(device)
            sample_gt = sample["gt"].to(device)
            sample_output = model(sample_input)
            # 对模型输出进行 clamping 处理（若输出不在 [0,1] 内）
            sample_output = torch.clamp(sample_output, 0, 1)
            
            grid_input = utils.make_grid(sample_input, nrow=4, normalize=True)
            grid_output = utils.make_grid(sample_output, nrow=4, normalize=True)
            grid_gt = utils.make_grid(sample_gt, nrow=4, normalize=True)
            writer.add_image("Input", grid_input, epoch+1)
            writer.add_image("Output", grid_output, epoch+1)
            writer.add_image("GroundTruth", grid_gt, epoch+1)

        # 保存模型检查点：每隔 args.save_interval 个 epoch 保存一次，或保存最后一个 epoch
        save_interval = max(1, args.save_interval)
        is_last = (epoch + 1) == args.epochs
        if ((epoch + 1) % save_interval == 0) or is_last:
            checkpoint_path = os.path.join(args.save_folder, f"epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)

    writer.close()
    try:
        loss_file.close()
    except Exception:
        pass

if __name__ == "__main__":
    main()