工作总结

一、数据标注
    LabelMe 标注，保留每一对原始图像与对应掩码，重点修改了TRO字母的标注错误问题和六边形标注不明显的问题。
    原始数据分为12组：
    1：骨头 2：石头 3：海螺 4：树枝 5：螺丝刀头 6：螺母 7：平行线 8：放射线 9：TRO 10：小点 11：六边形 12：大圆


二、数据增强
    数据增强由 `scripts/augment.py` 完整实现，脚本通过读取每个分组目录下的 RGB 图片与对应的 LabelMe JSON 文件，将 JSON 中的多边形使用 PIL 的 ImageDraw rasterize 成单通道二值掩码（'L'，值域 0 或 255），并将原始对以 `<group>/images/<base>_orig.png` 与 `<group>/gt/<base>_orig.png` 的形式写入输出目录以保证“保留原始数据”。在此基础上，每张原图会生成若干增强样本（由命令行参数 `--n_per_image` 控制，默认 5），增强变换包括按概率的水平翻转（约 50%）、少量垂直翻转（约 20%）、仿射变换（旋转角度随机在 ±25°、平移幅度上限为图像宽高的 10%、缩放因子在 0.9–1.1 之间、以及小幅 shear，脚本优先使用 `torchvision.transforms.functional.affine` 保证对图像使用双线性插值而对掩码使用最近邻插值以保持二值性；当 torchvision 不可用时回退到 PIL 的旋转等基本变换），色彩增强包括随机亮度（0.8–1.2）、对比度（0.8–1.2）与色彩饱和度（0.8–1.3）扰动，另有概率添加高斯噪声（sigma 在 3–12 之间）以增强鲁棒性，所有增强样本与其掩码按 `augmented/<group>/images` 与 `augmented/<group>/gt` 结构保存。

三、数据划分
    数据划分使用两套脚本管理：`scripts/create_splits.py` 提供单次划分功能并将结果写入带时间戳的子目录以避免覆盖（例如 `run_YYYYmmdd_HHMMSS_s<seed>/train.txt, test.txt`），而 `scripts/create_custom_splits.py` 支持批量生成多组划分（默认 20 组），其约定是强制包含组 '7' 和 '9'、排除组 '6'，然后从剩余组中随机选择 4 个与之合并构成 6 个训练组，训练集与测试集目标样本数由参数 `train_total` 与 `test_total` 指定（常用为 1000），采样策略为按训练组平均抽样（若某组样本不足则取全部，若抽样过多则随机截断到目标大小），测试集从全部 12 组均匀抽样并截断到目标大小，输出为相对路径（示例 `12/images/00001.png`）写入 `train.txt` 与 `test.txt`。

四、数据加载
    数据加载器位于 `dataset.py`，按 split 文件逐行读取相对路径并构建输入路径（`root_dir + rel_path`），输入以 LANCZOS 重采样 resize 到指定尺寸并转换为 Tensor，GT 路径按规则从 `group/images/...` 映射到 `group/gt/...`，若第一种映射不存在则尝试备用拼接（提高兼容性），每个样本返回一个字典包含键 `('color', -1)` 对应输入张量与键 `('gt', -1)` 对应目标张量，训练/测试批次通过自定义 `collate` 将这些键合并为标准的 batch 字典，其中 'color' 会被堆叠为形状 (N, C, H, W) 的输入张量、'gt' 会被堆叠为形状 (N, C, H, W) 的目标张量；

五、模型训练与调整
    模型定义在 `model.py` 中，类名为 `UNet4Layer`，采用 4 层编码器-解码器结构：编码器由四个下采样模块组成，每个模块含两次 3x3 卷积 + BatchNorm + ReLU，之后下采样使用 2x2 最大池化；瓶颈为两层卷积块，解码器通过转置卷积上采样并与对应的 skip connection 拼接后再做两次卷积块恢复空间分辨率，最终使用 1x1 卷积映射到输出通道（默认 in_channels=3, out_channels=3）。

    训练逻辑由 `train.py` 实现，使用 Adam 优化器（默认学习率 lr=1e-4），损失由像素级 L1Loss 与自定义 SSIM 损失构成：代码中实现的 `ssim_loss` 基于局部均值和方差估计（使用 3x3 平均池化），返回 (1-SSIM)/2 的均值作为损失项，整体训练损失为 loss = l1_weight * L1 + ssim_weight * SSIM，默认权重 l1_weight=1.0、ssim_weight=0.1，可通过命令行参数调整；训练参数如 batch_size、epochs、学习率、保存间隔等均可由 CLI 控制，训练脚本在每个 batch 与每个 epoch 将损失写入 `<save_folder>/losses.csv`（字段包括 type、global_step、epoch、batch_idx、loss_total、loss_l1、loss_ssim、timestamp），同时把标量与样本网格写入 tensorboardX 的 `--log_dir`。模型保存由 `--save_interval` 控制（每隔 N 个 epoch 保存一次，脚本也会在最后一个 epoch 强制保存），默认保存目录为 `--save_folder`（默认 `checkpoints/U-Net/`，批量实验时通常写到 `checkpoints/custom_run_*/`），在大规模实验中也实现了检查点修整策略（例如按要求删除非 20 的倍数 checkpoint，仅保留 epoch_20, epoch_40, ...）以节省磁盘空间。评估由 `test.py` 完成：测试时模型输出首先 clamp 到 [0,1]，SSIM 由训练中的 `ssim_loss` 反算为 SSIM 值（SSIM = 1 - 2 * loss_ssim），IoU 的计算先对多通道输出按通道平均得到单通道预测，再以阈值 0.5 二值化预测与 GT 后计算交并比（对 batch 求平均），RMSE 则为均方误差的平方根，脚本会将每个预测按顺序保存为归一化图像到 `--results_dir` 指定的目录。为方便大规模、多次划分的实验，本仓库还包含调度脚本 `scripts/run_custom_exps.py`（顺序训练 `split_runs_custom` 下的每个 run、将训练 stdout 写入 `tf-logs/custom_<run>/train_stdout.txt`、将每次训练生成的 `losses.csv` 归档并追加到 `all_losses.csv`，并在训练后自动调用 `test.py` 汇总评估结果到 `exp_results_custom.csv`），以及更通用的 `scripts/exp_runner.py` 用于网格化实验。最终结果管理包含将所有 `results_*`、预览压缩包等统一移动到顶层 `results/`，并根据 IoU 从每个 run 的最终 checkpoint 中选取最优模型复制到 `best_model/`，以便后续部署或更深入分析。总体而言，本仓库保证了原始数据的保留与可回溯性、提供可配置的数据增强与划分策略、实现了基于 L1+SSIM 的联合损失训练流程、并通过 CSV 与 TensorBoard 的双重记录保证训练可追溯性，同时提供了批量实验的自动化运行脚本与磁盘管理策略。

初次训练分别采用6、7、8组数据进行训练，每种数量随机取五次进行训练，取得的训练结果如图所示：
![results preview](results1.png)
通过对比结果，注意到6组训练已经可以取得不错的结果。另外，如果没有TRO以及四条平行线进入训练集，则测试结果不佳，而螺母却不放入训练集效果更好。
取TRO和平行线进入数据，同时再螺母螺母以外的数据取一共6组进行训练，随机取20次进行训练，得到的结果如下所示。可以发现2、3、5、7、9，12的效果最好。


训练20组给出的原始数据：（loss数据已经保存在csv里面）
Testing run_20251025_165702_01 epoch_200.pth
-> run_20251025_165702_01 SSIM= 0.9659 IoU= 0.8042 RMSE= 0.0699
Testing run_20251025_165702_02 epoch_200.pth
-> run_20251025_165702_02 SSIM= 0.9688 IoU= 0.7748 RMSE= 0.0815
Testing run_20251025_165702_03 epoch_200.pth
-> run_20251025_165702_03 SSIM= 0.9625 IoU= 0.8137 RMSE= 0.0682
Testing run_20251025_165702_04 epoch_200.pth
-> run_20251025_165702_04 SSIM= 0.9673 IoU= 0.8026 RMSE= 0.0761
Testing run_20251025_165702_05 epoch_200.pth
-> run_20251025_165702_05 SSIM= 0.9646 IoU= 0.8018 RMSE= 0.0728
Testing run_20251025_165702_06 epoch_200.pth
-> run_20251025_165702_06 SSIM= 0.9686 IoU= 0.8054 RMSE= 0.0709
Testing run_20251025_165702_07 epoch_200.pth
-> run_20251025_165702_07 SSIM= 0.971 IoU= 0.7547 RMSE= 0.0857
Testing run_20251025_165702_08 epoch_200.pth
-> run_20251025_165702_08 SSIM= 0.9535 IoU= 0.7669 RMSE= 0.0829
Testing run_20251025_165702_09 epoch_200.pth
-> run_20251025_165702_09 SSIM= 0.9556 IoU= 0.8032 RMSE= 0.0743
Testing run_20251025_165702_10 epoch_200.pth
-> run_20251025_165702_10 SSIM= 0.9728 IoU= 0.7907 RMSE= 0.0758
Testing run_20251025_165702_11 epoch_200.pth
-> run_20251025_165702_11 SSIM= 0.9692 IoU= 0.7928 RMSE= 0.076
Testing run_20251025_165702_12 epoch_200.pth
-> run_20251025_165702_12 SSIM= 0.9733 IoU= 0.8157 RMSE= 0.0691
Testing run_20251025_165702_13 epoch_200.pth
-> run_20251025_165702_13 SSIM= 0.9676 IoU= 0.802 RMSE= 0.0734
Testing run_20251025_165702_14 epoch_200.pth
-> run_20251025_165702_14 SSIM= 0.9711 IoU= 0.7794 RMSE= 0.0811
Testing run_20251025_165702_15 epoch_200.pth
-> run_20251025_165702_15 SSIM= 0.9607 IoU= 0.7982 RMSE= 0.0728
Testing run_20251025_165702_16 epoch_200.pth
-> run_20251025_165702_16 SSIM= 0.9715 IoU= 0.8131 RMSE= 0.0686
Testing run_20251025_165702_17 epoch_200.pth
-> run_20251025_165702_17 SSIM= 0.9667 IoU= 0.731 RMSE= 0.0945
Testing run_20251025_165702_18 epoch_200.pth
-> run_20251025_165702_18 SSIM= 0.9632 IoU= 0.7804 RMSE= 0.0785
Testing run_20251025_165702_19 epoch_200.pth
-> run_20251025_165702_19 SSIM= 0.9674 IoU= 0.8026 RMSE= 0.0709
Testing run_20251025_165702_20 epoch_200.pth
-> run_20251025_165702_20 SSIM= 0.9697 IoU= 0.8036 RMSE= 0.073

Top 5:
('run_20251025_165702_12', 'checkpoints/custom_run_20251025_165702_12/epoch_200.pth', 0.9733, 0.8157, 0.0691)
('run_20251025_165702_03', 'checkpoints/custom_run_20251025_165702_03/epoch_200.pth', 0.9625, 0.8137, 0.0682)
('run_20251025_165702_16', 'checkpoints/custom_run_20251025_165702_16/epoch_200.pth', 0.9715, 0.8131, 0.0686)
('run_20251025_165702_06', 'checkpoints/custom_run_20251025_165702_06/epoch_200.pth', 0.9686, 0.8054, 0.0709)
('run_20251025_165702_01', 'checkpoints/custom_run_20251025_165702_01/epoch_200.pth', 0.9659, 0.8042, 0.0699)


附：运行说明
1) augment.py — 生成掩码并做数据增强（保留原始对）
python3 scripts/augment.py --input_dir biao --output_dir augmented --n_per_image 5
2) create_splits.py — 生成一次性划分（时间戳子目录，避免覆盖）
python3 scripts/create_splits.py --input_dir augmented --out_dir split --train_groups_count 6 --train_frac 0.6 
3) create_custom_splits.py — 批量生成多组划分（支持指定包含/排除规则）
python3 scripts/create_custom_splits.py --input_dir augmented --out_dir split_runs_custom --n_splits 20 --train_total 1000 --test_total 1000 --seed 100
4) train.py — 训练一个模型（单次运行）
python3 train.py --data_root augmented --split_file split_runs_custom/run_20251025_165702_12/train.txt --batch_size 32 --epochs 200 --lr 1e-4 --save_interval 40 --save_folder checkpoints/custom_run_20251025_165702_12 --log_dir tf-logs/custom_run_20251025_165702_12
5) test.py — 对单个 checkpoint 做评估并保存预测图
python3 test.py --data_root augmented --split_file split_runs_custom/run_20251025_165702_12/test.txt --checkpoint checkpoints/custom_run_20251025_165702_12/epoch_200.pth --results_dir results/custom_run_20251025_165702_12
6) run_custom_exps.py — 顺序训练并评估 `split_runs_custom` 下的所有 run（自动化流程）
python3 scripts/run_custom_exps.py
7) exp_runner.py — 更通用的实验调度（用于网格或自定义配置）
python3 scripts/exp_runner.py --config <config.yaml>

新增：CNN 与 DDPM 流程示例（保持与 UNet 相同的目录结构与 split 文件格式）

1) CNN 训练
python3 train_CNN.py --data_root augmented --split_file split_runs_custom/run_xxx/train.txt --save_folder checkpoints/CNN_run --log_dir tf-logs/CNN_run

2) CNN 测试
python3 test_CNN.py --data_root augmented --split_file split_runs_custom/run_xxx/test.txt --checkpoint checkpoints/CNN_run/epoch_100.pth --results_dir results/CNN_run

3) DDPM 训练
python3 train_DDPM.py --data_root augmented --split_file split_runs_custom/run_xxx/train.txt --save_folder checkpoints/DDPM_run --log_dir tf-logs/DDPM_run

4) DDPM 测试
python3 test_DDPM.py --data_root augmented --split_file split_runs_custom/run_xxx/test.txt --checkpoint checkpoints/DDPM_run/epoch_200.pth --results_dir results/DDPM_run
