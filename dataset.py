import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import os
from PIL import Image
import torch
from torch.utils.data import Dataset

import os
from PIL import Image
import torch
from torch.utils.data import Dataset

class Dataset(Dataset):
    """
    用于UNet训练的Dataset示例：
      - 从 split_file 中读取每个图像的相对路径，例如 "12/500Hz/raw/h/00206.png"
      - 输入图像路径: root_dir + rel_path
      - 对应真值图像: 根据每行文件最开头的 id 对应的路径:
            root_dir/<id>/gt/shape.png
      - 返回的样本是一个字典，包含:
            ("color", -1): 输入图像张量
            ("gt", folder_id, -1): 真值图像张量，其中 folder_id 为最前面的 id（转换为整数，如果可能）
    参数:
        root_dir: 数据根目录（例如 data_net）
        split_file: 包含图像相对路径的文本文件（如 train.txt/val.txt/test.txt）
        height, width: 输出图像尺寸
        transform: 对输入图像的预处理操作
        target_transform: 对真值图像的预处理操作
    """
    def __init__(self, root_dir, split_file, height, width, transform=None, target_transform=None):
        super(Dataset, self).__init__()
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.transform = transform
        self.target_transform = target_transform

        # 读取 split_file 中的所有相对路径
        self.filenames = []
        with open(split_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.filenames.append(line)
                    
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, index):
        
        rel_path = self.filenames[index]

        # 输入图像路径
        input_path = os.path.join(self.root_dir, rel_path)
        input_img = Image.open(input_path).convert("RGB")
        input_img = input_img.resize((self.width, self.height), Image.LANCZOS)

        # 构造 GT 路径："12/gt/500Hz/raw/h/00206.png"
        path_parts = os.path.normpath(rel_path).split(os.sep)
        folder_id = path_parts[0]
        # 兼容多种目录布局：
        # - 如果 rel_path 是 "<group>/images/xxx.png"，则对应的 gt 在 "<group>/gt/xxx.png"
        # - 否则，按原始方式将 path_parts[1:] 拼接到 gt 下
        if len(path_parts) >= 2 and path_parts[1].lower() == 'images':
            gt_rel_path = os.path.join(folder_id, 'gt', *path_parts[2:])
        else:
            gt_rel_path = os.path.join(folder_id, 'gt', *path_parts[1:])
        gt_path = os.path.join(self.root_dir, gt_rel_path)

        # 如果构造的 gt_path 不存在，尝试另一种常见布局（避免死掉）
        if not os.path.exists(gt_path):
            # 尝试把最后一层父目录去掉（兼容一些不同的组织）
            alt_rel = os.path.join(folder_id, 'gt', os.path.basename(rel_path))
            alt_path = os.path.join(self.root_dir, alt_rel)
            if os.path.exists(alt_path):
                gt_path = alt_path

        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = gt_img.resize((self.width, self.height), Image.LANCZOS)

        # 应用 transforms（如有）
        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            gt_img = self.target_transform(gt_img)

        return {
            ("color", -1): input_img,
            ("gt", -1): gt_img
        }

