import os
from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    """DDPM 训练/测试使用的数据集，与 UNet 版本保持一致。"""

    def __init__(self, root_dir, split_file, height, width, transform=None, target_transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.height = height
        self.width = width
        self.transform = transform
        self.target_transform = target_transform

        self.filenames = []
        with open(split_file, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.filenames.append(line)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):
        rel_path = self.filenames[index]

        input_path = os.path.join(self.root_dir, rel_path)
        input_img = Image.open(input_path).convert("RGB")
        input_img = input_img.resize((self.width, self.height), Image.LANCZOS)

        path_parts = os.path.normpath(rel_path).split(os.sep)
        folder_id = path_parts[0]
        if len(path_parts) >= 2 and path_parts[1].lower() == "images":
            gt_rel_path = os.path.join(folder_id, "gt", *path_parts[2:])
        else:
            gt_rel_path = os.path.join(folder_id, "gt", *path_parts[1:])
        gt_path = os.path.join(self.root_dir, gt_rel_path)

        if not os.path.exists(gt_path):
            alt_rel = os.path.join(folder_id, "gt", os.path.basename(rel_path))
            alt_path = os.path.join(self.root_dir, alt_rel)
            if os.path.exists(alt_path):
                gt_path = alt_path

        gt_img = Image.open(gt_path).convert("RGB")
        gt_img = gt_img.resize((self.width, self.height), Image.LANCZOS)

        if self.transform:
            input_img = self.transform(input_img)
        if self.target_transform:
            gt_img = self.target_transform(gt_img)

        return {
            ("color", -1): input_img,
            ("gt", -1): gt_img,
        }

