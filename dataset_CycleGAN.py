import os
import random
from typing import Callable, List, Optional

from PIL import Image
from torch.utils.data import Dataset as TorchDataset


class CycleGANDataset(TorchDataset):
    """A dataset that yields unpaired samples from domain A and domain B.

    It mirrors the light-weight dataset helpers used for the CNN/DDPM
    experiments so that the training/testing scripts can remain focused on
    the modelling logic. The dataset expects two text files that list image
    paths relative to ``root_dir`` for each domain. Sampling is done without
    pairing assumptions, which is the standard CycleGAN workflow.
    """

    def __init__(
        self,
        root_dir: str,
        split_file_a: str,
        split_file_b: str,
        transform_a: Optional[Callable] = None,
        transform_b: Optional[Callable] = None,
        return_paths: bool = False,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.transform_a = transform_a
        self.transform_b = transform_b
        self.return_paths = return_paths

        self.paths_a = self._read_split(split_file_a)
        self.paths_b = self._read_split(split_file_b)
        if not self.paths_a:
            raise ValueError("split_file_a did not provide any valid entries")
        if not self.paths_b:
            raise ValueError("split_file_b did not provide any valid entries")

    def _read_split(self, split_file: str) -> List[str]:
        paths: List[str] = []
        with open(split_file, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                paths.append(line)
        return paths

    def __len__(self) -> int:
        return max(len(self.paths_a), len(self.paths_b))

    def _load_image(self, rel_path: str) -> Image.Image:
        path = os.path.join(self.root_dir, rel_path)
        img = Image.open(path).convert("RGB")
        return img

    def __getitem__(self, idx: int):
        idx_a = idx % len(self.paths_a)
        img_a = self._load_image(self.paths_a[idx_a])

        idx_b = random.randint(0, len(self.paths_b) - 1)
        img_b = self._load_image(self.paths_b[idx_b])

        if self.transform_a:
            img_a = self.transform_a(img_a)
        if self.transform_b:
            img_b = self.transform_b(img_b)

        if self.return_paths:
            return {
                "A": img_a,
                "B": img_b,
                "path_A": self.paths_a[idx_a],
                "path_B": self.paths_b[idx_b],
            }
        return {"A": img_a, "B": img_b}


def build_inference_dataset(
    root_dir: str,
    split_file: str,
    transform: Optional[Callable] = None,
    return_paths: bool = True,
) -> TorchDataset:
    """Utility helper that only iterates over a single split.

    This is used by the testing script to generate translated samples from a
    specific domain. It reuses the same text file listing convention used by
    :class:`CycleGANDataset`.
    """

    entries = []
    with open(split_file, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            entries.append(line)

    if not entries:
        raise ValueError("split file is empty; please provide at least one path")

    class _SingleDomainDataset(TorchDataset):
        def __len__(self) -> int:
            return len(entries)

        def __getitem__(self, index: int):
            rel_path = entries[index]
            img = Image.open(os.path.join(root_dir, rel_path)).convert("RGB")
            if transform:
                img_t = transform(img)
            else:
                img_t = img
            if return_paths:
                return {"image": img_t, "path": rel_path}
            return {"image": img_t}

    return _SingleDomainDataset()
