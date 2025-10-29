import argparse
import os
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import transforms, utils

from dataset_CycleGAN import build_inference_dataset
from model_CycleGAN import build_generator


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * 0.5 + 0.5


def generate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    out_dir: str,
    max_samples: Optional[int] = None,
    prefix: str = "output",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    processed = 0
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            paths = batch.get("path")
            outputs = model(images)
            outputs = denormalize(outputs)

            for idx in range(outputs.size(0)):
                rel_path = None
                if paths is not None:
                    rel_path = paths[idx]
                base = (
                    f"{prefix}_{processed + idx:05d}.png"
                    if not rel_path
                    else os.path.splitext(os.path.basename(rel_path))[0] + ".png"
                )
                save_path = os.path.join(out_dir, base)
                utils.save_image(outputs[idx].cpu().clamp(0, 1), save_path)

            processed += outputs.size(0)
            if max_samples is not None and processed >= max_samples:
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate CycleGAN generators")
    parser.add_argument("--data_root", type=str, default="data_net")
    parser.add_argument("--split_a", type=str, default="split/testA.txt")
    parser.add_argument("--split_b", type=str, default="split/testB.txt")
    parser.add_argument("--checkpoint_g_ab", type=str, required=True)
    parser.add_argument("--checkpoint_g_ba", type=str, required=True)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs/cyclegan")
    parser.add_argument("--max_samples", type=int, default=None)
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

    dataset_A = build_inference_dataset(
        root_dir=args.data_root,
        split_file=args.split_a,
        transform=transform,
    )
    dataset_B = build_inference_dataset(
        root_dir=args.data_root,
        split_file=args.split_b,
        transform=transform,
    )

    loader_A = DataLoader(
        dataset_A,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    loader_B = DataLoader(
        dataset_B,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    netG_AB = build_generator(3, 3)
    netG_BA = build_generator(3, 3)
    netG_AB.load_state_dict(torch.load(args.checkpoint_g_ab, map_location="cpu"))
    netG_BA.load_state_dict(torch.load(args.checkpoint_g_ba, map_location="cpu"))
    netG_AB.to(device)
    netG_BA.to(device)

    out_a2b = os.path.join(args.output_dir, "AtoB")
    out_b2a = os.path.join(args.output_dir, "BtoA")

    generate(netG_AB, loader_A, device, out_a2b, args.max_samples, prefix="AtoB")
    generate(netG_BA, loader_B, device, out_b2a, args.max_samples, prefix="BtoA")


if __name__ == "__main__":
    main()
