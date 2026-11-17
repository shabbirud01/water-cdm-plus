import os

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, split="train", image_size=256):
        self.input_dir = os.path.join(root_dir, split, "input")
        self.target_dir = os.path.join(root_dir, split, "target")
        self.image_size = image_size

        self.input_images = sorted([
            f for f in os.listdir(self.input_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])
        self.target_images = sorted([
            f for f in os.listdir(self.target_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
        ])

        assert len(self.input_images) == len(self.target_images), (
            f"Mismatch: {len(self.input_images)} inputs vs {len(self.target_images)} targets"
        )

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        inp = Image.open(os.path.join(self.input_dir, self.input_images[idx])).convert("RGB")
        tgt = Image.open(os.path.join(self.target_dir, self.target_images[idx])).convert("RGB")
        return self.transform(inp), self.transform(tgt)


def get_train_loader(config):
    ds = UnderwaterDataset(config.DATA_DIR, split="train", image_size=config.IMAGE_SIZE)
    return DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
