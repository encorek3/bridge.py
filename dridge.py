
import argparse
import os
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ---------------------------------------------------------------------------
# 1. 資料集
# ---------------------------------------------------------------------------
class CrackSegmentationDataset(Dataset):


    def __init__(self, images_dir: str, masks_dir: str, image_size: int = 256):
        self.image_paths = sorted(glob(os.path.join(images_dir, "*")))
        self.masks_dir = masks_dir
        self.image_size = image_size

        self.img_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # -> [0,1], shape (3,H,W)
        ])
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),  # -> [0,1], shape (1,H,W)
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        filename = os.path.basename(img_path)
        mask_path = os.path.join(self.masks_dir, filename)

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # 灰階

        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        mask = (mask > 0.5).float()  # 二值化,確保是 0 或 1

        return image, mask


# ---------------------------------------------------------------------------
# 2. U-Net 模型(輕量版,從零訓練,不依賴預訓練權重)
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):


    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, base_ch=32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base_ch)
        self.enc2 = DoubleConv(base_ch, base_ch * 2)
        self.enc3 = DoubleConv(base_ch * 2, base_ch * 4)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(base_ch * 4, base_ch * 8)

        self.up3 = nn.ConvTranspose2d(base_ch * 8, base_ch * 4, 2, stride=2)
        self.dec3 = DoubleConv(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 2, stride=2)
        self.dec2 = DoubleConv(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, 2, stride=2)
        self.dec1 = DoubleConv(base_ch * 2, base_ch)

        self.out_conv = nn.Conv2d(base_ch, out_channels, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))

        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))  # skip connection: 保留淺層的空間細節
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_conv(d1)  # 未經 sigmoid 的 logits,配合 BCEWithLogitsLoss 使用


# ---------------------------------------------------------------------------
# 3. 損失函數: BCE + Dice(分割任務常見組合,能兼顧像素準確度與區域重疊度)
# ---------------------------------------------------------------------------
def dice_loss(pred_logits, target, eps=1e-6):
    pred = torch.sigmoid(pred_logits)
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def combined_loss(pred_logits, target):
    bce = F.binary_cross_entropy_with_logits(pred_logits, target)
    dice = dice_loss(pred_logits, target)
    return bce + dice


# ---------------------------------------------------------------------------
# 4. 訓練迴圈
# ---------------------------------------------------------------------------
def train(data_dir: str, epochs: int = 20, batch_size: int = 8,
          lr: float = 1e-3, image_size: int = 256,
          checkpoint_path: str = "crack_unet.pth"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用裝置: {device}")

    dataset = CrackSegmentationDataset(
        images_dir=os.path.join(data_dir, "images"),
        masks_dir=os.path.join(data_dir, "masks"),
        image_size=image_size,
    )
    if len(dataset) == 0:
        raise RuntimeError(f"在 {data_dir} 找不到任何影像,請確認 images/ masks/ 資料夾已放好照片")

    # 簡單切 80/20 訓練驗證集
    val_size = max(1, int(len(dataset) * 0.2))
    train_size = len(dataset) - val_size
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            preds = model(images)
            loss = combined_loss(preds, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        train_loss /= len(train_set)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                val_loss += combined_loss(preds, masks).item() * images.size(0)
        val_loss /= len(val_set)

        print(f"Epoch {epoch}/{epochs} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  -> 已儲存最佳模型至 {checkpoint_path}")

    print("訓練完成")
    return model


# ---------------------------------------------------------------------------
# 5. 推論: 對單張照片產生裂縫遮罩,並估算劣化範圍比例
# ---------------------------------------------------------------------------
def predict_mask(model_or_checkpoint, image_path: str, image_size: int = 256,
                  threshold: float = 0.5, device: str = None):

    import numpy as np

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    if isinstance(model_or_checkpoint, str):
        model = UNet().to(device)
        model.load_state_dict(torch.load(model_or_checkpoint, map_location=device))
    else:
        model = model_or_checkpoint.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()

    binary_mask = (prob > threshold).astype(np.uint8)
    crack_ratio = float(binary_mask.mean())

    return binary_mask, crack_ratio


def crack_ratio_to_deru_extent(crack_ratio: float) -> int:

    if crack_ratio >= 0.15:
        return 4
    if crack_ratio >= 0.08:
        return 3
    if crack_ratio >= 0.03:
        return 2
    if crack_ratio > 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# 6. 命令列進入點
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="裂縫分割模型訓練練習腳本")
    parser.add_argument("--data_dir", type=str, required=True,
                         help="資料集根目錄,底下需有 images/ 與 masks/ 兩個子資料夾")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
    )
