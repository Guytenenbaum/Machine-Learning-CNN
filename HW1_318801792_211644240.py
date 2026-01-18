import os
import copy
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from PIL import Image


# =========================
# Reproducibility
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =========================
# Dataset
# =========================
class BigCatDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        root_dir: path to 'train', 'valid' or 'test' folder.
                  Inside it: one subfolder per class with images.
        """
        self.root_dir = root_dir
        self.transform = transform

        self.classes = sorted([
            d for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # build list of (path, label)
        self.samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(root_dir, class_name)
            for fname in os.listdir(class_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(class_dir, fname), class_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        img_path, label = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

    def get_class(self, idx):
        return self.classes[idx]


# =========================
# Transforms
# =========================
train_transform = transforms.Compose([
    transforms.Resize((72, 72)),
    transforms.RandomResizedCrop(64, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])

val_test_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5),
                         (0.5, 0.5, 0.5))
])


# =========================
# EDA Helper Functions
# =========================
def count_images(root):
    per_class = {}
    for cls in os.listdir(root):
        cls_path = os.path.join(root, cls)
        if os.path.isdir(cls_path):
            per_class[cls] = len(os.listdir(cls_path))
    return per_class, sum(per_class.values())


def show_random_samples(dataset, n=6):
    idxs = random.sample(range(len(dataset)), n)
    plt.figure(figsize=(n * 2.2, 3))
    for i, idx in enumerate(idxs):
        img, label = dataset[idx]
        plt.subplot(1, n, i + 1)

        if isinstance(img, torch.Tensor):
            img = img.permute(1, 2, 0).numpy()
            img = (img * 0.5) + 0.5    # roughly unnormalize
        else:
            img = np.array(img)        # makes sure PIL → numpy

        plt.imshow(img)
        plt.title(dataset.classes[label])
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# =========================
# Model
# =========================
class BigCatsCNN(nn.Module):
    """
    Deeper CNN with controlled parameter count (< 500k):

    - 4 conv blocks, each with 2 conv layers:
        3 -> 24 -> 48 -> 96 -> 128 channels
    - Each conv: 3x3, padding=1, followed by BN + ReLU
    - MaxPool2d(2) after each block to downsample
    - AdaptiveAvgPool2d((1, 1)) + small classifier
    """

    def __init__(self, num_classes=10):
        super(BigCatsCNN, self).__init__()

        def conv_block(in_ch, out_ch, num_convs=2):
            layers = []
            for i in range(num_convs):
                ch_in = in_ch if i == 0 else out_ch
                layers += [
                    nn.Conv2d(ch_in, out_ch, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                ]
            layers.append(nn.MaxPool2d(2))  # downsample by 2
            return nn.Sequential(*layers)

        # Input: (B, 3, 64, 64)
        self.features = nn.Sequential(
            conv_block(3, 24),   # 64 -> 32
            conv_block(24, 48),  # 32 -> 16
            conv_block(48, 96),  # 16 -> 8
            conv_block(96, 128), # 8 -> 4
            nn.AdaptiveAvgPool2d((1, 1))   # -> (B, 128, 1, 1)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),        # (B, 128)
            nn.Linear(128, 128),
            nn.ReLU(True),
            nn.Dropout(p=0.5),
            nn.Linear(128, num_classes),
            nn.LogSoftmax(dim=1),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# Main Script
# =========================
def main():
    # reproducibility
    set_seed(42)

    DATA_DIR = "big_cats"

    # ---------- EDA ----------
    base_dir = "big_cats"
    train_dir = "big_cats/train"
    valid_dir = "big_cats/valid"
    test_dir = "big_cats/test"

    print("============= Exploratory Data Analysis =============\n")

    # Classes
    train_tmp = ImageFolder(train_dir)
    classes = train_tmp.classes
    print("==== Classes of Big Cats ====\n")
    print(classes)

    # Counts per split & class
    print("\n==== Count in Split and Classes ====")
    for name, d in [("Train", train_dir), ("Valid", valid_dir), ("Test", test_dir)]:
        per_class, total = count_images(d)
        print(f"\n{name} — {total} images total")
        for cls, n in per_class.items():
            print(f"  {cls}: {n}")

    # Show random samples from train
    print("\n==== Show Samples (Train) ====\n")
    sample_ds = ImageFolder(train_dir)
    random.seed(42)
    show_random_samples(sample_ds, n=6)

    # Image size distribution train
    print("\n==== Image Size Distribution (Train) ====\n")

    sizes = []
    for cls in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls)
        if os.path.isdir(cls_path):
            for img_name in os.listdir(cls_path):
                img_path = os.path.join(cls_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        sizes.append(img.size)
                except Exception:
                    pass

    sizes = np.array(sizes)
    widths = sizes[:, 0]
    heights = sizes[:, 1]

    print("Image size statistics (train set):\n")
    print(f"Mean     : width {widths.mean():.1f}, height {heights.mean():.1f}")
    print(f"Median   : width {np.median(widths):.1f}, height {np.median(heights):.1f}")
    print(f"Std dev  : width {widths.std():.1f}, height {heights.std():.1f}")

    # ---------- Datasets & Dataloaders ----------
    train_dataset = BigCatDataset(os.path.join(DATA_DIR, "train"), transform=train_transform)
    val_dataset = BigCatDataset(os.path.join(DATA_DIR, "valid"), transform=val_test_transform)
    test_dataset = BigCatDataset(os.path.join(DATA_DIR, "test"), transform=val_test_transform)

    BATCH_SIZE = 32

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    num_classes = len(train_dataset.classes)
    print("\nClasses:", train_dataset.classes)
    print("Train size:", len(train_dataset),
          "Val size:", len(val_dataset),
          "Test size:", len(test_dataset))

    # ---- Compute class weights (smoothed inverse frequency) ----
    labels_list = [label for _, label in train_dataset.samples]
    class_counts = np.bincount(labels_list, minlength=num_classes)
    print("\nClass counts:", class_counts)

    # Base inverse-frequency weights
    inv_freq = 1.0 / class_counts
    inv_freq = inv_freq / inv_freq.mean()  # normalize around 1

    # Smoothing parameter: 1.0 = original, 0.5 = milder, 0.0 = no weighting
    ALPHA = 0.5
    class_weights = inv_freq ** ALPHA
    class_weights = class_weights / class_weights.mean()  # normalize again

    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    print("Smoothed class weights (alpha = %.2f):" % ALPHA, class_weights)

    # ---------- Model / Training Setup ----------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    model = BigCatsCNN(num_classes=num_classes).to(device)

    # Use class-weighted loss
    class_weights = class_weights.to(device)
    criterion = nn.NLLLoss(weight=class_weights)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.3)

    num_params = sum(p.numel() for p in model.parameters())
    print("Number of parameters:", num_params)

    num_epochs = 150

    train_loss_hist = []
    val_loss_hist = []
    train_acc_hist = []
    val_acc_hist = []

    best_val_acc = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())
    best_epoch = 0

    # ---------- Training Loop ----------
    for epoch in range(num_epochs):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        epoch_train_loss = running_loss / running_total
        epoch_train_acc = running_correct / running_total

        # ---- Validation ----
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total

        train_loss_hist.append(epoch_train_loss)
        val_loss_hist.append(epoch_val_loss)
        train_acc_hist.append(epoch_train_acc)
        val_acc_hist.append(epoch_val_acc)

        scheduler.step()

        # Track best model
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_state_dict = copy.deepcopy(model.state_dict())
            best_epoch = epoch + 1

        print(
            f"Epoch [{epoch + 1}/{num_epochs}] "
            f"Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.3f} | "
            f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.3f}"
        )

    print(f"\nBest validation accuracy: {best_val_acc:.3f} at epoch {best_epoch}")
    model.load_state_dict(best_state_dict)

    # ---------- SAVE BEST WEIGHTS TO .PKL ----------
    weights_path = "bigcats_cnn_weights.pkl"
    torch.save(best_state_dict, weights_path)
    print(f"Saved best model weights to: {weights_path}")

    # ---------- Plots ----------
    epochs = range(1, num_epochs + 1)

    plt.figure()
    plt.plot(epochs, train_loss_hist, label="Train Loss")
    plt.plot(epochs, val_loss_hist, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure()
    plt.plot(epochs, train_acc_hist, label="Train Acc")
    plt.plot(epochs, val_acc_hist, label="Val Acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epochs")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ---------- Test Evaluation ----------
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    test_acc = correct / total
    print(f"\nTest Accuracy: {test_acc:.3f}")


if __name__ == "__main__":
    main()
