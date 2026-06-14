# train_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import argparse

from configs import settings
from src.data import get_supervised_dataset
from src.models import GaitBackbone, SupervisedReIDModel


def train_supervised(config, use_ssl=True):
    """Fase 2 supervisada — funciona con CASIA-B y Gait3D."""

    print(f"\n--- Entrenamiento Supervisado [{config.ACTIVE_DATASET.upper()}] ---")

    train_dataset = get_supervised_dataset(config, split='train', augment=True)
    train_loader  = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )

    num_classes = train_dataset.get_num_classes()
    print(f"Clases en train: {num_classes}")

    backbone = GaitBackbone(embed_dim=256)

    if use_ssl and os.path.exists(config.SSL_CHECKPOINT):
        ckpt = torch.load(config.SSL_CHECKPOINT, map_location=config.DEVICE,
                          weights_only=False)
        backbone.load_state_dict(ckpt['model_state_dict'])
        print(f"  ✓ Backbone SSL: {config.SSL_CHECKPOINT} "
              f"(epoch {ckpt.get('epoch','?')}, loss {ckpt.get('loss',0):.4f})")
    else:
        print(f"  ⚠ Sin SSL checkpoint — entrenando desde cero.")

    model = SupervisedReIDModel(backbone=backbone, num_classes=num_classes,
                                freeze_backbone=False).to(config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.SUPERVISED_LEARNING_RATE,
                           weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    print(f"Epochs: {config.SUPERVISED_EPOCHS} | "
          f"LR: {config.SUPERVISED_LEARNING_RATE} | Steps/epoch: {len(train_loader)}\n")

    best_loss = float('inf')

    for epoch in range(1, config.SUPERVISED_EPOCHS + 1):
        model.train()
        t_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            logits = model(images)
            loss   = criterion(logits, labels)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            t_loss  += loss.item()
            correct += logits.max(1)[1].eq(labels).sum().item()
            total   += labels.size(0)

        scheduler.step()
        avg  = t_loss / len(train_loader)
        acc  = 100. * correct / total
        print(f"Epoch {epoch:2d}/{config.SUPERVISED_EPOCHS} | "
              f"Loss: {avg:.4f} | Acc: {acc:.2f}% | Time: {time.time()-t0:.1f}s")

        is_best = avg < best_loss
        is_last = epoch == config.SUPERVISED_EPOCHS
        if is_best or is_last:
            if is_best: best_loss = avg
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'train_loss': avg, 'num_classes': num_classes},
                       config.SUPERVISED_CHECKPOINT)
            tag = "[✓ Mejor]" if is_best else "[✓ Último]"
            print(f"  {tag} → {config.SUPERVISED_CHECKPOINT}")

    print(f"\n--- Supervisado completado. Mejor loss: {best_loss:.4f} ---")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-ssl', action='store_true')
    args = parser.parse_args()
    settings.check_paths()
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision('high')
    train_supervised(settings, use_ssl=not args.no_ssl)
