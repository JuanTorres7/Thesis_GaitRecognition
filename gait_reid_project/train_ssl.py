# train_ssl.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time
import os

from configs import settings
from src.data import CASIAB_SSL
from src.models import GaitBackbone
from src.losses import NTXentLoss


def train_ssl(config):
    """Fase 1 SSL con aumentaciones π(·) opcionales desde settings."""

    # [NUEVO] Leer parámetros de aumentación desde config
    use_aug    = getattr(config, 'SSL_USE_GAIT_AUGMENTATION', False)
    erase_prob = getattr(config, 'SSL_ERASE_PROB',  0.5)
    torso_only = getattr(config, 'SSL_ERASE_TORSO_ONLY', True)
    scale_prob = getattr(config, 'SSL_SCALE_PROB',  0.5)
    scale_range = getattr(config, 'SSL_SCALE_RANGE', (0.85, 1.15))

    dataset = CASIAB_SSL(
        root_path=config.ROOT_PATH,
        img_size=config.IMG_SIZE,
        use_subset=config.SSL_USE_SUBSET,
        subset_subjects=config.SSL_SUBSET_SUBJECTS,
        subset_conditions=config.SSL_SUBSET_CONDITIONS,
        subset_angles=config.SSL_SUBSET_ANGLES,
        seq_len=config.SSL_SUBSET_FRAMES_PER_SEQ,
        use_augmentation=use_aug,
        erase_prob=erase_prob,
        torso_only=torso_only,
        scale_prob=scale_prob,
        scale_range=scale_range,
    )

    # num_workers=2 funciona en Windows cuando el script tiene if __name__=='__main__'
    # (que ya está en la línea final). Cada worker carga datos en paralelo mientras
    # la GPU procesa el batch anterior — elimina el cuello de botella de I/O.
    # No usar persistent_workers=True con num_workers=0, pero con 2 sí es válido.
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=(config.DEVICE == "cuda"),
        persistent_workers=True,
    )

    model     = GaitBackbone(embed_dim=256).to(config.DEVICE)
    criterion = NTXentLoss(temperature=config.SSL_TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=config.SSL_LEARNING_RATE, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.SSL_EPOCHS, eta_min=1e-6)

    # Mixed Precision (AMP): usa FP16 en las operaciones del forward pass y FP32
    # para los gradientes. En RTX 4060 esto da ~25-35% menos tiempo por época
    # sin pérdida de calidad en las representaciones SSL.
    # Activado por defecto si hay CUDA disponible.
    USE_MIXED_PRECISION = (config.DEVICE == "cuda")
    scaler = GradScaler('cuda') if USE_MIXED_PRECISION else None

    print(f"\n--- Configuración SSL ---")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Epochs: {config.SSL_EPOCHS}")
    print(f"Learning rate: {config.SSL_LEARNING_RATE}")
    print(f"Aumentaciones π(·): {'ACTIVADAS' if use_aug else 'desactivadas'}")
    print(f"Steps/epoch: {len(loader)}\n")

    best_loss = float('inf')

    for epoch in range(1, config.SSL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        start_time = time.time()

        for view1, view2 in loader:
            view1 = view1.to(config.DEVICE, non_blocking=True)
            view2 = view2.to(config.DEVICE, non_blocking=True)

            if scaler:
                with autocast('cuda'):
                    z1   = model(view1)
                    z2   = model(view2)
                    loss = criterion(z1, z2)
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                z1   = model(view1)
                z2   = model(view2)
                loss = criterion(z1, z2)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()

        scheduler.step()
        avg_loss   = epoch_loss / len(loader)
        epoch_time = time.time() - start_time

        print(
            f"Epoch {epoch:2d}/{config.SSL_EPOCHS} | Loss: {avg_loss:.4f} | "
            f"Time: {epoch_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save({
                'epoch':               epoch,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss':                best_loss,
            }, config.SSL_CHECKPOINT)

    print(f"\n[SSL] Entrenamiento completado.")
    print(f"  Mejor loss: {best_loss:.4f}")
    print(f"  Modelo guardado: {config.SSL_CHECKPOINT}")
    return model


if __name__ == "__main__":
    settings.check_paths()
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    train_ssl(settings)