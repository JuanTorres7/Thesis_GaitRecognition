# train_ssl.py
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import time

from configs import settings
from src.data import get_ssl_dataset
from src.models import GaitBackbone
from src.losses import NTXentLoss


def train_ssl(config):
    """Fase 1 SSL — funciona con CASIA-B y Gait3D via get_ssl_dataset()."""

    dataset = get_ssl_dataset(config)

    # num_workers: en cluster Linux con varios CPUs asignados, subir agresivamente.
    # Gait3D tiene 18940 secuencias (2.3x más que CASIA-B) — el cuello de botella
    # es CoM alignment por frame en CPU. Más workers = más paralelismo de I/O+CPU.
    # Verificar con `nproc` cuántos cores tiene el nodo asignado.
    num_workers = getattr(config, 'DATALOADER_NUM_WORKERS', 2)
    loader = DataLoader(
        dataset, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=num_workers, pin_memory=(config.DEVICE == "cuda"),
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )
    print(f"  DataLoader workers: {num_workers}")

    model     = GaitBackbone(embed_dim=256).to(config.DEVICE)
    criterion = NTXentLoss(temperature=config.SSL_TEMPERATURE)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.SSL_LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.SSL_EPOCHS, eta_min=1e-6
    )

    USE_AMP = (config.DEVICE == "cuda")
    scaler  = GradScaler('cuda') if USE_AMP else None

    print(f"\n--- Configuración SSL [{config.ACTIVE_DATASET.upper()}] ---")
    print(f"Epochs: {config.SSL_EPOCHS} | Batch: {config.BATCH_SIZE} | "
          f"LR: {config.SSL_LEARNING_RATE} | Steps/epoch: {len(loader)}")

    best_loss = float('inf')

    for epoch in range(1, config.SSL_EPOCHS + 1):
        model.train()
        epoch_loss = 0.0
        t0 = time.time()

        for step, (view1, view2) in enumerate(loader):
            view1 = view1.to(config.DEVICE, non_blocking=True)
            view2 = view2.to(config.DEVICE, non_blocking=True)
            if scaler:
                with autocast('cuda'):
                    loss = criterion(model(view1), model(view2))
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(model(view1), model(view2))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            epoch_loss += loss.item()

            # Progreso cada 20 steps — confirma que el loop avanza
            # y permite estimar tiempo/epoch antes de que termine.
            if (step + 1) % 20 == 0:
                elapsed   = time.time() - t0
                step_time = elapsed / (step + 1)
                eta_epoch = step_time * len(loader)
                print(f"    step {step+1:4d}/{len(loader)} | "
                      f"{step_time:.2f}s/step | ETA época: {eta_epoch/60:.1f} min",
                      flush=True)

        scheduler.step()
        avg = epoch_loss / len(loader)
        print(f"Epoch {epoch:2d}/{config.SSL_EPOCHS} | Loss: {avg:.4f} | "
              f"Time: {time.time()-t0:.1f}s | LR: {scheduler.get_last_lr()[0]:.2e}")

        if avg < best_loss:
            best_loss = avg
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_loss}, config.SSL_CHECKPOINT)

    print(f"\n[SSL] Completado. Mejor loss: {best_loss:.4f} → {config.SSL_CHECKPOINT}")
    return model


if __name__ == "__main__":
    settings.check_paths()
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision('high')
    train_ssl(settings)
