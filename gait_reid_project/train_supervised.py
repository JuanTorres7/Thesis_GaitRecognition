# train_supervised.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os

import argparse

# Importar desde nuestros módulos
from configs import settings
from src.data import CASIAB_Supervised
from src.models import GaitBackbone, SupervisedReIDModel
# Nota: La TripletLoss está en src.losses, pero este script usa CrossEntropy

def train_supervised(config, use_ssl=True):
    """Función principal de entrenamiento supervisado"""
    
    print("\n--- Cargando Datasets Supervisados ---")
    train_dataset = CASIAB_Supervised(
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['train_range'],
        conditions=config.SUPERVISED_CONFIG['conditions'],
        angles=config.SUPERVISED_CONFIG['angles'],
        seq_len=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=True
    )
    
    # Se elimina val_dataset para evitar label shifts y ahorrar tiempo computacional
    # El modelo se evaluará empíricamente por su Train Loss
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, persistent_workers=True)
    
    num_classes = train_dataset.get_num_classes()
    print(f"\nClases (personas) en train: {num_classes}")
    
    # --- Cargar Backbone Pre-entrenado ---
    backbone = GaitBackbone(embed_dim=256)
    
    if use_ssl and os.path.exists(config.SSL_CHECKPOINT):
        # weights_only=False asegura compatibilidad con diccionarios guardados en PyTorch 2.6
        checkpoint = torch.load(config.SSL_CHECKPOINT, map_location=config.DEVICE, weights_only=False)
        backbone.load_state_dict(checkpoint['model_state_dict'])
        print(f"  ✓ Backbone SSL cargado exitosamente desde: {config.SSL_CHECKPOINT}")
        print(f"    (Epoch: {checkpoint.get('epoch', 'N/A')}, Loss: {checkpoint.get('loss', 0.0):.4f})")
    else:
        if not use_ssl:
            print(f"  [ABLACIÓN] Modo --no-ssl activado. Ignorando Fase I.")
        else:
            print(f"  ⚠ ADVERTENCIA: No se encontró {config.SSL_CHECKPOINT}")
        print(f"    Entrenando desde cero (backbone aleatorio).")
    
    model = SupervisedReIDModel(
        backbone=backbone,
        num_classes=num_classes,
        freeze_backbone=False  # Fine-tuning de todo el modelo
    ).to(config.DEVICE)
    
    # --- Configuración de Entrenamiento ---
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.SUPERVISED_LEARNING_RATE, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    print(f"\n--- Configuración Supervisada ---")
    print(f"Epochs: {config.SUPERVISED_EPOCHS}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Learning rate: {config.SUPERVISED_LEARNING_RATE}")
    print(f"Steps/epoch: {len(train_loader)}\n")
    
    best_train_loss = float('inf')
    history = {'train_loss': [], 'train_acc': []}
    
    for epoch in range(1, config.SUPERVISED_EPOCHS + 1):
        # TRAINING
        model.train()
        train_loss, train_correct, train_total = 0, 0, 0
        start_time = time.time()
        
        for images, labels in train_loader:
            images, labels = images.to(config.DEVICE), labels.to(config.DEVICE)
            
            logits = model(images)
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = logits.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        scheduler.step()
        epoch_time = time.time() - start_time
        
        # Limpieza de logs: Enfocarse puramente en el Aprendizaje de la Red
        print(f"Epoch {epoch:2d}/{config.SUPERVISED_EPOCHS} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Train Acc: {train_acc:.2f}% | "
              f"Time: {epoch_time:.1f}s")
        
        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(train_acc)
        
        # REGLAS DE GUARDADO ROBUSTO DE LA TESIS
        is_best = avg_train_loss < best_train_loss
        is_last = (epoch == config.SUPERVISED_EPOCHS)
        
        if is_best or is_last:
            if is_best:
                best_train_loss = avg_train_loss
                
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'train_acc': train_acc,
                'num_classes': num_classes,
                'history': history
            }, config.SUPERVISED_CHECKPOINT)
            
            if is_best:
                print(f"  [✓] Nuevo mejor modelo guardado (Train Loss: {best_train_loss:.4f})")
            elif is_last:
                print(f"  [✓] FORZADO: Última época guardada por seguridad.")

    print(f"\n--- ENTRENAMIENTO SUPERVISADO COMPLETADO ---")
    print(f"Mejor Train Loss Lograda: {best_train_loss:.4f}")
    print(f"Archivo definitivo: {config.SUPERVISED_CHECKPOINT}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento Supervisado (Estabilización)")
    parser.add_argument('--no-ssl', action='store_true', help="Ignorar el pre-entrenamiento SSL y arrancar desde cero (Ablación 1).")
    args = parser.parse_args()

    # 1. Verificar rutas y configuraciones
    settings.check_paths()

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    
    # 2. Iniciar el entrenamiento
    use_ssl_flag = not args.no_ssl
    train_supervised(settings, use_ssl=use_ssl_flag)