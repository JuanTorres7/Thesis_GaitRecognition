# train_hybrid.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import sys
import numpy as np

from configs import settings
from src.data import get_supervised_dataset
from src.models import HybridGaitModel, SupervisedReIDModel, GaitBackbone
from src.losses import TripletLoss, CircleLoss, ProxyAnchorLoss
from src.samplers import RandomIdentitySampler


def compute_mini_metrics(dist_matrix, q_labels, g_labels):
    """Métrica local de validación sin dependencias externas."""
    num_queries = len(q_labels)
    if torch.is_tensor(dist_matrix):
        dist_matrix = dist_matrix.cpu().numpy()

    hits_top1, map_sum, valid_queries = 0, 0.0, 0
    for i in range(num_queries):
        dists_i        = dist_matrix[i, :]
        sorted_indices = np.argsort(dists_i)
        sorted_g_labels = g_labels[sorted_indices]
        matches        = (sorted_g_labels == q_labels[i]).astype(np.int32)
        num_tp         = np.sum(matches)
        if num_tp == 0:
            continue
        valid_queries += 1
        hits_top1     += matches[0]
        num_hits, sum_prec = 0, 0.0
        for j, match in enumerate(matches):
            if match == 1:
                num_hits  += 1
                sum_prec  += num_hits / (j + 1)
        map_sum += sum_prec / num_tp

    r1  = (hits_top1 / valid_queries * 100) if valid_queries > 0 else 0.0
    mAP = (map_sum   / valid_queries * 100) if valid_queries > 0 else 0.0
    return r1, mAP


@torch.no_grad()
def _compute_centroids(model, dataset, num_classes, embed_dim, device, batch_size=32):
    """
    Calcula centroides L2-normalizados por clase desde el modelo Fase 2.
    Se usa para inicializar los proxies PAL en una posición razonable,
    evitando que arranquen desde ruido y desperdicien épocas de Fase 3.
    """
    print("\n  [PAL] Calculando centroides para inicializar proxies...")
    model.eval()
    loader         = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                num_workers=2, pin_memory=True)
    sum_embeddings = torch.zeros(num_classes, embed_dim, device=device)
    counts         = torch.zeros(num_classes, device=device)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        _, embeddings = model(images)
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                sum_embeddings[c] += embeddings[mask].sum(dim=0)
                counts[c]         += mask.sum().float()

    counts    = counts.clamp(min=1.0)
    centroids = sum_embeddings / counts.unsqueeze(1)
    centroids = torch.nn.functional.normalize(centroids, p=2, dim=1)
    model.train()
    print(f"  [PAL] Centroides calculados. Shape: {centroids.shape}")
    return centroids.cpu()


def train_hybrid(config):
    print(f"\n[+] ENTRENAMIENTO HÍBRIDO [{config.ACTIVE_DATASET.upper()}]\n")

    # -------------------------------------------------------------------------
    # 1. Datasets
    # -------------------------------------------------------------------------
    train_dataset = get_supervised_dataset(config, split='train', augment=True)
    val_dataset = get_supervised_dataset(config, split='val', augment=False, return_info=True)
    centroid_dataset = get_supervised_dataset(config, split='train', augment=False)

    num_classes = train_dataset.get_num_classes()
    embed_dim   = 1024
    print(f"\nClases en train: {num_classes} | Embedding dim: {embed_dim}")

    sampler = RandomIdentitySampler(train_dataset,
                                    batch_size=config.BATCH_SIZE,
                                    num_instances=config.NUM_INSTANCES)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE,
                              sampler=sampler, num_workers=2,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset, batch_size=config.BATCH_SIZE,
                              shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------------------------------------------------
    # 2. Cargar modelo desde Fase 2
    # -------------------------------------------------------------------------
    dummy_backbone   = GaitBackbone()
    supervised_model = SupervisedReIDModel(backbone=dummy_backbone,
                                           num_classes=num_classes)

    # [NUEVO] Lógica de reanudación:
    # Si existe un checkpoint híbrido previo → reanudar desde ahí (para añadir épocas).
    # Si no → cargar desde Fase 2 como siempre.
    hybrid_path = config.HYBRID_CHECKPOINT
    sup_path    = config.SUPERVISED_CHECKPOINT
    start_epoch = 1  # época de inicio por defecto

    model = HybridGaitModel(supervised_model).to(config.DEVICE)

    # Diccionario para guardar el estado PAL del checkpoint si existe
    _pal_state_from_checkpoint = None

    if os.path.exists(hybrid_path):
        checkpoint = torch.load(hybrid_path, map_location=config.DEVICE, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1
            # Guardar estado PAL para restaurarlo DESPUÉS de crear criterion_metric
            _pal_state_from_checkpoint = checkpoint.get('pal_state_dict', None)
            print(f"\n[OK] Reanudando desde checkpoint HÍBRIDO (época {start_epoch-1}).")
            print(f"     → Continuando entrenamiento desde época {start_epoch}.")
            if _pal_state_from_checkpoint:
                print(f"     → Estado PAL guardado: se restaurará (proxies no se reinician).")
        except Exception as e:
            print(f"[!] No se pudo cargar el híbrido ({e}). Cargando desde Fase 2...")
            start_epoch = 1

    if start_epoch == 1:
        if not os.path.exists(sup_path):
            print(f"\n[X] ERROR: No se encontró el modelo supervisado en {sup_path}.")
            sys.exit(1)
        checkpoint = torch.load(sup_path, map_location=config.DEVICE, weights_only=False)
        try:
            supervised_model.load_state_dict(checkpoint['model_state_dict'])
            model = HybridGaitModel(supervised_model).to(config.DEVICE)
            print(f"\n[OK] Pesos Fase 2 inyectados correctamente.")
        except Exception as e:
            print(f"[X] Error cargando Fase 2: {e}")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # 3. Pérdida métrica — PAL o Triplet/Circle según settings
    # -------------------------------------------------------------------------
    criterion_ce = nn.CrossEntropyLoss()
    use_pal      = getattr(config, 'USE_PAL', False)

    if use_pal:
        pal_alpha  = getattr(config, 'PAL_ALPHA',  32)
        pal_delta  = getattr(config, 'PAL_DELTA',  0.1)
        pal_weight = getattr(config, 'PAL_WEIGHT', 0.5)
        pal_init   = getattr(config, 'PAL_INIT_FROM_SUPERVISED', True)

        criterion_metric = ProxyAnchorLoss(num_classes=num_classes,
                                           embed_dim=embed_dim,
                                           alpha=pal_alpha,
                                           delta=pal_delta)
        if _pal_state_from_checkpoint is not None:
            # Reanudación: restaurar proxies exactamente como quedaron
            # en la época guardada — no reinicializar desde centroides
            criterion_metric = criterion_metric.to(config.DEVICE)
            criterion_metric.load_state_dict(_pal_state_from_checkpoint)
            print(f"  [PAL] Proxies restaurados desde checkpoint (época {start_epoch-1}).")
        elif pal_init:
            # Primera vez: inicializar desde centroides de Fase 2
            centroids = _compute_centroids(model, centroid_dataset,
                                           num_classes, embed_dim,
                                           config.DEVICE)
            criterion_metric.init_proxies_from_centroids(centroids)
            criterion_metric = criterion_metric.to(config.DEVICE)
        else:
            criterion_metric = criterion_metric.to(config.DEVICE)
        metric_weight    = pal_weight
        print(f"[*] Pérdida métrica: Proxy Anchor Loss "
              f"(alpha={pal_alpha}, delta={pal_delta}, weight={pal_weight})")

    elif config.USE_CIRCLE_LOSS:
        criterion_metric = CircleLoss(m=0.25, gamma=80)
        metric_weight    = config.HYBRID_TRIPLET_WEIGHT
        print(f"[*] Pérdida métrica: Circle Loss")
    else:
        criterion_metric = TripletLoss(margin=config.HYBRID_MARGIN)
        metric_weight    = config.HYBRID_TRIPLET_WEIGHT
        print(f"[*] Pérdida métrica: Triplet Loss (margin={config.HYBRID_MARGIN})")

    # -------------------------------------------------------------------------
    # 4. Optimizador diferencial (backbone 10× más lento que cabezales)
    # -------------------------------------------------------------------------
    spatial_params, head_params = [], []
    for name, param in model.named_parameters():
        if 'encoder' in name or 'spatial' in name:
            spatial_params.append(param)
        else:
            head_params.append(param)

    param_groups = [
        {'params': spatial_params, 'lr': config.HYBRID_LEARNING_RATE * 0.1},
        {'params': head_params,    'lr': config.HYBRID_LEARNING_RATE},
    ]
    if use_pal:
        param_groups.append({
            'params': criterion_metric.parameters(),
            'lr':     config.HYBRID_LEARNING_RATE
        })

    optimizer = optim.Adam(param_groups, weight_decay=1e-4)

    # -------------------------------------------------------------------------
    # 5. Ciclo de entrenamiento con AMP
    # -------------------------------------------------------------------------
    use_amp = (config.DEVICE == "cuda")
    scaler  = torch.amp.GradScaler('cuda') if use_amp else None

    best_map      = 0.0
    save_last     = getattr(config, 'SAVE_LAST_EPOCH', False)
    checkpoint_last = getattr(config, 'HYBRID_CHECKPOINT_LAST',
                              config.HYBRID_CHECKPOINT.replace('.pth', '_last.pth'))

    print("\n--- COMENZANDO CICLO DE ENTRENAMIENTO ---")
    for epoch in range(start_epoch, config.HYBRID_EPOCHS + 1):
        model.train()
        if use_pal:
            criterion_metric.train()

        train_loss, ce_loss_sum, metric_loss_sum, hard_frac_sum = 0.0, 0.0, 0.0, 0.0
        start_time = time.time()

        for images, labels in train_loader:
            images = images.to(config.DEVICE, non_blocking=True)
            labels = labels.to(config.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, embeddings = model(images)
                    loss_ce            = criterion_ce(logits, labels)
                    loss_metric, frac  = criterion_metric(embeddings, labels)
                    loss               = loss_ce + metric_weight * loss_metric
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits, embeddings = model(images)
                loss_ce            = criterion_ce(logits, labels)
                loss_metric, frac  = criterion_metric(embeddings, labels)
                loss               = loss_ce + metric_weight * loss_metric
                loss.backward()
                optimizer.step()

            train_loss      += loss.item()
            ce_loss_sum     += loss_ce.item()
            metric_loss_sum += loss_metric.item()
            hard_frac_sum   += frac

        steps      = len(train_loader)
        avg_ce     = ce_loss_sum     / steps
        avg_metric = metric_loss_sum / steps
        avg_hard   = (hard_frac_sum  / steps) * 100
        epoch_time = time.time() - start_time

        loss_label = "L_PAL" if use_pal else ("L_Circle" if config.USE_CIRCLE_LOSS else "L_Trip")
        print(f"Epoch {epoch:2d}/{config.HYBRID_EPOCHS} | "
              f"L_CE: {avg_ce:.3f} | {loss_label}: {avg_metric:.3f} | "
              f"Hard%: {avg_hard:.1f} | Time: {epoch_time:.1f}s")

        # -----------------------------------------------------------------
        # Mini-Val cada 5 épocas
        # -----------------------------------------------------------------
        if epoch % 5 == 0 or epoch == config.HYBRID_EPOCHS:
            model.eval()
            all_embeds, all_labels, all_conds = [], [], []

            with torch.no_grad():
                for v_data in val_loader:
                    v_img, v_lbl, v_cond, _, _ = v_data
                    v_img = v_img.to(config.DEVICE)
                    _, v_emb = model(v_img)
                    all_embeds.append(v_emb.cpu())
                    all_labels.extend(v_lbl.tolist())
                    all_conds.extend(v_cond)

            if all_embeds:
                all_embeds = torch.cat(all_embeds, dim=0)
                all_labels = np.array(all_labels)
                all_conds  = np.array(all_conds)

                # -------------------------------------------------------
                # Split Gallery/Query del Mini-Val según el dataset activo
                # -------------------------------------------------------
                # CASIA-B: gallery = nm-01..04, query = el resto (bg, cl, nm-05/06)
                # Gait3D:  no tiene condiciones fijas — se hace un split aleatorio
                #          por sujeto: primera mitad de secuencias = gallery,
                #          segunda mitad = query. Así siempre habrá ambas partes.
                if config.ACTIVE_DATASET == 'casiab':
                    g_mask = np.isin(all_conds, ['nm-01', 'nm-02', 'nm-03', 'nm-04'])
                    q_mask = ~g_mask
                else:
                    # Para Gait3D: split 50/50 por índice dentro de cada sujeto
                    unique_labels = np.unique(all_labels)
                    g_mask = np.zeros(len(all_labels), dtype=bool)
                    for uid in unique_labels:
                        idxs = np.where(all_labels == uid)[0]
                        mid  = max(1, len(idxs) // 2)
                        g_mask[idxs[:mid]] = True
                    q_mask = ~g_mask

                g_e, g_l = all_embeds[g_mask], all_labels[g_mask]
                q_e, q_l = all_embeds[q_mask], all_labels[q_mask]

                if len(q_e) > 0 and len(g_e) > 0:
                    dist_m      = torch.cdist(q_e, g_e, p=2)
                    v_r1, v_map = compute_mini_metrics(dist_m, q_l, g_l)
                    print(f"  -> Mini-Val | Rank-1: {v_r1:.1f}% | mAP: {v_map:.1f}%")

                    # Guardar siempre el checkpoint más reciente
                    # (Mini-Val está saturado en 100% → no es discriminativo)
                    # El checkpoint principal siempre tiene el modelo más entrenado.
                    torch.save({
                        'epoch':                epoch,
                        'model_state_dict':     model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_map':             v_map,
                        'num_classes':          num_classes,
                        'pal_state_dict':       criterion_metric.state_dict() if use_pal else None,
                        'use_pal':              use_pal,
                    }, config.HYBRID_CHECKPOINT)
                    if v_map > best_map:
                        best_map = v_map
                    print(f"  [✓] Checkpoint actualizado (época {epoch}, mAP Mini-Val: {v_map:.1f}%)")

        # -----------------------------------------------------------------
        # Guardar última época incondicionalmente para comparar en evaluate.py
        # -----------------------------------------------------------------
        if save_last and epoch == config.HYBRID_EPOCHS:
            torch.save({
                'epoch':                epoch,
                'model_state_dict':     model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes':          num_classes,
                'pal_state_dict':       criterion_metric.state_dict() if use_pal else None,
                'use_pal':              use_pal,
            }, checkpoint_last)
            print(f"  [✓] Última época guardada en: {checkpoint_last}")

    print(f"\n[+] ENTRENAMIENTO HÍBRIDO FINALIZADO.")
    print(f"    Mejor mAP Mini-Val: {best_map:.2f}%")
    print(f"    Checkpoint principal: {config.HYBRID_CHECKPOINT}")
    if save_last:
        print(f"    Checkpoint última época: {checkpoint_last}")

    if "--no-eval" not in sys.argv:
        print(f"[*] Lanzando Evaluación Exhaustiva...")
        import subprocess
        subprocess.run([sys.executable, "evaluate.py", "--tta", "--tta-n", "4"])


if __name__ == "__main__":
    settings.check_paths()

    torch.backends.cudnn.benchmark    = True
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32   = True
    torch.set_float32_matmul_precision('high')

    train_hybrid(settings)
