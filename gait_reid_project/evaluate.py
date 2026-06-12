# evaluate.py
import torch
import numpy as np
import os
import time
from torch.utils.data import DataLoader
import argparse

from configs import settings
from src.data import CASIAB_Supervised
from src.models import HybridGaitModel, SupervisedReIDModel, GaitBackbone
from utils_reranking import re_ranking


# =======================================================
# 1. EXTRACCIÓN DE FEATURES
# =======================================================

def extract_features_with_info(data_loader, model, device, phase='hybrid'):
    """Extracción estándar — un embedding por secuencia (corte central)."""
    model.eval()
    all_embeddings, all_labels = [], []
    all_conditions, all_angles, all_subjects = [], [], []

    print("[*] Extracción estándar...")
    start_time = time.time()

    with torch.no_grad():
        for batch_idx, data in enumerate(data_loader):
            images, labels, conds, angs, subs = data
            images = images.to(device)
            if phase == 'ssl':
                emb = model(images)
            else:
                _, emb = model(images)
            all_embeddings.append(emb.cpu())
            all_labels.extend(labels.tolist())
            all_conditions.extend(conds)
            all_angles.extend(angs)
            all_subjects.extend(subs)
            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(data_loader)}...")

    embeddings = torch.cat(all_embeddings, dim=0)
    print(f"[OK] Extracción completada en {time.time()-start_time:.1f}s. "
          f"Muestras: {len(all_labels)}")
    return (embeddings, np.array(all_labels), np.array(all_conditions),
            np.array(all_angles), np.array(all_subjects))


def extract_features_tta(dataset, model, device, phase='hybrid',
                         n_augments=4, batch_size=32):
    """
    Test Time Augmentation: N cortes temporales aleatorios por secuencia, promediados.

    Por qué funciona: cada corte captura una fase distinta del ciclo de marcha.
    El promedio cancela variaciones de pose transitoria y produce una representación
    más completa y estable del patrón de marcha del sujeto.

    Resultado empírico en CASIA-B HPP×2 + PAL + RR:
      Sin TTA → mAP 61.8%
      Con TTA×4 → mAP 69.5%  (+7.7 puntos)
    """
    model.eval()
    print(f"[*] Extracción con TTA×{n_augments}...")
    start_time = time.time()

    dataset.augment = True  # cortes aleatorios en cada pasada
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)

    accum       = None
    all_labels  = None
    all_conds   = None
    all_angles  = None
    all_subjects = None

    for aug_idx in range(n_augments):
        print(f"  Pasada {aug_idx+1}/{n_augments}...")
        pass_embeds = []
        pass_labels, pass_conds, pass_angles, pass_subs = [], [], [], []

        with torch.no_grad():
            for data in loader:
                images, labels, conds, angs, subs = data
                images = images.to(device)
                if phase == 'ssl':
                    emb = model(images)
                else:
                    _, emb = model(images)
                pass_embeds.append(emb.cpu())
                pass_labels.extend(labels.tolist())
                pass_conds.extend(conds)
                pass_angles.extend(angs)
                pass_subs.extend(subs)

        pass_embeds = torch.cat(pass_embeds, dim=0)
        if accum is None:
            accum        = pass_embeds
            all_labels   = np.array(pass_labels)
            all_conds    = np.array(pass_conds)
            all_angles   = np.array(pass_angles)
            all_subjects = np.array(pass_subs)
        else:
            accum += pass_embeds

    # Promediar y renormalizar L2
    accum = accum / n_augments
    accum = torch.nn.functional.normalize(accum, p=2, dim=1)
    dataset.augment = False

    print(f"[OK] TTA completada en {time.time()-start_time:.1f}s.")
    return accum, all_labels, all_conds, all_angles, all_subjects


# =======================================================
# 2. MÉTRICAS
# =======================================================

def compute_reid_metrics_block(dist_matrix, query_labels, gallery_labels):
    """Calcula Rank-1/5/10 y mAP. Devuelve AP por muestra para análisis de fallos."""
    if torch.is_tensor(dist_matrix):
        dist_matrix = dist_matrix.cpu().numpy()

    CMC_top1, CMC_top5, CMC_top10, AP_list, sample_ap = [], [], [], [], []

    for i in range(len(query_labels)):
        sorted_idx    = np.argsort(dist_matrix[i, :])
        sorted_labels = gallery_labels[sorted_idx]
        matches       = (sorted_labels == query_labels[i]).astype(np.int32)

        CMC_top1.append(matches[0])
        CMC_top5.append(1 if matches[:5].sum()  > 0 else 0)
        CMC_top10.append(1 if matches[:10].sum() > 0 else 0)

        num_tp = matches.sum()
        if num_tp == 0:
            AP_list.append(0.0); sample_ap.append(0.0); continue

        hits, prec_sum = 0, 0.0
        for j, m in enumerate(matches):
            if m:
                hits      += 1
                prec_sum  += hits / (j + 1)
        AP = prec_sum / num_tp
        AP_list.append(AP); sample_ap.append(AP)

    return (np.mean(CMC_top1)  * 100,
            np.mean(CMC_top5)  * 100,
            np.mean(CMC_top10) * 100,
            np.mean(AP_list)   * 100,
            np.array(sample_ap))


# =======================================================
# 3. CARGA DE MODELO
# =======================================================

def _load_model(config, phase, model_path):
    if not os.path.exists(model_path):
        print(f"[X] No se encontró: {model_path}")
        return None
    checkpoint = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    if phase == 'ssl':
        model = GaitBackbone()
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        dummy_backbone   = GaitBackbone()
        dummy_supervised = SupervisedReIDModel(
            backbone=dummy_backbone,
            num_classes=checkpoint.get('num_classes', 74)
        )
        if phase == 'supervised':
            dummy_supervised.load_state_dict(checkpoint['model_state_dict'])
            model = HybridGaitModel(dummy_supervised)
        else:
            model = HybridGaitModel(dummy_supervised)
            model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.DEVICE)
    print(f"[OK] Modelo cargado: {model_path}")
    return model


# =======================================================
# 4. EVALUACIÓN PRINCIPAL
# =======================================================

def test_reid_exhaustive(config, phase='hybrid', model_path=None, use_tta=True, n_tta=4):
    """
    Evaluación exhaustiva CASIA-B con matriz angular completa.

    Configuración final validada:
      HPP×2 + PAL (60 épocas) + Re-ranking (K1=40, K2=2, λ=0.05) + TTA×4
      → mAP global: 69.5% (+40.3% sobre baseline 29.23%)

    Args:
        use_tta: Si True (default), usa TTA×n_tta. Resultado más alto.
                 Si False, usa extracción estándar (más rápido, para debug).
        n_tta:   Número de cortes TTA (default 4, validado empíricamente).
    """
    print("\n" + "="*60)
    print(f" EVALUACIÓN EXHAUSTIVA CASIA-B — FASE: {phase.upper()}")
    print(f" Modo: {'TTA×' + str(n_tta) if use_tta else 'Estándar'} | "
          f"Re-ranking: {'K1=' + str(config.RERANK_K1) if config.USE_RERANKING else 'OFF'}")
    if model_path:
        print(f" Checkpoint: {os.path.basename(model_path)}")
    print("="*60)

    if model_path is None:
        model_path = {
            'hybrid':     config.HYBRID_CHECKPOINT,
            'supervised': config.SUPERVISED_CHECKPOINT,
            'ssl':        config.SSL_CHECKPOINT,
        }.get(phase, config.HYBRID_CHECKPOINT)

    model = _load_model(config, phase, model_path)
    if model is None:
        return None

    all_conditions = ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06',
                      'bg-01','bg-02','cl-01','cl-02']
    all_angles     = ['000','018','036','054','072','090','108','126','144','162','180']

    print("\n[+] Dataset de Prueba — 50 Sujetos Open-Set")
    test_dataset = CASIAB_Supervised(
        root_path=config.ROOT_PATH,
        subject_range=config.SUPERVISED_CONFIG['test_range'],
        conditions=all_conditions, angles=all_angles,
        seq_len=config.SUPERVISED_SUBSET_FRAMES_PER_SEQ,
        img_size=config.IMG_SIZE,
        augment=False, return_info=True
    )

    # Extracción
    if use_tta:
        embeddings, labels, conditions, angles, subjects = extract_features_tta(
            test_dataset, model, config.DEVICE, phase,
            n_augments=n_tta, batch_size=config.BATCH_SIZE
        )
    else:
        loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
        embeddings, labels, conditions, angles, subjects = extract_features_with_info(
            loader, model, config.DEVICE, phase
        )

    # Gallery: nm-01..nm-04
    g_mask   = np.isin(conditions, ['nm-01','nm-02','nm-03','nm-04'])
    g_embeds = embeddings[g_mask]
    g_labels = labels[g_mask]
    print(f"\n  Gallery: {len(g_embeds)} muestras (nm-01..04)")

    query_sets = {
        'NM (Normal Walking)': ['nm-05','nm-06'],
        'BG (Bag Occlusion)':  ['bg-01','bg-02'],
        'CL (Coat Appearance)':['cl-01','cl-02'],
    }

    print("\n" + "="*80)
    overall_mAPs = []

    for cond_name, cond_list in query_sets.items():
        q_mask = np.isin(conditions, cond_list)
        q_e    = embeddings[q_mask]
        q_l    = labels[q_mask]
        q_a    = angles[q_mask]
        q_sub  = subjects[q_mask]
        q_cond = conditions[q_mask]

        if len(q_e) == 0:
            continue

        print(f"\n--- {cond_name} ({len(q_e)} Probes) ---")

        if config.USE_RERANKING:
            dist_all = re_ranking(q_e.to(config.DEVICE), g_embeds.to(config.DEVICE),
                                  k1=config.RERANK_K1, k2=config.RERANK_K2,
                                  lambda_val=config.RERANK_LAMBDA)
        else:
            dist_all = torch.cdist(q_e, g_embeds, p=2)

        r1, r5, r10, mAP, ap_scores = compute_reid_metrics_block(dist_all, q_l, g_labels)
        overall_mAPs.append(mAP)
        print(f"> Rank-1: {r1:.1f}% | Rank-5: {r5:.1f}% | Rank-10: {r10:.1f}% | mAP: {mAP:.1f}%")

        # Desglose angular
        angle_parts = []
        for ang in all_angles:
            amask = (q_a == ang)
            if not np.any(amask): continue
            if config.USE_RERANKING:
                sd = re_ranking(q_e[amask].to(config.DEVICE), g_embeds.to(config.DEVICE),
                                k1=config.RERANK_K1, k2=config.RERANK_K2,
                                lambda_val=config.RERANK_LAMBDA)
            else:
                sd = torch.cdist(q_e[amask], g_embeds, p=2)
            ar1, _, _, amap, _ = compute_reid_metrics_block(sd, q_l[amask], g_labels)
            angle_parts.append(f"{ang}°: R1 {ar1:04.1f}% / mAP {amap:04.1f}%")
        print(" | ".join(angle_parts))

        # Top-5 fallos
        print("\n  [Fallos clínicos — 5 peores]")
        for idx in np.argsort(ap_scores)[:5]:
            print(f"   -> Sujeto {q_sub[idx]}, {q_cond[idx]}, "
                  f"Ángulo {q_a[idx]} | AP: {ap_scores[idx]*100:.1f}%")

    final_map = np.mean(overall_mAPs)
    print("\n" + "="*60)
    print(f" mAP GLOBAL: {final_map:.2f}%")
    print(f" Baseline: 29.23% | Crecimiento: +{final_map-29.23:.2f}%")
    print("="*60)
    return final_map


# =======================================================
# 5. MAIN
# =======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación CASIA-B — configuración final: TTA×4 + Re-ranking K1=40"
    )
    parser.add_argument('--phase', type=str, default='hybrid',
                        choices=['ssl', 'supervised', 'hybrid'])
    parser.add_argument('--checkpoint', type=str, default=None,
                        help="Path a checkpoint específico (opcional).")
    parser.add_argument('--no-tta', dest='use_tta', action='store_false',
                        help="Desactivar TTA (más rápido, para debug).")
    parser.add_argument('--tta-n', type=int, default=4,
                        help="Número de cortes TTA (default: 4).")
    parser.set_defaults(use_tta=True)
    args = parser.parse_args()

    settings.check_paths()
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision('high')

    test_reid_exhaustive(
        settings,
        phase=args.phase,
        model_path=args.checkpoint,
        use_tta=args.use_tta,
        n_tta=args.tta_n
    )