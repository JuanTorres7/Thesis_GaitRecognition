# evaluate.py
import torch
import numpy as np
import os
import time
from torch.utils.data import DataLoader
import argparse

from configs import settings
from src.data import get_supervised_dataset, Gait3D_Test
from src.models import HybridGaitModel, SupervisedReIDModel, GaitBackbone
from utils_reranking import re_ranking


# =======================================================
# 1. EXTRACCIÓN DE FEATURES
# =======================================================

def extract_features_with_info(data_loader, model, device, phase='hybrid'):
    """Extracción estándar — un embedding por secuencia."""
    model.eval()
    all_emb, all_lbl, all_cond, all_ang, all_sub = [], [], [], [], []
    print("[*] Extracción estándar...")
    t0 = time.time()
    with torch.no_grad():
        for b, data in enumerate(data_loader):
            imgs, lbls, conds, angs, subs = data
            imgs = imgs.to(device)
            emb  = model(imgs)[1] if phase != 'ssl' else model(imgs)
            all_emb.append(emb.cpu())
            all_lbl.extend(lbls.tolist())
            all_cond.extend(conds); all_ang.extend(angs); all_sub.extend(subs)
            if (b+1) % 10 == 0:
                print(f"  Batch {b+1}/{len(data_loader)}...")
    emb = torch.cat(all_emb, dim=0)
    print(f"[OK] {len(all_lbl)} muestras en {time.time()-t0:.1f}s")
    return emb, np.array(all_lbl), np.array(all_cond), np.array(all_ang), np.array(all_sub)


def extract_features_tta(dataset, model, device, phase='hybrid', n_augments=4, batch_size=32):
    """TTA: N cortes temporales por secuencia, promediados y renormalizados."""
    model.eval()
    print(f"[*] TTA×{n_augments}...")
    t0 = time.time()
    dataset.augment = True
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=2, pin_memory=True)
    accum = None
    meta  = None
    for i in range(n_augments):
        print(f"  Pasada {i+1}/{n_augments}...")
        pass_emb, pass_lbl, pass_cond, pass_ang, pass_sub = [], [], [], [], []
        with torch.no_grad():
            for data in loader:
                imgs, lbls, conds, angs, subs = data
                imgs = imgs.to(device)
                emb  = model(imgs)[1] if phase != 'ssl' else model(imgs)
                pass_emb.append(emb.cpu())
                pass_lbl.extend(lbls.tolist())
                pass_cond.extend(conds); pass_ang.extend(angs); pass_sub.extend(subs)
        pass_emb = torch.cat(pass_emb, dim=0)
        if accum is None:
            accum = pass_emb
            meta  = (np.array(pass_lbl), np.array(pass_cond),
                     np.array(pass_ang), np.array(pass_sub))
        else:
            accum += pass_emb
    accum = torch.nn.functional.normalize(accum / n_augments, p=2, dim=1)
    dataset.augment = False
    print(f"[OK] TTA completada en {time.time()-t0:.1f}s")
    return accum, meta[0], meta[1], meta[2], meta[3]


# =======================================================
# 2. MÉTRICAS
# =======================================================

def compute_reid_metrics_block(dist_matrix, query_labels, gallery_labels):
    if torch.is_tensor(dist_matrix): dist_matrix = dist_matrix.cpu().numpy()
    top1, top5, top10, aps, sample_ap = [], [], [], [], []
    for i in range(len(query_labels)):
        si  = np.argsort(dist_matrix[i])
        m   = (gallery_labels[si] == query_labels[i]).astype(np.int32)
        top1.append(m[0])
        top5.append(1 if m[:5].sum() > 0 else 0)
        top10.append(1 if m[:10].sum() > 0 else 0)
        ntp = m.sum()
        if ntp == 0:
            aps.append(0.); sample_ap.append(0.); continue
        hits, sp = 0, 0.
        for j, v in enumerate(m):
            if v: hits += 1; sp += hits/(j+1)
        ap = sp/ntp; aps.append(ap); sample_ap.append(ap)
    return (np.mean(top1)*100, np.mean(top5)*100, np.mean(top10)*100,
            np.mean(aps)*100, np.array(sample_ap))


# =======================================================
# 3. CARGA DE MODELO
# =======================================================

def _load_model(config, phase, model_path):
    if not os.path.exists(model_path):
        print(f"[X] No encontrado: {model_path}"); return None
    ckpt = torch.load(model_path, map_location=config.DEVICE, weights_only=False)
    if phase == 'ssl':
        m = GaitBackbone(); m.load_state_dict(ckpt['model_state_dict'])
    else:
        dummy = SupervisedReIDModel(GaitBackbone(),
                                    num_classes=ckpt.get('num_classes', 74))
        if phase == 'supervised':
            dummy.load_state_dict(ckpt['model_state_dict'])
            m = HybridGaitModel(dummy)
        else:
            m = HybridGaitModel(dummy)
            m.load_state_dict(ckpt['model_state_dict'])
    print(f"[OK] Modelo: {model_path}")
    return m.to(config.DEVICE)


# =======================================================
# 4. EVALUACIÓN PRINCIPAL — CASIA-B
# =======================================================

def evaluate_casiab(config, model, use_tta=True, n_tta=4):
    """Evaluación CASIA-B con matriz angular completa."""
    all_conditions = ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06',
                      'bg-01','bg-02','cl-01','cl-02']
    all_angles     = ['000','018','036','054','072','090','108','126','144','162','180']

    print("\n[+] Dataset Test CASIA-B — 50 sujetos open-set")
    test_ds = get_supervised_dataset(config, split='test', augment=False, return_info=True)

    if use_tta:
        emb, lbl, cond, ang, sub = extract_features_tta(
            test_ds, model, config.DEVICE, n_augments=n_tta,
            batch_size=config.BATCH_SIZE)
    else:
        loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
        emb, lbl, cond, ang, sub = extract_features_with_info(loader, model, config.DEVICE)

    g_mask   = np.isin(cond, ['nm-01','nm-02','nm-03','nm-04'])
    g_emb, g_lbl = emb[g_mask], lbl[g_mask]
    print(f"  Gallery: {len(g_emb)} | Modo: {'TTA×'+str(n_tta) if use_tta else 'Estándar'}")

    query_sets = {
        'NM': ['nm-05','nm-06'],
        'BG': ['bg-01','bg-02'],
        'CL': ['cl-01','cl-02'],
    }
    overall = []
    print("\n" + "="*80)

    for cname, clist in query_sets.items():
        qm  = np.isin(cond, clist)
        q_e = emb[qm]; q_l = lbl[qm]; q_a = ang[qm]
        q_s = sub[qm]; q_c = cond[qm]
        if len(q_e) == 0: continue
        print(f"\n--- {cname} ({len(q_e)} Probes) ---")

        dm = (re_ranking(q_e.to(config.DEVICE), g_emb.to(config.DEVICE),
                         k1=config.RERANK_K1, k2=config.RERANK_K2,
                         lambda_val=config.RERANK_LAMBDA)
              if config.USE_RERANKING
              else torch.cdist(q_e, g_emb, p=2))

        r1,r5,r10,mAP,aps = compute_reid_metrics_block(dm, q_l, g_lbl)
        overall.append(mAP)
        print(f"> R1:{r1:.1f}% R5:{r5:.1f}% R10:{r10:.1f}% mAP:{mAP:.1f}%")

        parts = []
        for ang_v in all_angles:
            am = (q_a == ang_v)
            if not np.any(am): continue
            sd = (re_ranking(q_e[am].to(config.DEVICE), g_emb.to(config.DEVICE),
                             k1=config.RERANK_K1, k2=config.RERANK_K2,
                             lambda_val=config.RERANK_LAMBDA)
                  if config.USE_RERANKING else torch.cdist(q_e[am], g_emb, p=2))
            ar1,_,_,am_,_ = compute_reid_metrics_block(sd, q_l[am], g_lbl)
            parts.append(f"{ang_v}°: R1 {ar1:04.1f}% / mAP {am_:04.1f}%")
        print(" | ".join(parts))

        print("\n  [Fallos — 5 peores]")
        for idx in np.argsort(aps)[:5]:
            print(f"   -> {q_s[idx]}, {q_c[idx]}, {q_a[idx]} | AP:{aps[idx]*100:.1f}%")

    final = np.mean(overall)
    print(f"\n{'='*60}\n mAP GLOBAL CASIA-B: {final:.2f}% (+{final-29.23:.2f}% vs baseline)\n{'='*60}")
    return final


# =======================================================
# 5. EVALUACIÓN PRINCIPAL — GAIT3D
# =======================================================

def evaluate_gait3d(config, model, use_tta=True, n_tta=4):
    """
    Evaluación Gait3D según protocolo oficial del JSON:
      Gallery = todas las secuencias de TEST_SET excepto las del PROBE_SET
      Query   = secuencias del PROBE_SET (1 por sujeto, 1000 total)
    """
    print("\n[+] Dataset Test Gait3D")
    test_ds = Gait3D_Test(
        root_path=config.GAIT3D_ROOT_PATH,
        json_path=config.GAIT3D_JSON_PATH,
        seq_len=config.GAIT3D_SEQ_LEN,
        img_size=config.GAIT3D_IMG_SIZE,
        return_info=True,
    )

    if use_tta:
        emb, lbl, cond, ang, sub = extract_features_tta(
            test_ds, model, config.DEVICE, n_augments=n_tta,
            batch_size=config.BATCH_SIZE)
    else:
        loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE,
                            shuffle=False, num_workers=2, pin_memory=True)
        emb, lbl, cond, ang, sub = extract_features_with_info(loader, model, config.DEVICE)

    # Gait3D: condition = 'probe' | 'gallery' (asignado en Gait3D_Test)
    g_mask = (cond == 'gallery')
    q_mask = (cond == 'probe')
    g_emb, g_lbl = emb[g_mask], lbl[g_mask]
    q_emb, q_lbl = emb[q_mask], lbl[q_mask]
    q_sub         = sub[q_mask]

    print(f"  Gallery: {len(g_emb)} | Queries (probe): {len(q_emb)}")
    print(f"  Modo: {'TTA×'+str(n_tta) if use_tta else 'Estándar'}")

    if len(q_emb) == 0 or len(g_emb) == 0:
        print("[!] Gallery o queries vacías — verificar rutas y JSON.")
        return 0.0

    dm = (re_ranking(q_emb.to(config.DEVICE), g_emb.to(config.DEVICE),
                     k1=config.RERANK_K1, k2=config.RERANK_K2,
                     lambda_val=config.RERANK_LAMBDA)
          if config.USE_RERANKING
          else torch.cdist(q_emb, g_emb, p=2))

    r1, r5, r10, mAP, aps = compute_reid_metrics_block(dm, q_lbl, g_lbl)

    print(f"\n{'='*60}")
    print(f" RESULTADOS GAIT3D")
    print(f" Rank-1: {r1:.2f}% | Rank-5: {r5:.2f}% | Rank-10: {r10:.2f}% | mAP: {mAP:.2f}%")

    print("\n  [Fallos — 5 peores]")
    for idx in np.argsort(aps)[:5]:
        print(f"   -> Sujeto {q_sub[idx]} | AP: {aps[idx]*100:.1f}%")
    print("="*60)
    return mAP


# =======================================================
# 6. DISPATCHER — elige evaluación según dataset activo
# =======================================================

def test_reid_exhaustive(config, phase='hybrid', model_path=None,
                         use_tta=True, n_tta=4):
    """Punto de entrada unificado — enruta a CASIA-B o Gait3D."""
    print(f"\n{'='*60}")
    print(f" EVALUACIÓN [{config.ACTIVE_DATASET.upper()}] — {phase.upper()}")
    print(f" Modo: {'TTA×'+str(n_tta) if use_tta else 'Estándar'} | "
          f"RR: {'K1='+str(config.RERANK_K1) if config.USE_RERANKING else 'OFF'}")
    print("="*60)

    if model_path is None:
        model_path = {'hybrid':     config.HYBRID_CHECKPOINT,
                      'supervised': config.SUPERVISED_CHECKPOINT,
                      'ssl':        config.SSL_CHECKPOINT}.get(phase)

    model = _load_model(config, phase, model_path)
    if model is None: return None

    if config.ACTIVE_DATASET == 'casiab':
        return evaluate_casiab(config, model, use_tta=use_tta, n_tta=n_tta)
    elif config.ACTIVE_DATASET == 'gait3d':
        return evaluate_gait3d(config, model, use_tta=use_tta, n_tta=n_tta)
    else:
        raise ValueError(f"Dataset no reconocido: {config.ACTIVE_DATASET}")


# =======================================================
# 7. MAIN
# =======================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluación — CASIA-B y Gait3D"
    )
    parser.add_argument('--phase', default='hybrid',
                        choices=['ssl','supervised','hybrid'])
    parser.add_argument('--checkpoint', default=None)
    parser.add_argument('--no-tta', dest='use_tta', action='store_false')
    parser.add_argument('--tta-n', type=int, default=4)
    parser.set_defaults(use_tta=True)
    args = parser.parse_args()

    settings.check_paths()
    torch.backends.cudnn.benchmark        = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32       = True
    torch.set_float32_matmul_precision('high')

    test_reid_exhaustive(settings, phase=args.phase,
                         model_path=args.checkpoint,
                         use_tta=args.use_tta, n_tta=args.tta_n)
