# src/data.py
import torch
import torch.nn.functional as F_interp
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as F_vision
from PIL import Image
from pathlib import Path
import numpy as np
import random
import json


# =============================================================================
# UTILIDADES COMUNES
# =============================================================================

def center_of_mass_align(img_tensor):
    """Centra silueta por Centro de Masa. Común a todos los datasets."""
    assert img_tensor.dim() == 3 and img_tensor.size(0) == 1
    np_img = img_tensor.squeeze().cpu().numpy()
    y_coords, x_coords = np.nonzero(np_img > 0.1)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return img_tensor
    cy = int(np.mean(y_coords))
    cx = int(np.mean(x_coords))
    H, W = np_img.shape
    dy = H // 2 - cy
    dx = W // 2 - cx
    return F_vision.affine(
        img_tensor, angle=0.0, translate=[dx, dy],
        scale=1.0, shear=[0.0, 0.0],
        interpolation=F_vision.InterpolationMode.BILINEAR
    )


# =============================================================================
# AUMENTACIONES π(·) — compartidas por todos los datasets
# =============================================================================

def gait_add_noise(volume: torch.Tensor, noise_prob: float = 0.3,
                   noise_std: float = 0.05) -> torch.Tensor:
    """Ruido gaussiano — simula siluetas ruidosas de Gait3D outdoor."""
    if random.random() >= noise_prob:
        return volume
    noise = torch.randn_like(volume) * noise_std
    return torch.clamp(volume + noise, 0.0, 1.0)


def gait_erode_volume(volume: torch.Tensor, erode_prob: float = 0.3) -> torch.Tensor:
    """Erosión morfológica — simula variaciones de segmentación cross-dataset."""
    if random.random() >= erode_prob:
        return volume
    T, C, H, W = volume.shape
    vol_flat = volume.view(T * C, 1, H, W)
    eroded   = F_interp.avg_pool2d(vol_flat, kernel_size=3, stride=1, padding=1)
    eroded   = (eroded > 0.4).float()
    return eroded.view(T, C, H, W)


def gait_erase_volume(volume: torch.Tensor,
                      erase_prob: float = 0.5,
                      torso_only: bool = True) -> torch.Tensor:
    """Random Erasing horizontal — simula BG/CL en CASIA-B y oclusión en Gait3D."""
    if random.random() >= erase_prob:
        return volume
    T, C, H, W = volume.shape
    out = volume.clone()
    eh  = random.randint(int(H * 0.15), int(H * 0.35))
    ey  = random.randint(0, max(0, H // 3 - eh)) if torso_only else random.randint(0, max(0, H - eh))
    ew  = random.randint(int(W * 0.40), int(W * 0.80))
    ex  = random.randint(0, max(0, W - ew))
    out[:, :, ey:ey + eh, ex:ex + ew] = 0.0
    return out


def apply_ssl_augmentations(view: torch.Tensor,
                             erase_prob: float = 0.5,
                             torso_only: bool = True) -> torch.Tensor:
    """Aplica el pipeline completo de aumentaciones SSL sobre un volumen."""
    view = gait_erase_volume(view, erase_prob, torso_only)
    view = gait_add_noise(view, noise_prob=0.3, noise_std=0.05)
    view = gait_erode_volume(view, erode_prob=0.2)
    return view


# =============================================================================
# CASIA-B SSL
# =============================================================================

class CASIAB_SSL(Dataset):
    """
    Dataset Fase 1 SSL para CASIA-B.
    Devuelve dos vistas volumétricas [T,C,H,W] del mismo video.
    """
    def __init__(self, root_path, img_size=(64, 64), use_subset=True,
                 subset_subjects=74, subset_conditions=None, subset_angles=None,
                 seq_len=24, use_augmentation=True, erase_prob=0.5,
                 torso_only=True, scale_prob=0.5, scale_range=(0.85, 1.15),
                 use_cache=False):

        self.seq_len          = seq_len
        self.img_size         = img_size
        self.use_augmentation = use_augmentation
        self.erase_prob       = erase_prob
        self.torso_only       = torso_only
        self.sequences        = []

        root         = Path(root_path)
        all_subjects = sorted([d for d in root.iterdir() if d.is_dir()])
        subjects     = all_subjects[:subset_subjects] if use_subset else all_subjects

        print(f"\n[CASIA-B SSL] Cargando {len(subjects)} sujetos...")
        print(f"  Aumentaciones π(·): {'ACTIVADAS' if use_augmentation else 'desactivadas'}")

        for subject_dir in subjects:
            for condition_dir in subject_dir.iterdir():
                if not condition_dir.is_dir(): continue
                if subset_conditions and condition_dir.name not in subset_conditions: continue
                for angle_dir in condition_dir.iterdir():
                    if not angle_dir.is_dir(): continue
                    if subset_angles and angle_dir.name not in subset_angles: continue
                    frames = sorted([f for f in angle_dir.iterdir()
                                     if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    if len(frames) >= seq_len:
                        self.sequences.append(frames)

        print(f"  Secuencias (>= {seq_len} frames): {len(self.sequences)}")
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])

    def _sample_frames(self, frames, start_idx):
        sampled = []
        for i in range(self.seq_len):
            idx   = min(start_idx + i, len(frames) - 1)
            img   = Image.open(frames[idx]).convert("L")
            img_t = self.base_transform(img)
            img_t = center_of_mass_align(img_t)
            sampled.append(img_t)
        return torch.stack(sampled, dim=0)

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        frames    = self.sequences[idx]
        total_len = len(frames)
        start1    = random.randint(0, total_len - self.seq_len) if total_len > self.seq_len else 0
        start2    = random.randint(0, total_len - self.seq_len) if total_len > self.seq_len else 0
        view1     = self._sample_frames(frames, start1)
        view2     = self._sample_frames(frames, start2)
        if self.use_augmentation:
            view1 = apply_ssl_augmentations(view1, self.erase_prob, self.torso_only)
            view2 = apply_ssl_augmentations(view2, self.erase_prob, self.torso_only)
        return view1, view2


# =============================================================================
# CASIA-B SUPERVISADO
# =============================================================================

class CASIAB_Supervised(Dataset):
    """
    Dataset PKSampler-Friendly para Fases 2 y 3 en CASIA-B.
    Devuelve (tensor [T,C,H,W], label) o con return_info=True añade metadatos.
    """
    def __init__(self, root_path, subject_range, conditions, angles=None,
                 seq_len=24, img_size=(64, 64), augment=False, return_info=False):

        self.root        = Path(root_path)
        self.seq_len     = seq_len
        self.img_size    = img_size
        self.augment     = augment
        self.return_info = return_info
        self.samples     = []
        self.subject_to_label = {}

        all_subjects    = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        start_idx, end_idx = subject_range
        subjects        = all_subjects[start_idx:end_idx]

        print(f"\n[CASIA-B Supervised] Rango: {start_idx+1:03d}-{end_idx:03d} ({len(subjects)} sujetos)")

        for label_id, subject_name in enumerate(subjects):
            self.subject_to_label[subject_name] = label_id

        for subject_name in subjects:
            subject_dir = self.root / subject_name
            label_id    = self.subject_to_label[subject_name]
            for condition_dir in subject_dir.iterdir():
                if not condition_dir.is_dir() or condition_dir.name not in conditions: continue
                for angle_dir in condition_dir.iterdir():
                    if not angle_dir.is_dir(): continue
                    if angles and angle_dir.name not in angles: continue
                    frames = sorted([f for f in angle_dir.iterdir()
                                     if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    if len(frames) >= self.seq_len:
                        self.samples.append({
                            'frames':    frames,
                            'label':     label_id,
                            'subject':   subject_name,
                            'condition': condition_dir.name,
                            'angle':     angle_dir.name,
                        })

        print(f"  Secuencias: {len(self.samples)} | Clases: {len(self.subject_to_label)}")
        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        frames    = sample['frames']
        total_len = len(frames)

        do_scale, scale_factor = False, 1.0
        do_erase, erase_params = False, {}

        if self.augment:
            start_idx = random.randint(0, total_len - self.seq_len)
            if random.random() < 0.5:
                do_scale     = True
                scale_factor = random.uniform(0.85, 1.15)
            if random.random() < 0.3:
                do_erase = True
                H, W = self.img_size
                eh = random.randint(int(H * 0.05), int(H * 0.15))
                ew = random.randint(int(W * 0.05), int(W * 0.15))
                erase_params = {'y': random.randint(0, H-eh), 'x': random.randint(0, W-ew),
                                'h': eh, 'w': ew}
        else:
            start_idx = (total_len - self.seq_len) // 2

        sampled_tensors = []
        for i in range(self.seq_len):
            idx_f = min(start_idx + i, total_len - 1)
            img   = Image.open(frames[idx_f]).convert("L")
            if do_scale:
                w_orig, h_orig = img.size
                new_w = int(w_orig * scale_factor)
                img_s = img.resize((new_w, h_orig), Image.BILINEAR)
                new_img = Image.new("L", (w_orig, h_orig), 0)
                if scale_factor < 1.0:
                    new_img.paste(img_s, ((w_orig - new_w) // 2, 0))
                else:
                    offset = (new_w - w_orig) // 2
                    new_img.paste(img_s.crop((offset, 0, offset + w_orig, h_orig)))
                img = new_img
            img_t = self.base_transform(img)
            img_t = center_of_mass_align(img_t)
            if do_erase:
                img_t[:, erase_params['y']:erase_params['y']+erase_params['h'],
                         erase_params['x']:erase_params['x']+erase_params['w']] = 0.0
            sampled_tensors.append(img_t)

        seq_tensor = torch.stack(sampled_tensors, dim=0)
        if self.return_info:
            return seq_tensor, sample['label'], sample['condition'], sample['angle'], sample['subject']
        return seq_tensor, sample['label']

    def get_num_classes(self): return len(self.subject_to_label)


# =============================================================================
# GAIT3D — Dataset loader
# =============================================================================
# Estructura de carpetas:
#   {root_path}/{subject_id}/{camidX_videoidY}/{seqZ}/{frame_XXXXX.png}
#
# Ejemplo:
#   2D_Silhouettes/3999/camid1_videoid3/seq0/human_crop_f24551.png
#
# El JSON define:
#   TRAIN_SET:  lista de subject_ids (3000 sujetos) → Fases 1, 2, 3
#   TEST_SET:   lista de subject_ids (1000 sujetos) → Gallery en evaluación
#   PROBE_SET:  lista de "subjectid-camidX_videoidY-seqZ" → Queries exactas
#
# Protocolo de evaluación Gait3D:
#   - Gallery = TODAS las secuencias de los sujetos en TEST_SET
#               excepto las secuencias en PROBE_SET
#   - Query   = Las secuencias específicas en PROBE_SET (1 por sujeto)
# =============================================================================

class Gait3D_SSL(Dataset):
    """
    Dataset Fase 1 SSL para Gait3D.
    Usa los sujetos de TRAIN_SET del JSON.
    Devuelve dos vistas volumétricas con aumentaciones cross-domain.
    """
    def __init__(self, root_path, json_path, img_size=(64, 44),
                 seq_len=24, use_augmentation=True,
                 erase_prob=0.5, torso_only=True):

        self.seq_len          = seq_len
        self.img_size         = img_size
        self.use_augmentation = use_augmentation
        self.erase_prob       = erase_prob
        self.torso_only       = torso_only
        self.sequences        = []

        root = Path(root_path)
        with open(json_path, 'r') as f:
            split_data = json.load(f)
        train_subjects = set(split_data['TRAIN_SET'])

        print(f"\n[Gait3D SSL] Cargando {len(train_subjects)} sujetos de TRAIN_SET...")
        print(f"  Aumentaciones π(·): {'ACTIVADAS' if use_augmentation else 'desactivadas'}")

        found = 0
        for subject_id in sorted(train_subjects):
            subject_dir = root / subject_id
            if not subject_dir.exists():
                continue
            found += 1
            # Iterar cam_video dirs: camid0_videoid1, camid0_videoid2, ...
            for cam_video_dir in sorted(subject_dir.iterdir()):
                if not cam_video_dir.is_dir(): continue
                # Iterar seq dirs: seq0, seq1, ...
                for seq_dir in sorted(cam_video_dir.iterdir()):
                    if not seq_dir.is_dir(): continue
                    frames = sorted([f for f in seq_dir.iterdir()
                                     if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    if len(frames) >= seq_len:
                        self.sequences.append(frames)

        print(f"  Sujetos encontrados en disco: {found}/{len(train_subjects)}")
        print(f"  Secuencias (>= {seq_len} frames): {len(self.sequences)}")

        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])

    def _sample_frames(self, frames, start_idx):
        sampled = []
        for i in range(self.seq_len):
            idx   = min(start_idx + i, len(frames) - 1)
            img   = Image.open(frames[idx]).convert("L")
            img_t = self.base_transform(img)
            img_t = center_of_mass_align(img_t)
            sampled.append(img_t)
        return torch.stack(sampled, dim=0)

    def __len__(self): return len(self.sequences)

    def __getitem__(self, idx):
        frames    = self.sequences[idx]
        total_len = len(frames)
        start1    = random.randint(0, total_len - self.seq_len) if total_len > self.seq_len else 0
        start2    = random.randint(0, total_len - self.seq_len) if total_len > self.seq_len else 0
        view1     = self._sample_frames(frames, start1)
        view2     = self._sample_frames(frames, start2)
        if self.use_augmentation:
            view1 = apply_ssl_augmentations(view1, self.erase_prob, self.torso_only)
            view2 = apply_ssl_augmentations(view2, self.erase_prob, self.torso_only)
        return view1, view2


class Gait3D_Supervised(Dataset):
    """
    Dataset PKSampler-Friendly para Fases 2 y 3 en Gait3D.
    Usa los sujetos de TRAIN_SET del JSON.

    Cada muestra es una secuencia completa identificada por
    (subject_id, cam_video, seq_dir).

    return_info=True devuelve:
        (tensor, label, seq_key, seq_key, subject_id)
        donde seq_key = "camidX_videoidY/seqZ" — equivalente a condition/angle en CASIA-B.
    """
    def __init__(self, root_path, json_path, split='train',
                 seq_len=24, img_size=(64, 44), augment=False, return_info=False,
                 max_subjects=None):
        """
        Args:
            max_subjects: Si se especifica, limita el número de sujetos cargados.
                          Útil para Mini-Val rápido (max_subjects=20).
        """
        self.root        = Path(root_path)
        self.seq_len     = seq_len
        self.img_size    = img_size
        self.augment     = augment
        self.return_info = return_info
        self.samples     = []
        self.subject_to_label = {}

        with open(json_path, 'r') as f:
            split_data = json.load(f)

        if split == 'train':
            subject_ids = sorted(split_data['TRAIN_SET'])
        elif split == 'test':
            subject_ids = sorted(split_data['TEST_SET'])
        else:
            raise ValueError(f"Split no reconocido: {split}. Opciones: 'train', 'test'")

        # Limitar sujetos si se especifica (para Mini-Val rápido)
        if max_subjects is not None:
            subject_ids = subject_ids[:max_subjects]

        print(f"\n[Gait3D Supervised] Split: {split.upper()} — {len(subject_ids)} sujetos"
              + (f" (limitado a {max_subjects})" if max_subjects else ""))

        for label_id, subject_id in enumerate(subject_ids):
            self.subject_to_label[subject_id] = label_id

        found_subjects = 0
        for subject_id in subject_ids:
            subject_dir = self.root / subject_id
            if not subject_dir.exists():
                continue
            found_subjects += 1
            label_id = self.subject_to_label[subject_id]

            for cam_video_dir in sorted(subject_dir.iterdir()):
                if not cam_video_dir.is_dir(): continue
                for seq_dir in sorted(cam_video_dir.iterdir()):
                    if not seq_dir.is_dir(): continue
                    frames = sorted([f for f in seq_dir.iterdir()
                                     if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    if len(frames) >= self.seq_len:
                        # seq_key es el identificador único de la secuencia
                        seq_key = f"{cam_video_dir.name}/{seq_dir.name}"
                        self.samples.append({
                            'frames':    frames,
                            'label':     label_id,
                            'subject':   subject_id,
                            'seq_key':   seq_key,
                            # Aliases para compatibilidad con evaluate.py
                            'condition': seq_key,
                            'angle':     seq_key,
                        })

        print(f"  Sujetos en disco: {found_subjects}/{len(subject_ids)}")
        print(f"  Secuencias totales: {len(self.samples)} | Clases: {len(self.subject_to_label)}")

        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        frames    = sample['frames']
        total_len = len(frames)

        if self.augment:
            start_idx = random.randint(0, total_len - self.seq_len)
        else:
            start_idx = (total_len - self.seq_len) // 2

        sampled_tensors = []
        for i in range(self.seq_len):
            idx_f = min(start_idx + i, total_len - 1)
            img   = Image.open(frames[idx_f]).convert("L")
            img_t = self.base_transform(img)
            img_t = center_of_mass_align(img_t)
            sampled_tensors.append(img_t)

        seq_tensor = torch.stack(sampled_tensors, dim=0)

        if self.return_info:
            return (seq_tensor, sample['label'],
                    sample['condition'], sample['angle'], sample['subject'])
        return seq_tensor, sample['label']

    def get_num_classes(self): return len(self.subject_to_label)


class Gait3D_Test(Dataset):
    """
    Dataset de evaluación para Gait3D.
    Carga TODAS las secuencias de los sujetos en TEST_SET.
    El split Gallery/Query se hace en evaluate.py usando PROBE_SET del JSON:
      - Query:   secuencias que aparecen en PROBE_SET
      - Gallery: resto de secuencias del mismo sujeto

    return_info=True devuelve:
        (tensor, label, is_probe_str, seq_key, subject_id)
        donde is_probe_str = 'probe' si la secuencia está en PROBE_SET, 'gallery' si no.
    """
    def __init__(self, root_path, json_path,
                 seq_len=24, img_size=(64, 44), return_info=False):

        self.root        = Path(root_path)
        self.seq_len     = seq_len
        self.img_size    = img_size
        self.augment     = False
        self.return_info = return_info
        self.samples     = []
        self.subject_to_label = {}

        with open(json_path, 'r') as f:
            split_data = json.load(f)

        test_subjects = sorted(split_data['TEST_SET'])

        # Construir set de claves probe para lookup O(1)
        # Formato probe: "0002-camid33_videoid2-seq0"
        # → subject=0002, cam_video=camid33_videoid2, seq=seq0
        probe_set = set()
        for entry in split_data['PROBE_SET']:
            parts   = entry.split('-')
            subject = parts[0]
            cam_video = parts[1]
            seq     = parts[2]
            probe_set.add(f"{subject}/{cam_video}/{seq}")

        print(f"\n[Gait3D Test] {len(test_subjects)} sujetos de TEST_SET")
        print(f"  PROBE_SET: {len(probe_set)} secuencias query")

        for label_id, subject_id in enumerate(test_subjects):
            self.subject_to_label[subject_id] = label_id

        found_subjects = 0
        probe_count, gallery_count = 0, 0

        for subject_id in test_subjects:
            subject_dir = self.root / subject_id
            if not subject_dir.exists(): continue
            found_subjects += 1
            label_id = self.subject_to_label[subject_id]

            for cam_video_dir in sorted(subject_dir.iterdir()):
                if not cam_video_dir.is_dir(): continue
                for seq_dir in sorted(cam_video_dir.iterdir()):
                    if not seq_dir.is_dir(): continue
                    frames = sorted([f for f in seq_dir.iterdir()
                                     if f.suffix.lower() in ['.png', '.jpg', '.bmp']])
                    if len(frames) < self.seq_len:
                        continue

                    seq_key  = f"{subject_id}/{cam_video_dir.name}/{seq_dir.name}"
                    is_probe = seq_key in probe_set
                    role     = 'probe' if is_probe else 'gallery'

                    if is_probe: probe_count += 1
                    else:        gallery_count += 1

                    self.samples.append({
                        'frames':    frames,
                        'label':     label_id,
                        'subject':   subject_id,
                        'seq_key':   seq_key,
                        'is_probe':  role,
                        # Aliases para compatibilidad con evaluate.py
                        'condition': role,
                        'angle':     f"{cam_video_dir.name}_{seq_dir.name}",
                    })

        print(f"  Sujetos en disco: {found_subjects}/{len(test_subjects)}")
        print(f"  Secuencias gallery: {gallery_count} | probe (query): {probe_count}")

        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        frames    = sample['frames']
        total_len = len(frames)
        start_idx = (total_len - self.seq_len) // 2

        sampled_tensors = []
        for i in range(self.seq_len):
            idx_f = min(start_idx + i, total_len - 1)
            img   = Image.open(frames[idx_f]).convert("L")
            img_t = self.base_transform(img)
            img_t = center_of_mass_align(img_t)
            sampled_tensors.append(img_t)

        seq_tensor = torch.stack(sampled_tensors, dim=0)
        if self.return_info:
            return (seq_tensor, sample['label'],
                    sample['condition'], sample['angle'], sample['subject'])
        return seq_tensor, sample['label']

    def get_num_classes(self): return len(self.subject_to_label)


# =============================================================================
# FACTORY — obtener datasets según dataset activo
# =============================================================================

def get_ssl_dataset(config):
    """Devuelve el dataset SSL correcto según config.ACTIVE_DATASET."""
    if config.ACTIVE_DATASET == 'casiab':
        return CASIAB_SSL(
            root_path=config.CASIAB_ROOT_PATH,
            img_size=config.CASIAB_IMG_SIZE,
            use_subset=config.SSL_USE_SUBSET,
            subset_subjects=config.SSL_SUBSET_SUBJECTS,
            subset_conditions=config.SSL_SUBSET_CONDITIONS,
            subset_angles=config.SSL_SUBSET_ANGLES,
            seq_len=config.CASIAB_SEQ_LEN,
            use_augmentation=config.SSL_USE_GAIT_AUGMENTATION,
            erase_prob=config.SSL_ERASE_PROB,
            torso_only=config.SSL_ERASE_TORSO_ONLY,
        )
    elif config.ACTIVE_DATASET == 'gait3d':
        return Gait3D_SSL(
            root_path=config.GAIT3D_ROOT_PATH,
            json_path=config.GAIT3D_JSON_PATH,
            img_size=config.GAIT3D_IMG_SIZE,
            seq_len=config.GAIT3D_SEQ_LEN,
            use_augmentation=config.SSL_USE_GAIT_AUGMENTATION,
            erase_prob=config.SSL_ERASE_PROB,
            torso_only=config.SSL_ERASE_TORSO_ONLY,
        )
    else:
        raise ValueError(f"Dataset no reconocido: {config.ACTIVE_DATASET}")


def get_supervised_dataset(config, split='train', augment=False, return_info=False):
    """Devuelve el dataset supervisado correcto según config.ACTIVE_DATASET."""
    if config.ACTIVE_DATASET == 'casiab':
        if split == 'train':
            subject_range = config.CASIAB_CONFIG['train_range']
        elif split == 'val':
            subject_range = config.CASIAB_CONFIG['val_range']
        elif split == 'test':
            subject_range = config.CASIAB_CONFIG['test_range']
        else:
            raise ValueError(f"Split no reconocido para CASIA-B: {split}")
        return CASIAB_Supervised(
            root_path=config.CASIAB_ROOT_PATH,
            subject_range=subject_range,
            conditions=config.CASIAB_CONFIG['conditions'],
            angles=config.CASIAB_CONFIG['angles'],
            seq_len=config.CASIAB_SEQ_LEN,
            img_size=config.CASIAB_IMG_SIZE,
            augment=augment,
            return_info=return_info,
        )
    elif config.ACTIVE_DATASET == 'gait3d':
        if split == 'train':
            return Gait3D_Supervised(
                root_path=config.GAIT3D_ROOT_PATH,
                json_path=config.GAIT3D_JSON_PATH,
                split='train',
                seq_len=config.GAIT3D_SEQ_LEN,
                img_size=config.GAIT3D_IMG_SIZE,
                augment=augment,
                return_info=return_info,
            )
        elif split == 'val':
            # Gait3D no tiene val_range oficial.
            # Para Mini-Val usamos los primeros 20 sujetos de TRAIN_SET —
            # suficiente para monitorear convergencia sin costo computacional alto.
            return Gait3D_Supervised(
                root_path=config.GAIT3D_ROOT_PATH,
                json_path=config.GAIT3D_JSON_PATH,
                split='train',
                seq_len=config.GAIT3D_SEQ_LEN,
                img_size=config.GAIT3D_IMG_SIZE,
                augment=False,
                return_info=return_info,
                max_subjects=20,   # limitar para que Mini-Val sea rápido
            )
        elif split == 'test':
            return Gait3D_Test(
                root_path=config.GAIT3D_ROOT_PATH,
                json_path=config.GAIT3D_JSON_PATH,
                seq_len=config.GAIT3D_SEQ_LEN,
                img_size=config.GAIT3D_IMG_SIZE,
                return_info=return_info,
            )
    else:
        raise ValueError(f"Dataset no reconocido: {config.ACTIVE_DATASET}")
