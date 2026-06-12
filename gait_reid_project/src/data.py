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


def center_of_mass_align(img_tensor):
    """
    Centra la silueta binaria por Centro de Masa. Sin cambios respecto al original.
    """
    assert img_tensor.dim() == 3 and img_tensor.size(0) == 1
    np_img = img_tensor.squeeze().cpu().numpy()
    y_coords, x_coords = np.nonzero(np_img > 0.1)
    if len(y_coords) == 0 or len(x_coords) == 0:
        return img_tensor
    cy, cx = int(np.mean(y_coords)), int(np.mean(x_coords))
    H, W = np_img.shape
    dy = H // 2 - cy
    dx = W // 2 - cx
    return F_vision.affine(
        img_tensor, angle=0.0, translate=[dx, dy],
        scale=1.0, shear=[0.0, 0.0],
        interpolation=F_vision.InterpolationMode.BILINEAR
    )


# =============================================================================
# Aumentaciones π(·) — versión BARATA sin F.interpolate
# =============================================================================
# El problema de la versión anterior:
#   gait_augment_volume() llamaba F.interpolate sobre [T,C,H,W] en CPU.
#   Con num_workers=0 en Windows esto bloqueaba el hilo principal 2 veces
#   por batch (una por cada vista). Era más lento que el SSL original.
#
# Solución: solo Random Erasing (zeroing de píxeles = O(1), sin resize).
#   - Simula abrigo/bolso igual de bien que el scaling
#   - Costo: asignación de ceros sobre un slice de tensor → casi gratuito
#   - El scaling se elimina: la invarianza de escala la aprende el CoM alignment
#     más las capas convolucionales del backbone, no hace falta forzarla aquí
# =============================================================================

# =============================================================================
# Aumentaciones cross-domain para generalización a otros datasets
# =============================================================================
# Gait3D tiene siluetas ruidosas (outdoor, oclusión parcial, segmentación imperfecta).
# OU-MVLP tiene aspecto ratio distinto (64×44 vs 64×64).
# Estas aumentaciones simulan esas distribuciones durante SSL para que el backbone
# aprenda representaciones robustas ANTES de ver esos datasets en test.
# Costo computacional: O(1) — solo operaciones sobre tensores existentes.

def gait_add_noise(volume: torch.Tensor, noise_prob: float = 0.3,
                   noise_std: float = 0.05) -> torch.Tensor:
    """
    Añade ruido gaussiano sobre la silueta para simular segmentaciones ruidosas
    de Gait3D (outdoor, oclusión, sombras). Costo: una sola operación randn.
    """
    if random.random() >= noise_prob:
        return volume
    noise = torch.randn_like(volume) * noise_std
    return torch.clamp(volume + noise, 0.0, 1.0)


def gait_erode_volume(volume: torch.Tensor, erode_prob: float = 0.3) -> torch.Tensor:
    """
    Simula erosión morfológica (siluetas más delgadas) usando max pooling inverso.
    Modela siluetas de sujetos delgados o con ropa ajustada en Gait3D.
    Implementación: erosión = 1 - dilation(1 - mask) usando avg_pool2d.
    """
    if random.random() >= erode_prob:
        return volume
    T, C, H, W = volume.shape
    # Erosión simple: reducir píxeles de borde usando avg_pool y umbral
    vol_flat = volume.view(T * C, 1, H, W)
    eroded   = F_interp.avg_pool2d(vol_flat, kernel_size=3, stride=1, padding=1)
    eroded   = (eroded > 0.4).float()  # umbral para mantener estructura binaria
    return eroded.view(T, C, H, W)


def gait_erase_volume(volume: torch.Tensor,
                      erase_prob: float = 0.5,
                      torso_only: bool = True) -> torch.Tensor:
    """
    Aplica Random Erasing horizontal sobre volumen [T, C, H, W].
    Solo zeroing de píxeles — sin resize, sin interpolate.
    Consistente en el tiempo: el mismo parche se aplica a todos los T frames.
    """
    if random.random() >= erase_prob:
        return volume

    T, C, H, W = volume.shape
    out = volume.clone()

    eh = random.randint(int(H * 0.15), int(H * 0.35))
    if torso_only:
        ey = random.randint(0, max(0, H // 3 - eh))
    else:
        ey = random.randint(0, max(0, H - eh))

    ew = random.randint(int(W * 0.40), int(W * 0.80))
    ex = random.randint(0, max(0, W - ew))

    out[:, :, ey:ey + eh, ex:ex + ew] = 0.0
    return out


class CASIAB_SSL(Dataset):
    """
    Dataset Fase 1 SSL. Igual que el original pero con Random Erasing barato.
    Sin caché — el SSL original ya era rápido, no necesita caché.
    Las aumentaciones son solo zeroing de píxeles: costo despreciable.
    """
    def __init__(self, root_path, img_size=(64, 64), use_subset=True, subset_subjects=10,
                 subset_conditions=None, subset_angles=None, seq_len=15,
                 use_augmentation=True,
                 erase_prob=0.5, torso_only=True,
                 # Estos parámetros se aceptan pero ya no se usan (compatibilidad)
                 scale_prob=0.5, scale_range=(0.85, 1.15),
                 use_cache=False):

        self.seq_len          = seq_len
        self.img_size         = img_size
        self.use_augmentation = use_augmentation
        self.erase_prob       = erase_prob
        self.torso_only       = torso_only
        self.sequences        = []

        root = Path(root_path)
        all_subjects = sorted([d for d in root.iterdir() if d.is_dir()])
        subjects = all_subjects[:subset_subjects] if use_subset else all_subjects

        print(f"\n[Dataset SSL Volumétrico] Cargando desde: {root_path}")
        print(f"  Aumentaciones π(·) [Random Erasing]: {'ACTIVADAS' if use_augmentation else 'desactivadas'}")

        for subject_dir in subjects:
            for condition_dir in subject_dir.iterdir():
                if not condition_dir.is_dir(): continue
                if subset_conditions and condition_dir.name not in subset_conditions: continue
                for angle_dir in condition_dir.iterdir():
                    if not angle_dir.is_dir(): continue
                    if subset_angles and angle_dir.name not in subset_angles: continue
                    frames = sorted([
                        f for f in angle_dir.iterdir()
                        if f.suffix.lower() in ['.png', '.jpg', '.bmp']
                    ])
                    if len(frames) >= seq_len:
                        self.sequences.append(frames)

        print(f"  Secuencias volumétricas (>= {seq_len} frames): {len(self.sequences)}")

        self.base_transform = transforms.Compose([
            transforms.Resize(img_size, antialias=True),
            transforms.ToTensor(),
        ])

    def _sample_frames(self, frames, start_idx):
        """Carga seq_len frames desde start_idx aplicando CoM por frame."""
        sampled = []
        for i in range(self.seq_len):
            idx     = min(start_idx + i, len(frames) - 1)
            img     = Image.open(frames[idx]).convert("L")
            img_t   = self.base_transform(img)
            img_com = center_of_mass_align(img_t)
            sampled.append(img_com)
        return torch.stack(sampled, dim=0)  # [T, C, H, W]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        frames    = self.sequences[idx]
        total_len = len(frames)

        if total_len > self.seq_len:
            start1 = random.randint(0, total_len - self.seq_len)
            start2 = random.randint(0, total_len - self.seq_len)
        else:
            start1 = start2 = 0

        view1 = self._sample_frames(frames, start1)
        view2 = self._sample_frames(frames, start2)

        # Aumentaciones independientes por vista
        if self.use_augmentation:
            # 1. Random Erasing (simula BG/CL en CASIA-B)
            view1 = gait_erase_volume(view1, self.erase_prob, self.torso_only)
            view2 = gait_erase_volume(view2, self.erase_prob, self.torso_only)

            # 2. Gaussian noise (simula siluetas ruidosas de Gait3D outdoor)
            view1 = gait_add_noise(view1, noise_prob=0.3, noise_std=0.05)
            view2 = gait_add_noise(view2, noise_prob=0.3, noise_std=0.05)

            # 3. Erosión morfológica (simula variaciones de segmentación cross-dataset)
            view1 = gait_erode_volume(view1, erode_prob=0.2)
            view2 = gait_erode_volume(view2, erode_prob=0.2)

        return view1, view2


class CASIAB_Supervised(Dataset):
    """
    Dataset PKSampler-Friendly para Fases 2 y 3. Sin cambios respecto al original.
    """
    def __init__(self, root_path, subject_range, conditions, angles=None,
                 seq_len=15, img_size=(64, 64), augment=False, return_info=False):

        self.root        = Path(root_path)
        self.seq_len     = seq_len
        self.img_size    = img_size
        self.augment     = augment
        self.return_info = return_info
        self.samples     = []
        self.subject_to_label = {}

        all_subjects = sorted([d.name for d in self.root.iterdir() if d.is_dir()])
        start_idx, end_idx = subject_range
        subjects = all_subjects[start_idx:end_idx]

        print(f"\n[Dataset Supervised] Rango: {start_idx+1:03d}-{end_idx:03d} ({len(subjects)} sujetos)")
        if angles:
            print(f"  Ángulos: {angles}")

        for label_id, subject_name in enumerate(subjects):
            self.subject_to_label[subject_name] = label_id

        for subject_name in subjects:
            subject_dir = self.root / subject_name
            label_id    = self.subject_to_label[subject_name]

            for condition_dir in subject_dir.iterdir():
                if not condition_dir.is_dir() or condition_dir.name not in conditions:
                    continue
                for angle_dir in condition_dir.iterdir():
                    if not angle_dir.is_dir(): continue
                    if angles and angle_dir.name not in angles: continue
                    frames = sorted([
                        f for f in angle_dir.iterdir()
                        if f.suffix.lower() in ['.png', '.jpg', '.bmp']
                    ])
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample    = self.samples[idx]
        frames    = sample['frames']
        total_len = len(frames)

        do_scale     = False
        scale_factor = 1.0
        do_erase     = False
        erase_params = {}

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
                ey = random.randint(0, H - eh)
                ex = random.randint(0, W - ew)
                erase_params = {'y': ey, 'x': ex, 'h': eh, 'w': ew}
        else:
            start_idx = (total_len - self.seq_len) // 2

        sampled_tensors = []
        for i in range(self.seq_len):
            idx_f = min(start_idx + i, total_len - 1)
            img   = Image.open(frames[idx_f]).convert("L")

            if do_scale:
                w_orig, h_orig = img.size
                new_w          = int(w_orig * scale_factor)
                img_scaled     = img.resize((new_w, h_orig), Image.BILINEAR)
                new_img        = Image.new("L", (w_orig, h_orig), 0)
                if scale_factor < 1.0:
                    offset = (w_orig - new_w) // 2
                    new_img.paste(img_scaled, (offset, 0))
                else:
                    offset = (new_w - w_orig) // 2
                    new_img.paste(img_scaled.crop((offset, 0, offset + w_orig, h_orig)))
                img = new_img

            img_t       = self.base_transform(img)
            img_aligned = center_of_mass_align(img_t)

            if do_erase:
                y, x = erase_params['y'], erase_params['x']
                h_b, w_b = erase_params['h'], erase_params['w']
                img_aligned[:, y:y + h_b, x:x + w_b] = 0.0

            sampled_tensors.append(img_aligned)

        seq_tensor = torch.stack(sampled_tensors, dim=0)

        if self.return_info:
            return seq_tensor, sample['label'], sample['condition'], sample['angle'], sample['subject']
        return seq_tensor, sample['label']

    def get_num_classes(self):
        return len(self.subject_to_label)