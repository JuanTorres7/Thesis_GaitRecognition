# configs/settings.py
import torch
import os

# =============================================================================
# SELECCIÓN DE DATASET ACTIVO
# =============================================================================
# Cambia este valor para cambiar de dataset sin tocar nada más.
# Opciones: 'casiab' | 'gait3d'
ACTIVE_DATASET = 'casiab'

# --- Configuración General ---
DEVICE      = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE  = 32
NUM_INSTANCES = 4

# Workers del DataLoader. En el cluster (Linux), verificar con `nproc` cuántos
# cores tiene el nodo asignado y usar num_workers ~= nproc - 1 (deja 1 core libre
# para el proceso principal). En Windows/laptop mantener en 2.
DATALOADER_NUM_WORKERS = 8

# =============================================================================
# CONFIGURACIÓN CASIA-B
# =============================================================================
CASIAB_ROOT_PATH   = "C:\\Users\\JuanTF\\Desktop\\Gait_Recognition\\archive\\output"
CASIAB_IMG_SIZE    = (64, 64)
CASIAB_SEQ_LEN     = 24
CASIAB_SSL_SUBJECTS = 74
CASIAB_SSL_CONDITIONS = ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06',
                          'bg-01','bg-02','cl-01','cl-02']
CASIAB_SSL_ANGLES  = ['000','018','036','054','072','090','108',
                      '126','144','162','180']
CASIAB_CONFIG = {
    'conditions': ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06',
                   'bg-01','bg-02','cl-01','cl-02'],
    'angles':     ['000','018','036','054','072','090','108',
                   '126','144','162','180'],
    'train_range': (0, 74),
    'val_range':   (64, 74),
    'test_range':  (74, 124),
}

# =============================================================================
# CONFIGURACIÓN GAIT3D
# =============================================================================
GAIT3D_ROOT_PATH  = "C:\\Users\\JuanTF\\Desktop\\Gait_Recognition\\Gait3D\\2D_Silhouettes"
GAIT3D_JSON_PATH  = "C:\\Users\\JuanTF\\Desktop\\Gait_Recognition\\Gait3D\\Gait3D.json"
GAIT3D_IMG_SIZE   = (64, 44)   # Resolución nativa OU-MVLP compatible; Gait3D se redimensiona a esto
GAIT3D_SEQ_LEN    = 24
# Gait3D no tiene condiciones ni ángulos fijos — es outdoor con ángulo libre.
# El JSON define exactamente qué secuencias son train/test/probe.
GAIT3D_CONFIG = {
    'json_path':  GAIT3D_JSON_PATH,
    'root_path':  GAIT3D_ROOT_PATH,
    'img_size':   GAIT3D_IMG_SIZE,
    'seq_len':    GAIT3D_SEQ_LEN,
}

# =============================================================================
# PARÁMETROS ACTIVOS — se resuelven según ACTIVE_DATASET
# =============================================================================
def get_active_config():
    """Devuelve el bloque de configuración del dataset activo."""
    if ACTIVE_DATASET == 'casiab':
        return {
            'dataset':    'casiab',
            'root_path':  CASIAB_ROOT_PATH,
            'img_size':   CASIAB_IMG_SIZE,
            'seq_len':    CASIAB_SEQ_LEN,
            'config':     CASIAB_CONFIG,
        }
    elif ACTIVE_DATASET == 'gait3d':
        return {
            'dataset':    'gait3d',
            'root_path':  GAIT3D_ROOT_PATH,
            'img_size':   GAIT3D_IMG_SIZE,
            'seq_len':    GAIT3D_SEQ_LEN,
            'config':     GAIT3D_CONFIG,
        }
    else:
        raise ValueError(f"Dataset no reconocido: {ACTIVE_DATASET}. Opciones: 'casiab', 'gait3d'")

# Alias directos para compatibilidad con scripts existentes
_active = get_active_config()
ROOT_PATH = _active['root_path']
IMG_SIZE  = _active['img_size']
SUPERVISED_SUBSET_FRAMES_PER_SEQ = _active['seq_len']
SUPERVISED_CONFIG = _active['config'] if ACTIVE_DATASET == 'casiab' else None

# =============================================================================
# CONFIGURACIÓN SSL
# =============================================================================
SSL_EPOCHS      = 40
SSL_LEARNING_RATE = 3e-4
SSL_TEMPERATURE = 0.07
SSL_CHECKPOINT  = f"models/backbone_ssl_best_{ACTIVE_DATASET}.pth"

SSL_USE_SUBSET  = True  # Solo para CASIA-B; Gait3D usa TRAIN_SET del JSON
SSL_USE_GAIT_AUGMENTATION = True
SSL_ERASE_PROB       = 0.5
SSL_ERASE_TORSO_ONLY = True
SSL_SCALE_PROB       = 0.5
SSL_SCALE_RANGE      = (0.85, 1.15)
SSL_USE_CACHE        = False

# Parámetros SSL específicos de CASIA-B (ignorados si dataset=gait3d)
SSL_SUBSET_SUBJECTS   = CASIAB_SSL_SUBJECTS
SSL_SUBSET_CONDITIONS = CASIAB_SSL_CONDITIONS
SSL_SUBSET_ANGLES     = CASIAB_SSL_ANGLES
SSL_SUBSET_FRAMES_PER_SEQ = CASIAB_SEQ_LEN

# =============================================================================
# CONFIGURACIÓN SUPERVISADA
# =============================================================================
SUPERVISED_EPOCHS       = 40
SUPERVISED_LEARNING_RATE = 1e-4
SUPERVISED_MARGIN       = 0.3
SUPERVISED_CHECKPOINT   = f"models/supervised_model_{ACTIVE_DATASET}.pth"

# =============================================================================
# CONFIGURACIÓN HÍBRIDA
# =============================================================================
HYBRID_EPOCHS          = 60
HYBRID_LEARNING_RATE   = 1e-4
HYBRID_TRIPLET_WEIGHT  = 0.5
HYBRID_MARGIN          = 0.3
HYBRID_CHECKPOINT      = f"models/hybrid_model_final_{ACTIVE_DATASET}.pth"
HYBRID_CHECKPOINT_LAST = f"models/hybrid_model_last_{ACTIVE_DATASET}.pth"
SAVE_LAST_EPOCH        = True

# =============================================================================
# PÉRDIDA MÉTRICA Y RE-RANKING
# =============================================================================
USE_CIRCLE_LOSS = False
USE_RERANKING   = True
RERANK_K1       = 40
RERANK_K2       = 2
RERANK_LAMBDA   = 0.05

USE_PAL                  = True
PAL_ALPHA                = 32
PAL_DELTA                = 0.1
PAL_WEIGHT               = 0.5
PAL_INIT_FROM_SUPERVISED = True

# =============================================================================
# VERIFICACIÓN
# =============================================================================
def check_paths():
    if ACTIVE_DATASET == 'casiab':
        if not os.path.exists(CASIAB_ROOT_PATH):
            raise FileNotFoundError(f"CASIA-B no encontrado: {CASIAB_ROOT_PATH}")
    elif ACTIVE_DATASET == 'gait3d':
        if not os.path.exists(GAIT3D_ROOT_PATH):
            raise FileNotFoundError(f"Gait3D silhouettes no encontrado: {GAIT3D_ROOT_PATH}")
        if not os.path.exists(GAIT3D_JSON_PATH):
            raise FileNotFoundError(f"Gait3D JSON no encontrado: {GAIT3D_JSON_PATH}")
    os.makedirs("models", exist_ok=True)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset activo: {ACTIVE_DATASET.upper()}")
    print(f"Dataset Root:   {ROOT_PATH}")
