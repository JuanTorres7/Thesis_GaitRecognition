# configs/settings.py
import torch
import os

# --- Configuración General ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = (64, 64)
BATCH_SIZE = 32
NUM_INSTANCES = 4

# --- Ruta al Dataset ---
ROOT_PATH = "C:\\Users\\JuanTF\\Desktop\\Gait_Recognition\\archive\\output"

# --- Configuración SSL ---
SSL_EPOCHS = 40
SSL_LEARNING_RATE = 3e-4
SSL_TEMPERATURE = 0.07
SSL_CHECKPOINT = "models/backbone_ssl_best.pth"

SSL_USE_SUBSET = True
SSL_SUBSET_SUBJECTS = 74
SSL_SUBSET_CONDITIONS = ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02']
SSL_SUBSET_ANGLES = ['000','018','036','054','072','090','108','126','144','162','180']
SSL_SUBSET_FRAMES_PER_SEQ = 24

# Aumentaciones π(·) en SSL — Random Erasing de torso (simula BG/CL)
SSL_USE_GAIT_AUGMENTATION = True
SSL_ERASE_PROB        = 0.5
SSL_ERASE_TORSO_ONLY  = True
SSL_SCALE_PROB        = 0.5
SSL_SCALE_RANGE       = (0.85, 1.15)

# IMPORTANTE: False — el caché causaba colgado en Windows con num_workers=2
# El SSL sin caché con num_workers=2 ya es suficientemente rápido
SSL_USE_CACHE = False

# --- Configuración Supervisada ---
SUPERVISED_EPOCHS = 40
SUPERVISED_LEARNING_RATE = 1e-4
SUPERVISED_MARGIN = 0.3
SUPERVISED_CHECKPOINT = "models/supervised_model.pth"

SUPERVISED_USE_SUBSET = True
SUPERVISED_SUBSET_TRAIN_SUBJECTS = 74
SUPERVISED_SUBSET_VAL_SUBJECTS = 10
SUPERVISED_SUBSET_TEST_SUBJECTS = 50
SUPERVISED_SUBSET_FRAMES_PER_SEQ = 24

SUPERVISED_CONFIG = {
    'conditions': ['nm-01','nm-02','nm-03','nm-04','nm-05','nm-06','bg-01','bg-02','cl-01','cl-02'],
    'angles': ['000','018','036','054','072','090','108','126','144','162','180'],
    'train_range': (0, 74),
    'val_range': (64, 74),
    'test_range': (74, 124),
}

# --- Verificación de Directorios ---
def check_paths():
    if not os.path.exists(ROOT_PATH):
        raise FileNotFoundError(f"La ruta del dataset no existe: {ROOT_PATH}")
    os.makedirs("models", exist_ok=True)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Dataset Root: {ROOT_PATH}")

# --- Configuración Híbrida ---
HYBRID_EPOCHS = 60
HYBRID_LEARNING_RATE = 1e-4
HYBRID_TRIPLET_WEIGHT = 0.5
HYBRID_MARGIN = 0.3
HYBRID_CHECKPOINT = "models/hybrid_model_final_PRO.pth"

# --- Optimización de Inferencia ---
USE_CIRCLE_LOSS  = False
USE_RERANKING    = True
RERANK_K1        = 40
RERANK_K2        = 2
RERANK_LAMBDA    = 0.05

# --- Proxy Anchor Loss (PAL) ---
USE_PAL                  = True
PAL_ALPHA                = 32
PAL_DELTA                = 0.1
PAL_WEIGHT               = 0.5
PAL_INIT_FROM_SUPERVISED = True

# --- Checkpoint de respaldo para comparar ---
# Guarda adicionalmente el modelo de la última época para comparar vs mejor Mini-Val
SAVE_LAST_EPOCH = True
HYBRID_CHECKPOINT_LAST = "models/hybrid_model_last_epoch.pth"