# Un Enfoque Híbrido: Gait-Based Person Re-ID

Este proyecto implementa un pipeline de Deep Learning para la Re-Identificación de personas (Re-ID) basado en la marcha (gait), utilizando un enfoque híbrido que combina aprendizaje auto-supervisado (SSL) y supervisado. El modelo final se entrena con una pérdida dual para optimizar tanto la clasificación como la similitud de embeddings.

Este código es la implementación del trabajo de investigación "UN ENFOQUE HÍBRIDO: GAIT-BASED PERSON RE-ID" (Torres & Canto, 2025).

---

## Metodología

El entrenamiento se divide en tres fases distintas para construir un modelo robusto y modular:

1. **Fase 1: Auto-Supervisada (SSL)**

   * **Objetivo:** Aprender representaciones visuales robustas de la marcha sin necesidad de etiquetas.
   * **Método:** Se entrena un `GaitBackbone` usando SimCLR (NT-Xent Loss) sobre una gran cantidad de siluetas de marcha no etiquetadas.
   * **Script:** `train_ssl.py`
2. **Fase 2: Supervisada (Clasificación)**

   * **Objetivo:** Enseñar al modelo a *identificar* personas específicas.
   * **Método:** Se añade una capa de clasificación al backbone pre-entrenado (de la Fase 1) y se entrena con `CrossEntropyLoss` en un conjunto de datos etiquetado.
   * **Script:** `train_supervised.py`
3. **Fase 3: Híbrida (Fine-Tuning)**

   * **Objetivo:** Optimizar el modelo final tanto para clasificación como para extracción de características (embeddings).
   * **Método:** Se realiza un fine-tuning del modelo supervisado (de la Fase 2) usando una **pérdida híbrida combinada**: `CrossEntropyLoss` + `TripletLoss`.
   * **Script:** `train_hybrid.py`

---

## Estructura del Proyecto

El proyecto está organizado de la siguiente manera para mantener la modularidad y escalabilidad:

```bash
gait_reid_project/
├── configs/
│   └── settings.py
├── models/
│   ├── backbone_ssl_best.pth
│   ├── supervised_model.pth
│   └── hybrid_model_best.pth
├── notebooks/
│   └── MainTest.ipynb
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── models.py
│   └── losses.py
├── .gitignore
├── requirements.txt
├── train_ssl.py
├── train_supervised.py
├── train_hybrid.py
└── README.md
```

---

## Cómo Empezar

Sigue estos pasos para configurar y ejecutar el proyecto localmente.

### 1. Prerrequisitos

* Python 3.8+
* [PyTorch](https://pytorch.org/)
* Acceso al dataset (ej. CASIA-B)

### 2. Instalación

1. **Clona el repositorio:**

   ```bash
   git clone [https://github.com/demon21382024/TESIS_CODE_PROYECT_CS.git](https://github.com/demon21382024/TESIS_CODE_PROYECT_CS.git)
   cd TESIS_CODE_PROYECT_CS
   ```
2. **Configura la ruta del Dataset:**
   Abre `configs/settings.py` y modifica la variable `ROOT_PATH` para que apunte a la carpeta de tu dataset (ej. `.../archive/output`).
3. **Crea y activa un entorno virtual:**

   ```bash
   # Crear el entorno (puedes llamarlo 'venv' o como prefieras)
   python -m venv venv

   # Activar en Windows (PowerShell)
   .\venv\Scripts\Activate.ps1

   # Activar en Windows (CMD)
   .\venv\Scripts\activate.bat

   # Activar en macOS/Linux
   source venv/bin/activate
   ```
4. **Instala las dependencias:**

   ```bash
   pip install -r requirements.txt
   ```

### 3. Ejecutar el Pipeline de Entrenamiento

Ejecuta los scripts en orden. Cada script cargará el modelo guardado por el script anterior.

```bash
# Paso 1: Entrenar el backbone auto-supervisado
python train_ssl.py

# Paso 2: Entrenar el clasificador supervisado (usa el backbone de arriba)
python train_supervised.py

# Paso 3: Aplicar el fine-tuning híbrido (usa el modelo supervisado)
python train_hybrid.py
---
```

## Autores :

* **Harold Alexis Victor Canto Vidal**
* **Juan Manuel Torres Farfán**
