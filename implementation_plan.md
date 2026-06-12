# Plan de Ingeniería Detallado: Transición Modal a Múltiples Secuencias Temporales $[T, C, H, W]$

Como Doctor en Ciencias de la Computación, reestructuro formalmente este análisis en un plan secuencial que garantiza el salto evolutivo de tu modelo base (2D Estático) a un sistema biométrico moderno enfocado en la cinemática temporal.

## 1. Fase de Diseño y Justificación Arquitectónica

### A. Dataloader Volumétrico ($[T, C, H, W]$)
**Diseño y Justificación:** El ciclo armónico de la marcha es intrínsecamente temporal; un corte en un *frame* no captura biometría. Refactorizaremos `CASIAB_Supervised` (y su equivalente SSL) para leer carpetas continuas devolviendo tensores volumétricos donde $T$ (ej. $T=15$) codifica la progresión del paso. Además, inyectaremos un paso matemático de **Alineamiento por Centro de Masa (CoM)** antes de apilar los *frames*. Al trasladar el baricentro de los píxeles blancos al centro geométrico del tensor $(H/2, W/2)$, se elimina el ruido de traslación transversal (el *jitter*), permitiéndole a la red invertir todos sus filtros directamente en morfología intra-clase y trayectoria en lugar de tener que aprender invarianza a la traslación.

### B. Arquitectura Temporal (Extracción sobre el eje $T$)
**Diseño y Justificación:** La base será la ResNet-18 para decodificación espacial individual. Empaquetaremos las dimensiones `[B*T, C, H, W]` para el pase frontal. El tensor resultante `[B, T, Feature\_Dim]` contendrá representaciones espaciales profundas organizadas de forma secuencial temporal. Implementaremos **Pooling Temporal Dual**:
*   **Global Average Pooling (GAP) sobre $T$:** Promediará las siluetas temporalmente. Su efecto es comparable a un GEI (Gait Energy Image) pero operando al final de la CNN, conservando propiedades morfológicas estáticas altamente redundantes y corporales.
*   **Global Max Pooling (GMP) sobre $T$:** Actuará como un sensor de amplitudes pico. Identificará los momentos de elongación máxima del brazo/pierna a lo largo de los $T$ iteradores.
La concatenación matemática de ambas ramas $f = [GAP(z) \parallel GMP(z)]$ será la forma definitiva de la firma biométrica generada.

### C. Estrategia de mAP (PKSampler + Vectorized Batch Hard Triplet Loss)
**Diseño y Justificación:** Un mAP bajo indica un colapso en el *ranking* (alta varianza intra-clase mezclada). Desarrollaremos una *Batch Hard Triplet Loss* escrita con operaciones tensoriales vectorizadas eficientes (`torch.cdist`). Cada paso de la pérdida seleccionará explícitamente el "Falso Positivo más Duro" (la misma identidad más lejana morfológicamente dentro del tensor de bachos) y el "Falso Negativo más Duro" (identidad distinta genéticamente similar a nuestra ancla). Al forzar hiperplanos dinámicamente sobre los límites del ruido, el mAP empujará sus márgenes hasta sus máximos globales.

### D. SSL Contrastivo (InfoNCE Temporal)
**Diseño y Justificación:** La tarea de pre-texto no debe buscar rotaciones estáticas artificiales. La contrastación InfoNCE forzará una maximización de similitud y reducción del ruido temporal. Partiremos una secuencia natural de 30 frames en $V_{1} (frames \ 1\text{-}15)$ y $V_{2} (frames \ 10\text{-}25)$. Forzar al clasificador latente a ver estas partes como provenientes del mismo origen entrena sub-redes con gran "invarianza de fase" temporal: "no importa en qué inicio del ciclo de marcha veas al sujeto, él es el mismo".

---

## 2. Protocolo de "Señales Tempranas" (Validación Rápida de Científica)

Para asegurar la convergencia de nuestra teoría a código antes de quemar días de servidor, integramos este escudo térmico metodológico:

### A. Monitor de Tripletas (Alojamiento en Consola)
Implementaremos el *Active Hard Multipliers Log*. Durante iteraciones en el batch del entrenamiento, un print mostrará la "Fracción de Triplets Válidas" ($\%$ de combinaciones cuya distancia cumple la anomalía $ancla \rightarrow pos > ancla \rightarrow neg$).
*   **Estado Saludable:** Inicia en alto ~$90\%$ y decae como curva asintótica a medida que el espacio métrico aprende sus distancias.
*   **Métrica de Crisis:** Cae al $0\%$ o se estanca sin bajar, indicando un mal LR o colapso de representación latente, previniendo ceguera de entrenamiento de días enteros.

### B. Mini-Val Set Intratable (Validación Continua)
Programaremos un script asíncrono `evaluate_mini()` dentro del loop general configurado cada `2` épocas. Recolectará un subconjunto *Query-Gallery* minúsculo (10 sujetos puros en NM). Validará matemáticamente si Rank-1 abandona rápidamente la aleatoriedad (ej. trepar del 1% probabilístico al 45% en las primeras diez épocas). Es la brújula innegociable antes del *Train Complete Run*.

### C. Ablación Inicial de Diagnóstico (Prueba de 5 Épocas)
Consistirá en tomar una instantánea corta (5 epochs exactos). Compararemos:
1.  Tu Código Viejo Aislado (1 frame) bajo una iteración de validación.
2.  Nuestro Enfoque Volumétrico Total ($T=15$).
El despegue acelerado del mAP y validación empírica en esta ablación sentarán una prueba innegable científica para aprobar recursos computacionales a tu corrida principal. 

---

> [!CAUTION]
> ## 3. Fase de Ejecución Modular (Requiere Aprobación)
> 
> Todo el plan requiere ser implementado paso a paso para no corromper la trazabilidad, además nos asegura una portabilidad escalable rápida cuando transiciones por ejemplo a datos inmensos como OU-MVLP (más de 10,000 sujetos):
> 
> 1.  Modificar **`src/data.py`**: Integrar CoM (*Center of Mass*) estandarizado matemáticamente y redefinir indexación 3D para la silueta volumétrica.
> 2.  Modificar **`src/models.py`**: Replegar la topología del ResNet-18 sobre tensores combinados $B*T$ adaptando módulos `GAP/GMP` al tensor temporal re-expandido $T$, reduciéndolos espacialmente a secuencias cinemáticas lineales.
> 3.  Modificar **`src/losses.py` y `src/samplers.py`**: Triplet loss y PK-Sampler paralelos y rápidos en tensores nativos de CUDA.
> 4.  Modificar **`train_hybrid.py`**: Introducción a *Mini-Val set*, Monitor Triplete, y Ajuste Fino Diferencial (tasas de aprendizaje 10 veces menores para el Backbone respecto al Clasificador lineal).
> 
> **¿Cuento con tu aprobación para comenzar la ingeniería explícita de código sobre el PASO 1 (`src/data.py`) y escalar en el orden descrito?**
