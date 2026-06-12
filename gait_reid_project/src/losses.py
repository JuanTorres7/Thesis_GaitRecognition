# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """NT-Xent Loss (SimCLR) para InfoNCE Temporal. Sin cambios."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss(reduction="mean")

    def forward(self, z1, z2):
        batch_size = z1.size(0)

        # Cast explícito a FP32 antes de cualquier operación numérica.
        # Con AMP activado, autocast convierte tensores a FP16 (máx ~65504).
        # -1e9 desborda FP16 → RuntimeError. Solución: forzar FP32 aquí y
        # usar -1e4 como fill (suficientemente negativo para softmax, seguro en FP16).
        z1 = z1.float()
        z2 = z2.float()

        z          = torch.cat([z1, z2], dim=0)          # FP32 garantizado
        sim_matrix = torch.mm(z, z.t()) / self.temperature

        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix.masked_fill_(mask, -1e4)  # -1e4 seguro en FP16/FP32

        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),
            torch.arange(0, batch_size)
        ]).to(z.device)
        return self.criterion(sim_matrix, labels)


class TripletLoss(nn.Module):
    """
    Vectorized Batch Hard Triplet Loss (OPTIMIZADA en CUDA). Sin cambios.
    Devuelve (loss, fraction_active) para el monitor de señales tempranas.
    """
    def __init__(self, margin=0.3):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        device = embeddings.device

        dist_matrix = torch.cdist(embeddings, embeddings, p=2)

        labels_mat = labels.expand(B, B)
        is_pos = labels_mat.eq(labels_mat.t())
        is_neg = labels_mat.ne(labels_mat.t())

        identity_mask = torch.eye(B, dtype=torch.bool, device=device)
        is_pos = is_pos & ~identity_mask

        dist_ap = dist_matrix * is_pos.float()
        hardest_positive_dist, _ = dist_ap.max(dim=1)

        max_dist_val = dist_matrix.max() + 1e5
        dist_an = dist_matrix + (~is_neg).float() * max_dist_val
        hardest_negative_dist, _ = dist_an.min(dim=1)

        loss_components = F.relu(hardest_positive_dist - hardest_negative_dist + self.margin)

        active_triplets = loss_components[loss_components > 0]
        num_active = active_triplets.size(0)
        fraction_active = float(num_active) / float(B)

        if num_active > 0:
            loss = active_triplets.mean()
        else:
            loss = loss_components.sum() * 0.0

        return loss, fraction_active


class CircleLoss(nn.Module):
    """Circle Loss para aprendizaje métrico. Sin cambios."""
    def __init__(self, m=0.25, gamma=80):
        super().__init__()
        self.m = m
        self.gamma = gamma

    def forward(self, embeddings, labels):
        B = embeddings.size(0)
        device = embeddings.device

        sim_matrix = torch.matmul(embeddings, embeddings.t())

        labels_mat = labels.expand(B, B)
        is_pos = labels_mat.eq(labels_mat.t())
        is_neg = labels_mat.ne(labels_mat.t())

        identity_mask = torch.eye(B, dtype=torch.bool, device=device)
        is_pos = is_pos & ~identity_mask

        op = 1 + self.m
        on = -self.m
        dp = 1 - self.m
        dn = self.m

        ap = torch.clamp(op - sim_matrix, min=0.0)
        an = torch.clamp(sim_matrix - on, min=0.0)

        logit_p = -self.gamma * ap * (sim_matrix - dp)
        logit_n =  self.gamma * an * (sim_matrix - dn)

        logit_p = logit_p.masked_fill(~is_pos, -1e4)
        logit_n = logit_n.masked_fill(~is_neg, -1e4)

        logsum_p = torch.logsumexp(logit_p, dim=1)
        logsum_n = torch.logsumexp(logit_n, dim=1)

        loss_components = F.softplus(logsum_p + logsum_n)

        has_pos = is_pos.sum(dim=1) > 0
        has_neg = is_neg.sum(dim=1) > 0
        valid_rows = has_pos & has_neg

        if valid_rows.sum() > 0:
            loss = loss_components[valid_rows].mean()
        else:
            loss = loss_components.sum() * 0.0

        return loss, 0.0


# =============================================================================
# [NUEVO] Proxy Anchor Loss
# =============================================================================
# Referencia: Kim et al., "Proxy Anchor Loss for Deep Metric Learning", CVPR 2020.
#
# Motivación en el contexto de tu pipeline:
#   TripletLoss corrige solo el par (ancla, pos_más_lejano, neg_más_cercano)
#   por cada ancla en el batch → un gradiente por elemento.
#
#   PAL aprende un vector "proxy" por clase (inicializado desde centroides reales).
#   En cada forward, TODOS los embeddings del batch interactúan con TODOS los proxies:
#   - Embeddings de la clase C se acercan al proxy C  (término positivo)
#   - Embeddings de la clase C se alejan de proxies ≠ C (término negativo)
#   → Gradiente denso: O(B × num_classes) interacciones por step.
#
#   Para CL/BG esto es crítico: cuando un abrigo dispersa los embeddings de una clase,
#   PAL jala TODOS ellos de vuelta al proxy en el mismo paso, mientras Triplet
#   solo corría al más extremo.
#
# Compatibilidad con tu arquitectura:
#   - Recibe los mismos (embeddings, labels) que TripletLoss y CircleLoss.
#   - Devuelve (loss, 0.0) para mantener la firma del monitor de señales tempranas.
#   - Se inicializa con num_classes y embed_dim desde train_hybrid.py.
#   - Los proxies son nn.Parameter → se optimizan junto con el modelo.
# =============================================================================

class ProxyAnchorLoss(nn.Module):
    """
    Proxy Anchor Loss para aprendizaje métrico denso.

    Args:
        num_classes (int): Número de clases de entrenamiento (74 para CASIA-B train).
        embed_dim   (int): Dimensión del embedding L2-normalizado (1024 en tu modelo).
        alpha       (int): Escala del logit. Paper canónico: 32. Rango útil: 16–64.
                           Mayor alpha → separación más agresiva pero más riesgo de
                           colapso si los proxies no están bien inicializados.
        delta       (float): Margen de separación. Paper canónico: 0.1.
                             Interpretación: los pos deben superar proxy_sim + delta
                             y los neg deben estar por debajo proxy_sim - delta.
    """
    def __init__(self, num_classes: int, embed_dim: int, alpha: int = 32, delta: float = 0.1):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim   = embed_dim
        self.alpha       = alpha
        self.delta       = delta

        # Proxies: un vector por clase, forma [num_classes, embed_dim]
        # Se normalizan L2 en cada forward, igual que los embeddings.
        # Inicialización Kaiming: buena distribución inicial antes de
        # sobreescribir con centroides reales desde el checkpoint supervisado.
        self.proxies = nn.Parameter(torch.randn(num_classes, embed_dim))
        nn.init.kaiming_normal_(self.proxies, mode='fan_out')

    def init_proxies_from_centroids(self, centroids: torch.Tensor):
        """
        Sobreescribe los proxies con centroides pre-calculados desde Fase 2.

        Args:
            centroids: Tensor [num_classes, embed_dim] con el centroide L2-normalizado
                       de cada clase, calculado pasando el dataset de train por el
                       modelo supervisado (ver train_hybrid.py: _compute_centroids).
        """
        assert centroids.shape == (self.num_classes, self.embed_dim), (
            f"Shape de centroids {centroids.shape} no coincide con "
            f"({self.num_classes}, {self.embed_dim})"
        )
        with torch.no_grad():
            self.proxies.copy_(centroids)
        print(f"  [PAL] Proxies inicializados desde centroides reales ({self.num_classes} clases).")

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor):
        """
        Args:
            embeddings: [B, embed_dim] — embeddings L2-normalizados (salida de HybridGaitModel).
            labels:     [B]            — etiquetas enteras 0..num_classes-1.

        Returns:
            loss (Tensor escalar), 0.0 (fracción placeholder para compatibilidad con monitor).
        """
        # Normalizar proxies en cada forward (mantener sobre la hiperesfera unitaria)
        P = F.normalize(self.proxies, p=2, dim=1)   # [num_classes, embed_dim]
        E = embeddings                               # ya normalizados por HybridGaitModel

        # Similitud coseno entre todos los embeddings y todos los proxies
        # sim[i, c] = similitud del embedding i con el proxy de clase c
        sim = torch.matmul(E, P.t())  # [B, num_classes]

        # Máscara: pos_mask[i, c] = True si labels[i] == c
        pos_mask = torch.zeros(embeddings.size(0), self.num_classes,
                               dtype=torch.bool, device=embeddings.device)
        pos_mask.scatter_(1, labels.unsqueeze(1), True)   # [B, num_classes]
        neg_mask = ~pos_mask

        # --- Término positivo ---
        # Para cada proxy c que tiene al menos un positivo en el batch:
        #   log(1 + sum_{x: label=c} exp(-alpha * (sim(x,c) - delta)))
        # Esto penaliza embeddings positivos que están lejos del proxy c.
        #
        # Usamos logsumexp sobre la dimensión B (embeddings), agrupando por proxy.
        # Proxies sin ningún positivo en el batch se excluyen del promedio.

        # [B, num_classes]: contribución de cada embedding al término pos de cada proxy
        pos_logits = -self.alpha * (sim - self.delta)         # [B, num_classes]
        pos_logits = pos_logits * pos_mask.float()            # zeroed para negativos
        # Para proxies sin positivos en el batch, queremos excluirlos del logsumexp.
        # Llenamos con -inf los no-positivos para que no sumen en logsumexp.
        pos_logits_masked = pos_logits.masked_fill(neg_mask, -1e4)  # [B, num_classes]

        # Suma sobre embeddings (dim=0) → un valor por proxy
        # logsumexp(dim=0) = log(sum_i exp(pos_logits[i, c]))
        loss_pos_per_proxy = torch.log(1 + torch.exp(pos_logits_masked).sum(dim=0))  # [num_classes]

        # Solo promediar proxies que tienen al menos un positivo en el batch
        has_pos = pos_mask.sum(dim=0) > 0   # [num_classes] booleano
        if has_pos.sum() == 0:
            loss_pos = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        else:
            loss_pos = loss_pos_per_proxy[has_pos].mean()

        # --- Término negativo ---
        # Para cada proxy c que tiene al menos un negativo en el batch:
        #   log(1 + sum_{x: label≠c} exp(alpha * (sim(x,c) + delta)))
        # Esto penaliza embeddings negativos que están cerca del proxy c.

        neg_logits = self.alpha * (sim + self.delta)          # [B, num_classes]
        neg_logits_masked = neg_logits.masked_fill(pos_mask, -1e4)  # [B, num_classes]

        loss_neg_per_proxy = torch.log(1 + torch.exp(neg_logits_masked).sum(dim=0))  # [num_classes]

        has_neg = neg_mask.sum(dim=0) > 0
        if has_neg.sum() == 0:
            loss_neg = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
        else:
            loss_neg = loss_neg_per_proxy[has_neg].mean()

        loss = loss_pos + loss_neg

        # Devuelve 0.0 como segundo elemento para compatibilidad con el monitor
        # de señales tempranas (mismo contrato que TripletLoss y CircleLoss)
        return loss, 0.0