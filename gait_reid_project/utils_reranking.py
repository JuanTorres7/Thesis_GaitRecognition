import torch

def re_ranking(q_embeds, g_embeds, k1=20, k2=6, lambda_val=0.3):
    """
    K-Reciprocal Re-ranking optimizado completamente en PyTorch (ejecutado en GPU/CPU).
    q_embeds: [M, D] - Embeddings de Query
    g_embeds: [N, D] - Embeddings de Gallery
    """
    device = q_embeds.device
    M = q_embeds.size(0)
    N = g_embeds.size(0)
    L = M + N
    
    # 1. Concatenación y matriz de distancias Euclidiana original en GPU
    all_embeds = torch.cat([q_embeds, g_embeds], dim=0)
    original_dist = torch.cdist(all_embeds, all_embeds, p=2)
    original_dist = original_dist / original_dist.max() # Normalizar para evitar overflow
    
    # 2. Vecinos más cercanos iniciales
    initial_rank = torch.argsort(original_dist, dim=1)
    
    # 3. Máscara top-k1
    top_k1_mask = torch.zeros(L, L, device=device)
    top_k1_mask.scatter_(1, initial_rank[:, :k1], 1.0)
    
    # 4. K-reciprocal nearest neighbors
    k_reciprocal_mask = top_k1_mask * top_k1_mask.t()
    
    # 5. Expansión de Consultas Local Vectorizada
    V = torch.zeros(L, L, device=device)
    for i in range(L):
        candidates = torch.nonzero(k_reciprocal_mask[i]).squeeze(-1)
        if len(candidates) == 0:
            continue
        
        # Intersección y unión de vecinos de candidatos
        cand_neighbors = k_reciprocal_mask[candidates] # [num_candidates, L]
        intersection = cand_neighbors[:, candidates].sum(dim=1) # [num_candidates]
        total_neighbors = cand_neighbors.sum(dim=1) # [num_candidates]
        
        # Filtro de candidatos válidos para la expansión (> 2/3 de vecinos recíprocos comunes)
        valid_mask = intersection > (2.0 / 3.0 * total_neighbors)
        valid_candidates = candidates[valid_mask]
        
        if len(valid_candidates) > 0:
            V[i] = k_reciprocal_mask[valid_candidates].sum(dim=0).clamp(max=1.0)
        else:
            V[i] = k_reciprocal_mask[i]
            
    V = torch.clamp(V + k_reciprocal_mask, max=1.0)
    
    # 6. Ponderación por distancia exponencial
    V = V * torch.exp(-original_dist)
    
    # 7. Distancia Jaccard matricial en GPU
    intersection = torch.matmul(V, V.t())
    v_sum = V.sum(dim=1, keepdim=True)
    union = v_sum + v_sum.t() - intersection
    jaccard_dist = 1.0 - intersection / (union + 1e-9)
    
    # 8. Fusión de distancias
    final_dist = (1.0 - lambda_val) * jaccard_dist + lambda_val * original_dist
    
    # 9. Extraer la sección Query vs Gallery
    return final_dist[:M, M:]
