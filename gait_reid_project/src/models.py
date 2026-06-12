# src/models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# Número de franjas HPP — cambia este valor para escalar
# 2 franjas → [upper, lower]   hpp_project input = 512*2*2 = 2048
# 4 franjas → resolución 2×    hpp_project input = 512*4*2 = 4096
HPP_PARTS = 2

class TemporalPooling(nn.Module):
    """
    GAP + GMP a lo largo de la dimensión temporal T.
    Entrada: (B, T, Features)
    Salida:  (B, Features * 2)
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        gap = torch.mean(x, dim=1)      # [B, F]
        gmp = torch.max(x, dim=1)[0]   # [B, F]
        return torch.cat([gap, gmp], dim=1)  # [B, F*2]


class GaitBackbone(nn.Module):
    """
    Backbone ResNet-18 modificado para volumetría temporal [B, T, C, H, W].
    HPP_PARTS franjas horizontales + Dual Pooling temporal (GAP+GMP).
    """
    def __init__(self, embed_dim=256):
        super().__init__()

        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Adaptar primera capa a 1 canal (siluetas en escala de grises)
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        with torch.no_grad():
            self.conv1.weight.data = resnet.conv1.weight.data.mean(dim=1, keepdim=True)

        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

        # HPP: divide la altura en HPP_PARTS franjas horizontales
        # Con HPP_PARTS=4: [B*T, 512, 4, 1] → 4 franjas independientes
        self.avgpool = nn.AdaptiveAvgPool2d((HPP_PARTS, 1))

        # Proyección HPP: 512 canales × HPP_PARTS franjas × 2 (GAP+GMP) → 1024
        hpp_input_dim = 512 * HPP_PARTS * 2   # 4096 con HPP_PARTS=4
        self.hpp_project = nn.Linear(hpp_input_dim, 1024)
        nn.init.normal_(self.hpp_project.weight.data, 0.0, 0.01)
        nn.init.constant_(self.hpp_project.bias.data, 0.0)

        self.temporal_pool = TemporalPooling()

        # Proyección SSL InfoNCE
        self.projection = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, embed_dim)
        )

    def spatial_forward(self, x_fp):
        """Extracción espacial 2D con HPP. Retorna [B*T, 512, HPP_PARTS]."""
        x = self.conv1(x_fp)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)                           # [B*T, 512, HPP_PARTS, 1]
        x = x.view(x.size(0), 512, HPP_PARTS)        # [B*T, 512, HPP_PARTS]
        return x

    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.size()

        # 1. Plegar dimensión temporal para procesar todos los frames en paralelo
        x_fold      = x.view(B * T, C, H, W)

        # 2. Extracción espacial con HPP
        features_2d = self.spatial_forward(x_fold)    # [B*T, 512, HPP_PARTS]

        # 3. Restaurar dimensión temporal
        features_3d = features_2d.view(B, T, 512, HPP_PARTS)  # [B, T, 512, HPP_PARTS]

        # 4. Pooling temporal dual (GAP+GMP) por cada franja HPP
        franja_features = []
        for p in range(HPP_PARTS):
            z_p = self.temporal_pool(features_3d[:, :, :, p])  # [B, 1024]
            franja_features.append(z_p)

        # 5. Concatenar todas las franjas
        z_hpp = torch.cat(franja_features, dim=1)     # [B, 1024 * HPP_PARTS]

        # 6. Reducir a 1024 dimensiones
        temporal_embedding = self.hpp_project(z_hpp)  # [B, 1024]

        # 7. Proyección SSL
        z = self.projection(temporal_embedding)
        z = F.normalize(z, dim=1)
        return z


class SupervisedReIDModel(nn.Module):
    """Modelo supervisado Re-ID que hereda la topología temporal del backbone."""
    def __init__(self, backbone, num_classes, freeze_backbone=False):
        super().__init__()

        self.spatial_encoder = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
            backbone.avgpool  # AdaptiveAvgPool2d((HPP_PARTS, 1))
        )
        self.temporal_pool = backbone.temporal_pool
        self.hpp_project   = backbone.hpp_project

        if freeze_backbone:
            for param in self.spatial_encoder.parameters(): param.requires_grad = False
            for param in self.temporal_pool.parameters():   param.requires_grad = False
            for param in self.hpp_project.parameters():     param.requires_grad = False
            print("  ✓ Backbone congelado")

        feature_dim = 1024

        self.bn = nn.BatchNorm1d(feature_dim)
        nn.init.normal_(self.bn.weight.data, 1.0, 0.02)
        nn.init.constant_(self.bn.bias.data, 0.0)

        self.classifier = nn.Linear(feature_dim, num_classes)
        nn.init.normal_(self.classifier.weight.data, 0.0, 0.001)
        nn.init.constant_(self.classifier.bias.data, 0.0)

    def _extract_embedding(self, x):
        """Extracción de embedding compartida entre forward y evaluación."""
        B, T, C, H, W = x.size()

        x_fold          = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(x_fold)          # [B*T, 512, HPP_PARTS, 1]
        spatial_features = spatial_features.view(B * T, 512, HPP_PARTS)

        temporal_features = spatial_features.view(B, T, 512, HPP_PARTS)

        franja_features = []
        for p in range(HPP_PARTS):
            z_p = self.temporal_pool(temporal_features[:, :, :, p])
            franja_features.append(z_p)

        z_hpp           = torch.cat(franja_features, dim=1)       # [B, 1024*HPP_PARTS]
        pooled_features = self.hpp_project(z_hpp)                 # [B, 1024]
        return pooled_features

    def forward(self, x):
        pooled_features = self._extract_embedding(x)
        embeddings      = self.bn(pooled_features)
        logits          = self.classifier(embeddings)
        return logits

    def load_state_dict(self, state_dict, strict=True):
        missing = [k for k in ["hpp_project.weight", "hpp_project.bias"]
                   if k not in state_dict]
        if missing:
            print(f"\n  [!] Claves HPP faltantes: {missing}. Cargando parcialmente.")
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)


class HybridGaitModel(nn.Module):
    """
    Controlador maestro que devuelve (logits, embeddings_L2) para
    CrossEntropyLoss + PAL/TripletLoss simultáneamente.
    """
    def __init__(self, supervised_model: SupervisedReIDModel):
        super().__init__()
        self.spatial_encoder = supervised_model.spatial_encoder
        self.temporal_pool   = supervised_model.temporal_pool
        self.hpp_project     = supervised_model.hpp_project
        self.bn              = supervised_model.bn
        self.classifier      = supervised_model.classifier

    def _extract_embedding(self, x):
        B, T, C, H, W = x.size()

        x_fold           = x.view(B * T, C, H, W)
        spatial_features = self.spatial_encoder(x_fold)           # [B*T, 512, HPP_PARTS, 1]
        spatial_features = spatial_features.view(B * T, 512, HPP_PARTS)

        temporal_features = spatial_features.view(B, T, 512, HPP_PARTS)

        franja_features = []
        for p in range(HPP_PARTS):
            z_p = self.temporal_pool(temporal_features[:, :, :, p])
            franja_features.append(z_p)

        z_hpp           = torch.cat(franja_features, dim=1)        # [B, 1024*HPP_PARTS]
        pooled_features = self.hpp_project(z_hpp)                  # [B, 1024]
        return pooled_features

    def forward(self, x):
        pooled_features      = self._extract_embedding(x)
        embeddings           = self.bn(pooled_features)
        logits               = self.classifier(embeddings)
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        return logits, normalized_embeddings

    def load_state_dict(self, state_dict, strict=True):
        missing = [k for k in ["hpp_project.weight", "hpp_project.bias"]
                   if k not in state_dict]
        if missing:
            print(f"\n  [!] Claves HPP faltantes en Hybrid: {missing}. Cargando parcialmente.")
            return super().load_state_dict(state_dict, strict=False)
        return super().load_state_dict(state_dict, strict=strict)