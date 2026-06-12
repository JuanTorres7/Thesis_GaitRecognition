# src/samplers.py
import torch
import numpy as np
from torch.utils.data.sampler import Sampler
from collections import defaultdict

class RandomIdentitySampler(Sampler):
    """
    P-K Sampler estricto: Muestrea aleatoriamente P identidades (clases), y para cada una,
    extrae K instancias volumétricas (secuencias). El tamaño del Batch es P * K.
    """
    def __init__(self, dataset, batch_size, num_instances):
        self.dataset = dataset
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_pids_per_batch = self.batch_size // self.num_instances
        
        # Generar diccionario de índices {label: [idx1, idx2, ...]}
        self.index_dic = defaultdict(list)
        
        # Verificamos si el dataset tiene 'samples' accesible
        if hasattr(self.dataset, 'samples'):
            for index, item in enumerate(self.dataset.samples):
                # En CASIAB_Supervised cada item es conf: {'label': X, 'frames': [...], ...}
                pid = item['label']
                self.index_dic[pid].append(index)
        else:
            raise ValueError("El Dataloader Volumétrico no tiene el atributo .samples público.")
            
        self.pids = list(self.index_dic.keys())

        # Estimar tamaño artificial de un epoch (balanceado)
        self.length = 0
        for pid in self.pids:
            idxs = self.index_dic[pid]
            num = len(idxs)
            if num < self.num_instances:
                num = self.num_instances
            self.length += num - num % self.num_instances

    def __iter__(self):
        batch_idxs_dict = defaultdict(list)

        for pid in self.pids:
            idxs = self.index_dic[pid]
            
            # Sampling con Re-Acomodo (Bootstrap) si una ID tiene menos secuencias de las requeridas (K)
            if len(idxs) < self.num_instances:
                idxs = np.random.choice(idxs, size=self.num_instances, replace=True)
            
            np.random.shuffle(idxs)
            batch_idxs = []
            
            # Agrupar en pequeños bloques de K secuencias para esta Identidad
            for idx in idxs:
                batch_idxs.append(idx)
                if len(batch_idxs) == self.num_instances:
                    batch_idxs_dict[pid].append(batch_idxs)
                    batch_idxs = []

        avai_pids = list(self.pids)
        final_idxs = []

        # Construir bloques P x K
        while len(avai_pids) >= self.num_pids_per_batch:
            # Seleccionar P identidades
            selected_pids = np.random.choice(avai_pids, self.num_pids_per_batch, replace=False)
            
            for pid in selected_pids:
                # Tomar 1 bloque de K instancias para esta identidad
                batch_idxs = batch_idxs_dict[pid].pop(0)
                final_idxs.extend(batch_idxs)
                
                # Deshabilitar PID si se le terminan los repuestos
                if len(batch_idxs_dict[pid]) == 0:
                    avai_pids.remove(pid)

        return iter(final_idxs)

    def __len__(self):
        return self.length
