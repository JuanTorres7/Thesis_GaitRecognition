import torch
from src.models import GaitBackbone, SupervisedReIDModel
from src.samplers import RandomIdentitySampler
from configs import settings

def verify_model():
    print("Verifying Model Architecture...")
    try:
        backbone = GaitBackbone()
        model = SupervisedReIDModel(backbone, num_classes=10)
        x = torch.randn(2, 15, 1, 64, 64)
        logits = model(x)
        print(f"  [OK] Model forward pass successful. Output shape: {logits.shape}")
        if logits.shape == (2, 10):
            print("  [OK] Output shape is correct.")
        else:
            print(f"  [FAIL] Output shape mismatch. Expected (2, 10), got {logits.shape}")
    except Exception as e:
        print(f"  [FAIL] Model verification failed: {e}")

def verify_sampler():
    print("\nVerifying RandomIdentitySampler...")
    # Mock data: 10 identities, 10 images each
    data_source = []
    for pid in range(10):
        for i in range(10):
            data_source.append((f"img_{pid}_{i}.png", pid))
            
    class MockDataset:
        def __init__(self, data):
            self.samples = [{'label': pid, 'file': file} for file, pid in data]
            
    dataset = MockDataset(data_source)
    batch_size = 16
    num_instances = 4
    
    try:
        sampler = RandomIdentitySampler(dataset, batch_size, num_instances)
        print(f"  Data source size: {len(data_source)}")
        print(f"  Batch size: {batch_size}, Num instances: {num_instances}")
        
        iterator = iter(sampler)
        batch = []
        for _ in range(batch_size):
            batch.append(next(iterator))
            
        print(f"  Sampled batch indices: {batch}")
        
        # Check P x K property
        sampled_pids = [data_source[i][1] for i in batch]
        print(f"  Sampled PIDs: {sampled_pids}")
        
        from collections import Counter
        counts = Counter(sampled_pids)
        print(f"  PID Counts: {counts}")
        
        if all(c == num_instances for c in counts.values()):
            print("  [OK] Sampler produced correct P x K distribution.")
        else:
            print("  [FAIL] Sampler distribution incorrect.")
            
    except Exception as e:
        print(f"  [FAIL] Sampler verification failed: {e}")

if __name__ == "__main__":
    verify_model()
    verify_sampler()
