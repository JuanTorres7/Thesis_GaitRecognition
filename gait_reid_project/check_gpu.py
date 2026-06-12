# check_gpu.py
import torch
import sys

def verify_gpu():
    print("=" * 55)
    print(" VERIFICACIÓN DE HARDWARE BÁSICO PARA RE-ID VOLUMÉTRICO ")
    print("=" * 55)
    
    cuda_available = torch.cuda.is_available()
    print(f"[>] PyTorch versión: {torch.__version__}")
    print(f"[>] Python path: {sys.executable}")
    print(f"\n[>] CUDA Disponible: {'SÍ (Hardware Acelerado)' if cuda_available else 'NO (Atrapado en CPU)'}")
    
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"[>] Número de GPUs detectadas: {gpu_count}")
        for i in range(gpu_count):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
            properties = torch.cuda.get_device_properties(i)
            print(f"      VRAM Total: {properties.total_memory / 1e9:.2f} GB")
            print(f"      Compute Capability: {properties.major}.{properties.minor}")
            
        print(f"[>] Compilación CUDA de torch: {torch.version.cuda}")
        
        # Micro test de transferencia matricial 
        try:
            tensor = torch.zeros((100, 100), device='cuda')
            print("\n[✓] PRUEBA DE TRANSFERENCIA DE VRAM EXITOSA.")
            print("El Dataloader volumétrico PyTorch está 100% autorizado a despachar Tensores L2 [B,T,C,H,W] a la GPU.")
        except Exception as e:
            print(f"\n[X] Error crítico leyendo/escribiendo bloque VRAM: {e}")
    else:
        print("\n[!] ADVERTENCIA (CUELLO DE BOTELLA CRÍTICO)")
        print("CUDA ha devuelto 'False'. Significa que tu entorno virtual no ve ninguna tarjeta NVIDIA, ")
        print("o que la versión de pip-torch instalada es solo para CPU, ocasionando que la Batch Hard ")
        print("Triplet Loss vectorizada tome horas en lugar de minutos por cada Época.")

if __name__ == "__main__":
    verify_gpu()
