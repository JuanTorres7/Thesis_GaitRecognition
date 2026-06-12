# run_ablation.py
import os
import sys
import re
import time
import subprocess

def backup_settings():
    with open("configs/settings.py", "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

def restore_settings(content):
    with open("configs/settings.py", "w", encoding="utf-8") as f:
        f.write(content)

def update_settings(use_circle, use_rerank, checkpoint_path):
    with open("configs/settings.py", "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
    
    lines = content.splitlines()
    new_lines = []
    for line in lines:
        if line.strip().startswith("USE_CIRCLE_LOSS ="):
            new_lines.append(f"USE_CIRCLE_LOSS = {use_circle}")
        elif line.strip().startswith("USE_RERANKING ="):
            new_lines.append(f"USE_RERANKING = {use_rerank}")
        elif line.strip().startswith("HYBRID_CHECKPOINT ="):
            new_lines.append(f"HYBRID_CHECKPOINT = \"{checkpoint_path}\"")
        else:
            new_lines.append(line)
            
    with open("configs/settings.py", "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines) + "\n")

def run_experiment(exp_name, run_train, use_circle, use_rerank, checkpoint_path):
    print(f"\n==================================================")
    print(f" Ejecutando Experimento {exp_name}")
    print(f"   Circle Loss: {use_circle} | Re-Ranking: {use_rerank}")
    print(f"==================================================")
    
    # 1. Configurar settings.py
    update_settings(use_circle, use_rerank, checkpoint_path)
    
    # Carpeta de logs
    os.makedirs("logs_tesis", exist_ok=True)
    log_path = os.path.join("logs_tesis", f"experimento_{exp_name}.log")
    
    # 2. Entrenar si es necesario
    if run_train:
        # Verificar si ya existe para evitar re-entrenar de forma innecesaria
        if os.path.exists(checkpoint_path):
            print(f"[*] Checkpoint '{checkpoint_path}' ya existe. Omitiendo entrenamiento.")
        else:
            print(f"[*] Iniciando entrenamiento (train_hybrid.py)...")
            start = time.time()
            with open(log_path, "w", encoding="utf-8") as log_file:
                subprocess.run(
                    [sys.executable, "-u", "train_hybrid.py", "--no-eval"], 
                    stdout=log_file, 
                    stderr=subprocess.STDOUT,
                    check=True
                )
            print(f"[OK] Entrenamiento completado en {time.time() - start:.1f}s.")
            
    # 3. Evaluar
    print(f"[*] Iniciando evaluación (evaluate.py)...")
    start = time.time()
    # Si ya se creó log en entrenamiento, lo abrimos en modo append, sino en write
    mode = "a" if run_train and os.path.exists(log_path) else "w"
    with open(log_path, mode, encoding="utf-8") as log_file:
        subprocess.run(
            [sys.executable, "-u", "evaluate.py", "--phase", "hybrid"], 
            stdout=log_file, 
            stderr=subprocess.STDOUT,
            check=True
        )
    print(f"[OK] Evaluación completada en {time.time() - start:.1f}s.")

def parse_metrics(exp_name):
    log_path = os.path.join("logs_tesis", f"experimento_{exp_name}.log")
    if not os.path.exists(log_path):
        return 0.0, 0.0
        
    with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()
        
    # Buscar mAP del veredicto final (ignorar el baseline de 29.23%)
    map_match = re.search(r"Actual:?\s+mAP\s+([\d\.]+)", content)
    final_map = float(map_match.group(1)) if map_match else 0.0
    
    # Buscar Rank-1 de las condiciones de evaluación (ej. "> Promedio Condición | Rank-1: 99.2%")
    r1_matches = re.findall(r"Promedio Condici.*?Rank-1:\s+([\d\.]+)%", content)
    r1_scores = [float(val) for val in r1_matches]
    
    final_r1 = sum(r1_scores) / len(r1_scores) if r1_scores else 0.0
    
    return final_r1, final_map

def main():
    original_settings = backup_settings()
    
    experiments = [
        {"name": "A", "train": True,  "circle": False, "rerank": False, "checkpoint": "models/hybrid_model_final_PRO.pth", "loss_name": "Triplet"},
        {"name": "B", "train": True,  "circle": False, "rerank": True,  "checkpoint": "models/hybrid_model_final_PRO.pth", "loss_name": "Triplet"},
        {"name": "C", "train": True,  "circle": True,  "rerank": False, "checkpoint": "models/hybrid_model_final_Circle.pth", "loss_name": "Circle"},
        {"name": "D", "train": True,  "circle": True,  "rerank": True,  "checkpoint": "models/hybrid_model_final_Circle.pth", "loss_name": "Circle"},
    ]
    
    try:
        for exp in experiments:
            run_experiment(
                exp_name=exp["name"],
                run_train=exp["train"],
                use_circle=exp["circle"],
                use_rerank=exp["rerank"],
                checkpoint_path=exp["checkpoint"]
            )
            
        print("\n" + "="*50)
        print(" GENERANDO TABLA COMPARATIVA DE RESULTADOS")
        print("="*50)
        
        csv_path = "resumen_experimentos.csv"
        with open(csv_path, "w", encoding="utf-8") as csv_file:
            csv_file.write("Experimento,Perdida,ReRanking,Rank1,mAP\n")
            for exp in experiments:
                r1, map_val = parse_metrics(exp["name"])
                circle_str = exp["loss_name"]
                rerank_str = "Si" if exp["rerank"] else "No"
                csv_file.write(f"Experimento {exp['name']},{circle_str},{rerank_str},{r1:.2f}%,{map_val:.2f}%\n")
                print(f"Experimento {exp['name']} | Pérdida: {circle_str:8} | Re-Ranking: {rerank_str:3} | Rank-1: {r1:6.2f}% | mAP: {map_val:6.2f}%")
                
        print(f"\n[OK] Tabla de resultados guardada con éxito en: {csv_path}")
        
    finally:
        restore_settings(original_settings)
        print("[*] Archivo configs/settings.py restaurado a su estado original.")

if __name__ == "__main__":
    main()
