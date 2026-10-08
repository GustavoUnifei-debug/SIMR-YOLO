
import os
import time
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURACAO DE CPU
# ---------------------------------------------------------
# Configurar antes de importar torch/ultralytics.

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import torch
from ultralytics import YOLO


# ---------------------------------------------------------
# CONFIGURACAO DO PROJETO
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODELO = BASE_DIR / "modelo" / "best_v2.pt"

# Configuracoes de inferencia
TAMANHO_IMAGEM = 320
CONFIANCA_MINIMA = 0.25
DISPOSITIVO = "cpu"

# Limitar threads para evitar sobrecarga
torch.set_num_threads(1)


# ---------------------------------------------------------
# CARREGAR MODELO
# ---------------------------------------------------------

print("Carregando IA do SIMR...")

model = YOLO(str(MODELO))

print("Modelo carregado com sucesso.")
print(f"Dispositivo: {DISPOSITIVO}")
print(f"Resolucao de inferencia: {TAMANHO_IMAGEM}")
print(f"Threads CPU: {torch.get_num_threads()}")


# ---------------------------------------------------------
# FUNCAO DE DETECCAO
# ---------------------------------------------------------

def detectar_material(imagem):

    inicio = time.perf_counter()

    # Executar YOLOv8 V2
    resultados = model.predict(
        source=str(imagem),
        imgsz=TAMANHO_IMAGEM,
        conf=CONFIANCA_MINIMA,
        device=DISPOSITIVO,
        verbose=False
    )

    resultado = resultados[0]

    # Dimensoes originais da imagem
    altura, largura = resultado.orig_shape

    # Tempo de processamento
    tempo_ia = time.perf_counter() - inicio

    print(
        f"[SIMR IA] Inferencia={tempo_ia:.3f}s",
        flush=True
    )

    # -----------------------------------------------------
    # NENHUMA DETECCAO
    # -----------------------------------------------------

    if len(resultado.boxes) == 0:

        return {
            "detectado": False,
            "classe": None,
            "confianca": 0.0,
            "bbox": None,
            "largura": largura,
            "altura": altura
        }

    # -----------------------------------------------------
    # SELECIONAR MELHOR DETECCAO
    # -----------------------------------------------------

    melhor_box = max(
        resultado.boxes,
        key=lambda box: float(box.conf[0])
    )

    classe_id = int(melhor_box.cls[0])
    confianca = float(melhor_box.conf[0])
    classe = model.names[classe_id]

    # -----------------------------------------------------
    # COORDENADAS DO OBJETO
    # -----------------------------------------------------
    # Coordenadas na resolucao ORIGINAL da imagem.

    x1, y1, x2, y2 = [
        float(valor)
        for valor in melhor_box.xyxy[0].tolist()
    ]

    # -----------------------------------------------------
    # RESULTADO
    # -----------------------------------------------------

    return {
        "detectado": True,
        "classe": classe,
        "confianca": confianca,
        "bbox": [
            round(x1, 2),
            round(y1, 2),
            round(x2, 2),
            round(y2, 2)
        ],
        "largura": largura,
        "altura": altura
    }


# ---------------------------------------------------------
# TESTE LOCAL
# ---------------------------------------------------------

if __name__ == "__main__":

    imagem = BASE_DIR / "teste_real2.jpeg"

    resultado = detectar_material(str(imagem))

    print("\n===================================")
    print("       SIMR - RESULTADO IA")
    print("===================================")

    if resultado["detectado"]:

        print(f"Material: {resultado['classe']}")

        print(
            f"Confianca: "
            f"{resultado['confianca'] * 100:.2f}%"
        )

        print(f"Coordenadas: {resultado['bbox']}")

    else:

        print("Nenhum material detectado.")

    print("===================================")
