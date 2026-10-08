from ultralytics import YOLO
from pathlib import Path

# ---------------------------------------------------------
# CONFIGURACAO
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

MODELO = (
    BASE_DIR
    / "modelo"
    / "best_v2.pt"
)

# ---------------------------------------------------------
# CARREGAR MODELO
# ---------------------------------------------------------

print("Carregando IA do SIMR...")

model = YOLO(str(MODELO))

print("Modelo carregado com sucesso.")


# ---------------------------------------------------------
# FUNCAO DE DETECCAO
# ---------------------------------------------------------


def detectar_material(imagem):

    resultados = model.predict(
        source=imagem,
        conf=0.25,
        verbose=False
    )

    resultado = resultados[0]

    # Dimensoes originais da imagem
    altura, largura = resultado.orig_shape

    if len(resultado.boxes) == 0:

        return {
            "detectado": False,
            "classe": None,
            "confianca": 0.0,
            "bbox": None,
            "largura": largura,
            "altura": altura
        }

    # Mantem a deteccao de maior confianca
    melhor_box = max(
        resultado.boxes,
        key=lambda box: float(box.conf[0])
    )

    classe_id = int(melhor_box.cls[0])
    confianca = float(melhor_box.conf[0])
    classe = model.names[classe_id]

    # Coordenadas do retangulo:
    # x1, y1 = canto superior esquerdo
    # x2, y2 = canto inferior direito
    x1, y1, x2, y2 = [
        float(valor)
        for valor in melhor_box.xyxy[0].tolist()
    ]

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
# TESTE
# ---------------------------------------------------------

# ---------------------------------------------------------
# TESTE
# ---------------------------------------------------------

if __name__ == "__main__":

    imagem = BASE_DIR / "teste_real2.jpeg"

    resultado = detectar_material(str(imagem))

    print("\n===================================")
    print("       SIMR - RESULTADO IA")
    print("===================================")

    if resultado["detectado"]:

        print(
            f"Material: {resultado['classe']}"
        )

        print(
            f"Confianca: "
            f"{resultado['confianca'] * 100:.2f}%"
        )

    else:

        print("Nenhum material detectado.")

    print("===================================")